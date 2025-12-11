#!/usr/bin/env python3
"""
TTS Streaming Client - Real-time audio playback from API server
===============================================================

This module provides HTTP and WebSocket clients for receiving streamed
audio from the TTS API server and playing it in real-time.

Features:
    - HTTP streaming client: POST request with chunked PCM response
    - WebSocket client: Bidirectional real-time communication
    - Low-latency audio playback using sounddevice
    - Thread-safe audio queue management

Usage Examples:
    # HTTP streaming
    python -m scripts.cv2.server.client http \
        --server http://localhost:8000 \
        --prompt-wav prompts/wav/sample.wav \
        --prompt-text "こんにちは" \
        --text "今日はいい天気ですね。"

    # WebSocket streaming
    python -m scripts.cv2.server.client ws \
        --server ws://localhost:8000 \
        --prompt-wav prompts/wav/sample.wav \
        --prompt-text "こんにちは" \
        --text "今日はいい天気ですね。"

Requirements:
    pip install httpx websockets sounddevice numpy torchaudio
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
from logging import INFO, StreamHandler, getLogger
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
import sounddevice as sd

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False


# Audio constants (matching server output)
OUTPUT_SAMPLE_RATE = 22050
INPUT_SAMPLE_RATE = 16000
BIT_DEPTH = 16
CHANNELS = 1


class StreamingPlayer:
    """Real-time audio player that plays PCM chunks as they arrive.

    Uses a queue-based approach with a separate playback thread to ensure
    smooth, low-latency audio playback even when chunks arrive at irregular
    intervals.

    Attributes:
        sample_rate: Output audio sample rate in Hz
        channels: Number of audio channels (1 for mono)

    Example:
        >>> player = StreamingPlayer(sample_rate=22050)
        >>> player.start()
        >>> player.play_chunk(audio_data)  # numpy float32 array
        >>> player.stop()
    """

    def __init__(self, sample_rate: int = OUTPUT_SAMPLE_RATE, channels: int = CHANNELS):
        """Initialize the streaming player.

        Args:
            sample_rate: Output sample rate in Hz (default: 22050)
            channels: Number of audio channels (default: 1)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self.stream: sd.OutputStream | None = None
        self.is_playing = False
        self._playback_thread: threading.Thread | None = None

    def _playback_worker(self) -> None:
        """Worker thread that continuously plays audio chunks from the queue."""
        while self.is_playing:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is None:
                    break
                if self.stream is not None:
                    self.stream.write(chunk)
                self.audio_queue.task_done()
            except queue.Empty:
                continue

    def start(self) -> None:
        """Start the audio output stream and playback thread."""
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
        )
        self.stream.start()
        self.is_playing = True
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()
        logger.info(f"Audio player started (sample_rate={self.sample_rate}Hz)")

    def play_chunk(self, audio: np.ndarray) -> None:
        """Add an audio chunk to the playback queue.

        Args:
            audio: Audio data as numpy float32 array with values in [-1, 1]
        """
        self.audio_queue.put(audio)

    def play_pcm_bytes(self, pcm_bytes: bytes) -> None:
        """Convert PCM bytes to float32 and add to playback queue.

        Args:
            pcm_bytes: Raw PCM audio data (int16 format)
        """
        # Convert int16 PCM to float32
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        self.play_chunk(audio_float32)

    def stop(self) -> None:
        """Stop playback and close the audio stream."""
        self.audio_queue.put(None)
        self.is_playing = False

        if self._playback_thread is not None:
            self._playback_thread.join(timeout=2.0)

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        logger.info("Audio player stopped")

    def wait_until_done(self) -> None:
        """Block until all queued audio has been played."""
        self.audio_queue.join()


def load_prompt_audio(path: Path) -> bytes:
    """Load audio file and convert to 16kHz mono int16 PCM bytes.

    Args:
        path: Path to the audio file (WAV, MP3, etc.)

    Returns:
        Raw PCM bytes (16kHz, mono, int16)

    Raises:
        FileNotFoundError: If the audio file doesn't exist
    """
    import torchaudio

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Load audio
    waveform, sample_rate = torchaudio.load(str(path))

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != INPUT_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, INPUT_SAMPLE_RATE)

    # Convert to int16 PCM bytes
    pcm_array = (waveform.squeeze().numpy() * 32767).astype(np.int16)
    return pcm_array.tobytes()


async def http_stream_tts(
    server_url: str,
    prompt_wav_path: Path,
    prompt_text: str,
    tts_text: str,
    speed: float = 1.0,
) -> AsyncIterator[bytes]:
    """Stream TTS audio from the HTTP endpoint.

    Sends a POST request to the TTS endpoint with streaming enabled
    and yields PCM audio chunks as they arrive.

    Args:
        server_url: Base URL of the TTS server (e.g., "http://localhost:8000")
        prompt_wav_path: Path to the prompt audio file
        prompt_text: Transcription of the prompt audio
        tts_text: Text to synthesize
        speed: Speech speed multiplier (0.5-2.0)

    Yields:
        bytes: PCM audio chunks (22050Hz, mono, int16)

    Raises:
        httpx.HTTPError: If the HTTP request fails

    Example:
        >>> async for chunk in http_stream_tts(
        ...     "http://localhost:8000",
        ...     Path("prompt.wav"),
        ...     "Hello",
        ...     "How are you today?"
        ... ):
        ...     process_audio(chunk)
    """
    import httpx

    url = f"{server_url.rstrip('/')}/v1/tts/"

    # Read prompt audio file
    with open(prompt_wav_path, "rb") as f:
        prompt_audio_data = f.read()

    # Prepare multipart form data
    files = {
        "prompt_wav": (prompt_wav_path.name, prompt_audio_data, "audio/wav"),
    }
    data = {
        "tts_text": tts_text,
        "prompt_text": prompt_text,
    }
    params = {
        "speed": speed,
        "stream": "true",
        "format": "pcm",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        async with client.stream(
            "POST",
            url,
            files=files,
            data=data,
            params=params,
        ) as response:
            response.raise_for_status()

            # Log audio format info from headers
            sample_rate = response.headers.get("X-Sample-Rate", OUTPUT_SAMPLE_RATE)
            logger.info(f"Receiving audio stream (sample_rate={sample_rate}Hz)")

            # Stream chunks
            async for chunk in response.aiter_bytes(chunk_size=4096):
                if chunk:
                    yield chunk


async def websocket_stream_tts(
    server_url: str,
    prompt_wav_path: Path,
    prompt_text: str,
    tts_text: str,
    speed: float = 1.0,
) -> AsyncIterator[bytes]:
    """Stream TTS audio via WebSocket connection.

    Establishes a WebSocket connection, sends the prompt audio and
    configuration, then yields PCM audio chunks as they arrive.

    Args:
        server_url: WebSocket URL (e.g., "ws://localhost:8000" or "http://localhost:8000")
        prompt_wav_path: Path to the prompt audio file
        prompt_text: Transcription of the prompt audio
        tts_text: Text to synthesize
        speed: Speech speed multiplier (0.5-2.0)

    Yields:
        bytes: PCM audio chunks (22050Hz, mono, int16)

    Example:
        >>> async for chunk in websocket_stream_tts(
        ...     "ws://localhost:8000",
        ...     Path("prompt.wav"),
        ...     "Hello",
        ...     "How are you today?"
        ... ):
        ...     process_audio(chunk)
    """
    import websockets

    # Convert http:// to ws://
    ws_url = server_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url.rstrip('/')}/v1/tts/ws"

    # Load and convert prompt audio to PCM
    prompt_pcm = load_prompt_audio(prompt_wav_path)

    async with websockets.connect(ws_url) as websocket:
        logger.info(f"Connected to WebSocket: {ws_url}")

        # 1. Send prompt audio (binary)
        await websocket.send(prompt_pcm)
        response = await websocket.recv()
        msg = json.loads(response)
        if "error" in msg:
            raise RuntimeError(f"Server error: {msg['error']}")
        logger.info(f"Prompt received: {msg}")

        # 2. Send configuration (JSON)
        config = {"prompt_text": prompt_text, "speed": speed}
        await websocket.send(json.dumps(config))
        response = await websocket.recv()
        msg = json.loads(response)
        if "error" in msg:
            raise RuntimeError(f"Server error: {msg['error']}")
        logger.info(f"Config updated: {msg}")

        # 3. Send text to synthesize (JSON)
        await websocket.send(json.dumps({"text": tts_text}))

        # 4. Receive audio chunks
        while True:
            data = await websocket.recv()

            if isinstance(data, bytes):
                # Audio chunk
                yield data
            else:
                # JSON message
                msg = json.loads(data)
                if msg.get("status") == "done":
                    logger.info("Synthesis complete")
                    break
                elif "error" in msg:
                    raise RuntimeError(f"Server error: {msg['error']}")


async def run_http_client(
    server_url: str,
    prompt_wav_path: Path,
    prompt_text: str,
    tts_text: str,
    speed: float = 1.0,
    save_path: Optional[Path] = None,
) -> None:
    """Run the HTTP streaming client with real-time playback.

    Args:
        server_url: Base URL of the TTS server
        prompt_wav_path: Path to the prompt audio file
        prompt_text: Transcription of the prompt audio
        tts_text: Text to synthesize
        speed: Speech speed multiplier
        save_path: Optional path to save the output audio
    """
    player = StreamingPlayer()
    player.start()

    all_audio: list[bytes] = []
    total_bytes = 0
    chunk_count = 0

    try:
        async for chunk in http_stream_tts(
            server_url, prompt_wav_path, prompt_text, tts_text, speed
        ):
            chunk_count += 1
            total_bytes += len(chunk)
            player.play_pcm_bytes(chunk)

            if save_path:
                all_audio.append(chunk)

            # Log progress
            duration_sec = total_bytes / (OUTPUT_SAMPLE_RATE * 2)  # 2 bytes per sample
            logger.info(f"  Chunk {chunk_count}: {len(chunk)} bytes (total: {duration_sec:.2f}s)")

        player.wait_until_done()

    finally:
        player.stop()

    # Save audio if requested
    if save_path and all_audio:
        _save_audio(b"".join(all_audio), save_path)


async def run_websocket_client(
    server_url: str,
    prompt_wav_path: Path,
    prompt_text: str,
    tts_text: str,
    speed: float = 1.0,
    save_path: Optional[Path] = None,
) -> None:
    """Run the WebSocket streaming client with real-time playback.

    Args:
        server_url: WebSocket URL of the TTS server
        prompt_wav_path: Path to the prompt audio file
        prompt_text: Transcription of the prompt audio
        tts_text: Text to synthesize
        speed: Speech speed multiplier
        save_path: Optional path to save the output audio
    """
    player = StreamingPlayer()
    player.start()

    all_audio: list[bytes] = []
    total_bytes = 0
    chunk_count = 0

    try:
        async for chunk in websocket_stream_tts(
            server_url, prompt_wav_path, prompt_text, tts_text, speed
        ):
            chunk_count += 1
            total_bytes += len(chunk)
            player.play_pcm_bytes(chunk)

            if save_path:
                all_audio.append(chunk)

            # Log progress
            duration_sec = total_bytes / (OUTPUT_SAMPLE_RATE * 2)
            logger.info(f"  Chunk {chunk_count}: {len(chunk)} bytes (total: {duration_sec:.2f}s)")

        player.wait_until_done()

    finally:
        player.stop()

    # Save audio if requested
    if save_path and all_audio:
        _save_audio(b"".join(all_audio), save_path)


def _save_audio(pcm_data: bytes, path: Path) -> None:
    """Save PCM audio data to a WAV file.

    Args:
        pcm_data: Raw PCM bytes (int16 format)
        path: Output file path
    """
    import torchaudio
    import torch

    # Convert PCM bytes to tensor
    audio_array = np.frombuffer(pcm_data, dtype=np.int16)
    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float() / 32768.0

    # Save as WAV
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(
        str(path),
        audio_tensor,
        OUTPUT_SAMPLE_RATE,
        format="wav",
        encoding="PCM_S",
        bits_per_sample=16,
    )
    logger.info(f"Audio saved to: {path}")


def main() -> None:
    """Main entry point for the streaming client CLI."""
    parser = argparse.ArgumentParser(
        description="TTS Streaming Client - Real-time audio playback from API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # HTTP streaming
  python -m scripts.cv2.server.client http \\
      --server http://localhost:8000 \\
      --prompt-wav prompts/wav/sample.wav \\
      --prompt-text "こんにちは" \\
      --text "今日はいい天気ですね。"

  # WebSocket streaming
  python -m scripts.cv2.server.client ws \\
      --server ws://localhost:8000 \\
      --prompt-wav prompts/wav/sample.wav \\
      --prompt-text "こんにちは" \\
      --text "今日はいい天気ですね。"
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Client mode")

    # Common arguments
    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--server",
            type=str,
            default="http://localhost:8000",
            help="TTS server URL (default: http://localhost:8000)",
        )
        p.add_argument(
            "--prompt-wav",
            type=Path,
            required=True,
            help="Path to prompt audio file",
        )
        p.add_argument(
            "--prompt-text",
            type=str,
            required=True,
            help="Transcription of the prompt audio",
        )
        p.add_argument(
            "--text",
            type=str,
            required=True,
            help="Text to synthesize",
        )
        p.add_argument(
            "--speed",
            type=float,
            default=1.0,
            help="Speech speed multiplier (0.5-2.0, default: 1.0)",
        )
        p.add_argument(
            "--save",
            type=Path,
            default=None,
            help="Save output audio to this path (optional)",
        )

    # HTTP subcommand
    http_parser = subparsers.add_parser("http", help="Use HTTP streaming")
    add_common_args(http_parser)

    # WebSocket subcommand
    ws_parser = subparsers.add_parser("ws", help="Use WebSocket streaming")
    add_common_args(ws_parser)

    args = parser.parse_args()

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Server: {args.server}")
    logger.info(f"Prompt WAV: {args.prompt_wav}")
    logger.info(f"Prompt text: {args.prompt_text}")
    logger.info(f"TTS text: {args.text}")
    logger.info(f"Speed: {args.speed}")

    if args.mode == "http":
        asyncio.run(
            run_http_client(
                server_url=args.server,
                prompt_wav_path=args.prompt_wav,
                prompt_text=args.prompt_text,
                tts_text=args.text,
                speed=args.speed,
                save_path=args.save,
            )
        )
    elif args.mode == "ws":
        asyncio.run(
            run_websocket_client(
                server_url=args.server,
                prompt_wav_path=args.prompt_wav,
                prompt_text=args.prompt_text,
                tts_text=args.text,
                speed=args.speed,
                save_path=args.save,
            )
        )


if __name__ == "__main__":
    main()
