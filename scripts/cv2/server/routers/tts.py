"""
Text-to-Speech synthesis endpoints (HTTP and WebSocket).

This module provides both REST API and WebSocket interfaces for
synthesizing speech using the CosyVoice2 model with LoRA adapters.
"""

import asyncio
import io
import json
import logging
import struct
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional, Union

import numpy as np
import torch
import torchaudio
from scipy import signal
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from scripts.cv2.server.routers.health import get_model_state

# Logger
logger = logging.getLogger(__name__)

# Constants
PROMPT_SAMPLE_RATE = 16000  # Input prompt sample rate
OUTPUT_SAMPLE_RATE = 22050  # CosyVoice2 output sample rate
MAX_CONCURRENT_REQUESTS = 2  # Default, can be overridden

# Concurrency control (initialized with default, can be updated)
_tts_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Text segmentation settings
# Keep segments short to maintain seq_len < 600 for optimal RTF
MAX_SEGMENT_CHARS = 1000  # Avoid splitting text with PHON tags


def segment_text(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split text into segments at sentence boundaries for optimal streaming RTF.

    CosyVoice2's streaming mode uses cumulative context, causing seq_len to grow
    with each chunk. By splitting text into shorter segments, we reset seq_len
    and maintain RTF < 1 throughout the entire synthesis.

    PHON tags (<PHON_START>...<PHON_END>) are preserved and never split.

    Args:
        text: Input text to segment
        max_chars: Maximum characters per segment (default: 1000)

    Returns:
        List of text segments

    Example:
        >>> segment_text("こんにちは。お元気ですか？はい、元気です。")
        ["こんにちは。", "お元気ですか？", "はい、元気です。"]
    """
    if not text:
        return []

    # Primary delimiters (sentence endings) - always split here
    primary_delimiters = "。！？!?"

    segments = []
    current_segment = ""
    in_phon_tag = False
    i = 0

    while i < len(text):
        # Check for PHON_START tag
        if text[i:].startswith("<PHON_START>"):
            in_phon_tag = True
            current_segment += "<PHON_START>"
            i += len("<PHON_START>")
            continue

        # Check for PHON_END tag
        if text[i:].startswith("<PHON_END>"):
            in_phon_tag = False
            current_segment += "<PHON_END>"
            i += len("<PHON_END>")
            continue

        char = text[i]
        current_segment += char

        # Split at sentence endings, but not inside PHON tags
        if not in_phon_tag and char in primary_delimiters:
            segment = current_segment.strip()
            if segment:
                segments.append(segment)
            current_segment = ""

        # Safety: force split if segment gets too long (but not inside PHON tags)
        if not in_phon_tag and len(current_segment) >= max_chars:
            segment = current_segment.strip()
            if segment:
                segments.append(segment)
            current_segment = ""

        i += 1

    # Add remaining text
    if current_segment.strip():
        segments.append(current_segment.strip())

    return segments


def set_max_concurrent_requests(max_requests: int) -> None:
    """Update the concurrency semaphore limit.

    Should be called during application startup with config value.

    Args:
        max_requests: Maximum number of concurrent TTS requests
    """
    global _tts_semaphore
    _tts_semaphore = asyncio.Semaphore(max_requests)


# Request/Response Models
class TTSRequest(BaseModel):
    """Request model for TTS synthesis.

    Attributes:
        tts_text: Text to be synthesized into speech
        prompt_text: Transcription of the prompt audio
        speed: Speech speed multiplier (0.5 = slower, 2.0 = faster)
        stream: Whether to return streaming audio chunks
        format: Output audio format ("wav" or "pcm")
    """

    tts_text: str = Field(..., description="Text to synthesize", min_length=1)
    prompt_text: str = Field(..., description="Prompt audio transcription", min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    stream: bool = Field(default=False, description="Enable streaming response")
    format: str = Field(default="wav", description="Output format: 'wav' or 'pcm'")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate audio format is either 'wav' or 'pcm'.

        Args:
            v: Format string to validate

        Returns:
            str: Validated format string (lowercase)

        Raises:
            ValueError: If format is not 'wav' or 'pcm'
        """
        v = v.lower()
        if v not in ["wav", "pcm"]:
            raise ValueError("format must be 'wav' or 'pcm'")
        return v


class WebSocketMessage(BaseModel):
    """WebSocket message model for configuration updates.

    Attributes:
        prompt_text: Optional prompt text update
        text: Optional TTS text to synthesize
        speed: Optional speech speed update
    """

    prompt_text: Optional[str] = Field(None, description="Prompt transcription")
    text: Optional[str] = Field(None, description="Text to synthesize")
    speed: Optional[float] = Field(None, ge=0.5, le=2.0, description="Speech speed")


# Router instance
router = APIRouter(
    prefix="/v1/tts",
    tags=["tts"],
    responses={503: {"description": "Model not available"}},
)


# Helper Functions
async def load_and_preprocess_audio(
    audio_file: Union[UploadFile, bytes],
    target_sr: int = PROMPT_SAMPLE_RATE,
) -> torch.Tensor:
    """Load and preprocess audio file to the target sample rate.

    This function handles audio format conversion, resampling, and stereo-to-mono
    conversion. It supports various audio formats through torchaudio.

    Args:
        audio_file: Audio file (UploadFile from FastAPI or raw bytes)
        target_sr: Target sample rate in Hz (default: 16000)

    Returns:
        torch.Tensor: Preprocessed audio tensor with shape [1, num_samples]

    Raises:
        HTTPException: If audio loading or processing fails

    Example:
        >>> audio_tensor = await load_and_preprocess_audio(uploaded_file)
        >>> print(audio_tensor.shape)
        torch.Size([1, 48000])
    """
    try:
        # Read audio data from UploadFile or use raw bytes
        # Use duck typing for robustness (hasattr check instead of isinstance)
        if hasattr(audio_file, 'read'):
            # UploadFile or file-like object
            audio_data = await audio_file.read()
        elif isinstance(audio_file, bytes):
            audio_data = audio_file
        else:
            raise ValueError(f"Unsupported audio_file type: {type(audio_file)}")

        # Load audio using torchaudio
        audio_buffer = io.BytesIO(audio_data)
        waveform, sample_rate = await run_in_threadpool(
            lambda: torchaudio.load(audio_buffer)
        )

        # Convert stereo to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != target_sr:
            waveform = await run_in_threadpool(
                torchaudio.functional.resample,
                waveform,
                sample_rate,
                target_sr,
            )

        return waveform

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load or preprocess audio: {str(e)}",
        )


def trim_audio_vad(
    waveform: torch.Tensor,
    sample_rate: int,
    trigger_level: float = 7.0,
) -> torch.Tensor:
    """Trim silence from audio using Voice Activity Detection (VAD).

    Removes silence from both the beginning and end of the audio signal
    using torchaudio's VAD functionality.

    Args:
        waveform: Audio tensor with shape [channels, num_samples]
        sample_rate: Sample rate of the audio in Hz
        trigger_level: VAD sensitivity threshold (higher = more aggressive trimming)

    Returns:
        torch.Tensor: Trimmed audio tensor

    Example:
        >>> trimmed = trim_audio_vad(audio, 16000, trigger_level=7.0)
        >>> print(f"Trimmed from {audio.shape[-1]} to {trimmed.shape[-1]} samples")
    """
    # Trim beginning
    trimmed = torchaudio.functional.vad(waveform, sample_rate, trigger_level=trigger_level)

    # Trim end (reverse, VAD, reverse back)
    if trimmed.shape[-1] > 0:
        trimmed_rev = torchaudio.functional.vad(
            trimmed.flip(-1), sample_rate, trigger_level=trigger_level
        )
        trimmed = trimmed_rev.flip(-1)

    return trimmed


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = OUTPUT_SAMPLE_RATE) -> bytes:
    """Convert raw PCM audio data to WAV format.

    Args:
        pcm_data: Raw PCM audio bytes (int16 format)
        sample_rate: Sample rate in Hz

    Returns:
        bytes: WAV file data

    Example:
        >>> pcm_bytes = np.array([0, 1000, -1000], dtype=np.int16).tobytes()
        >>> wav_bytes = pcm_to_wav_bytes(pcm_bytes, 16000)
        >>> len(wav_bytes)  # WAV header + data
        50
    """
    # Convert PCM bytes to numpy array
    audio_array = np.frombuffer(pcm_data, dtype=np.int16)

    # Convert to torch tensor with shape [1, num_samples]
    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float() / 32768.0

    # Save to WAV format in memory
    wav_buffer = io.BytesIO()
    torchaudio.save(
        wav_buffer,
        audio_tensor,
        sample_rate,
        format="wav",
        encoding="PCM_S",
        bits_per_sample=16,
    )
    wav_buffer.seek(0)

    return wav_buffer.read()


class AsyncStreamingPipeline:
    """Pipeline for decoupling inference from WebSocket sending.

    This class separates the inference thread from the async send loop,
    allowing them to run independently. This improves RTF by:
    - Eliminating per-chunk threadpool overhead
    - Batching multiple chunks before sending
    - Providing backpressure via bounded queue

    Attributes:
        websocket: WebSocket connection to send audio to
        loop: Event loop for cross-thread communication
        queue: Async queue for chunk buffering
        batch_min_samples: Minimum samples before flushing buffer
        batch_max_wait_sec: Maximum time before flushing buffer
    """

    def __init__(
        self,
        websocket: WebSocket,
        loop: asyncio.AbstractEventLoop,
        queue_maxsize: int = 10,
        batch_min_samples: int = 22050 // 4,  # 250ms worth
        batch_max_wait_sec: float = 0.05,  # 50ms
        speed: float = 1.0,
    ):
        """Initialize the streaming pipeline.

        Args:
            websocket: WebSocket connection for sending audio
            loop: Event loop from the calling async context
            queue_maxsize: Max chunks in queue (backpressure)
            batch_min_samples: Min samples before sending batch
            batch_max_wait_sec: Max wait time before sending batch
            speed: Speech speed multiplier (0.5-2.0). Applied via resampling.
        """
        self.websocket = websocket
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self.batch_min_samples = batch_min_samples
        self.batch_max_wait_sec = batch_max_wait_sec
        self.speed = speed
        self._inference_done = asyncio.Event()
        self._send_task: Optional[asyncio.Task] = None
        self._error: Optional[Exception] = None
        self._cancelled = False

    async def start_sender(self):
        """Start the background send task."""
        self._send_task = asyncio.create_task(self._send_loop())

    async def _send_loop(self):
        """Async send loop - batches and sends audio chunks."""
        buffer: list[torch.Tensor] = []
        buffer_samples = 0
        last_flush_time = time.monotonic()

        try:
            while not self._cancelled:
                try:
                    chunk = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=0.05  # 50ms check interval
                    )
                except asyncio.TimeoutError:
                    # On timeout, flush if buffer exists and time elapsed
                    elapsed = time.monotonic() - last_flush_time
                    if buffer and (elapsed >= self.batch_max_wait_sec or self._inference_done.is_set()):
                        await self._flush_buffer(buffer)
                        buffer = []
                        buffer_samples = 0
                        last_flush_time = time.monotonic()
                    # Check for cancelled or done after flush
                    if self._cancelled:
                        break
                    if self._inference_done.is_set() and self.queue.empty():
                        break
                    continue

                if chunk is None:
                    # Sentinel received - flush remaining and exit
                    if buffer:
                        await self._flush_buffer(buffer)
                    break

                buffer.append(chunk)
                buffer_samples += chunk.shape[-1]

                # Flush on size or time threshold
                elapsed = time.monotonic() - last_flush_time
                if buffer_samples >= self.batch_min_samples or elapsed >= self.batch_max_wait_sec:
                    await self._flush_buffer(buffer)
                    buffer = []
                    buffer_samples = 0
                    last_flush_time = time.monotonic()

        except Exception as e:
            self._error = e
            # Exit loop on exception - break is implicit here

    async def _flush_buffer(self, buffer: list[torch.Tensor]):
        """Concatenate and send buffered chunks."""
        if not buffer:
            return

        # Concatenate chunks (already on CPU)
        combined = torch.cat(buffer, dim=-1)

        # Handle mono audio (channel dim check)
        if combined.dim() == 2 and combined.size(0) == 1:
            combined = combined[0]

        # Clamp to [-1, 1] to prevent overflow noise
        combined = combined.clamp(-1.0, 1.0)

        audio_data = combined.numpy()

        # Apply speed change via resampling if speed != 1.0
        if self.speed != 1.0 and len(audio_data) > 0:
            # To speed up by factor `speed`, resample to fewer samples
            # new_length = original_length / speed
            new_length = int(len(audio_data) / self.speed)
            if new_length > 0:
                audio_data = signal.resample(audio_data, new_length)

        pcm_array = (audio_data * 32767).astype(np.int16)

        try:
            await self.websocket.send_bytes(pcm_array.tobytes())
        except Exception as e:
            # Record send failure (connection closed, etc.)
            self._error = e
            self._cancelled = True
            # Raise to exit send loop immediately
            raise

    def put_chunk_sync(self, chunk: torch.Tensor):
        """Put a chunk from the inference thread (sync method).

        Args:
            chunk: Audio tensor chunk from model inference

        Raises:
            RuntimeError: If pipeline not started
        """
        if self._send_task is None:
            raise RuntimeError("Pipeline not started. Call start_sender() first.")

        if self._cancelled:
            return

        future = asyncio.run_coroutine_threadsafe(
            self.queue.put(chunk), self.loop
        )
        try:
            future.result(timeout=2.0)  # 2s backpressure timeout
        except TimeoutError:
            # Backpressure - enqueue is delayed, not dropped
            logger.warning("Pipeline backpressure: enqueue delayed")

    def mark_done(self):
        """Signal inference completion from the inference thread."""
        # Set flag first (race condition prevention)
        self._inference_done.set()

        # Put sentinel (ignore if queue full)
        try:
            self.loop.call_soon_threadsafe(
                lambda: self.queue.put_nowait(None) if not self.queue.full() else None
            )
        except Exception:
            pass  # Ignore if loop stopped

    async def wait_complete(self, timeout: Optional[float] = None):
        """Wait for send task completion.

        Args:
            timeout: Maximum wait time in seconds. None = wait indefinitely.

        Raises:
            Exception: Any error that occurred during sending
        """
        if self._send_task:
            try:
                if timeout:
                    await asyncio.wait_for(self._send_task, timeout=timeout)
                else:
                    # Wait indefinitely for inference to complete
                    await self._send_task
            except asyncio.TimeoutError:
                logger.warning("Pipeline send task timed out, cancelling")
                self._send_task.cancel()
                self._cancelled = True
                # Await the cancelled task to clean up properly
                try:
                    await self._send_task
                except asyncio.CancelledError:
                    pass  # Expected
            except asyncio.CancelledError:
                pass  # Task was cancelled externally
        if self._error:
            raise self._error


async def synthesize_speech(
    model,
    tts_text: str,
    prompt_text: str,
    prompt_audio: torch.Tensor,
    speed: float = 1.0,
    stream: bool = True,
) -> AsyncIterator[torch.Tensor]:
    """Synthesize speech using the TTS model with true streaming.

    This function runs the model inference and yields audio chunks as they
    are generated, enabling low-latency playback (minimal TTFT).

    Args:
        model: CosyVoice2 model instance
        tts_text: Text to synthesize
        prompt_text: Transcription of the prompt audio
        prompt_audio: Prompt audio tensor [1, num_samples]
        speed: Speech speed multiplier (>1.0 = faster, <1.0 = slower)
        stream: Enable true streaming (chunk-by-chunk generation)

    Yields:
        torch.Tensor: Audio chunks as they are generated (on CPU)

    Example:
        >>> async for chunk in synthesize_speech(model, "Hello", "Hi", prompt):
        ...     print(chunk.shape)
    """
    import queue
    import threading

    # Use a queue to pass chunks from sync iterator to async generator
    chunk_queue: queue.Queue = queue.Queue()
    error_holder: list = []

    def _streaming_inference():
        """Run synchronous streaming inference in background thread."""
        try:
            for wav_dict in model.inference_zero_shot(
                tts_text=tts_text,
                prompt_text=prompt_text,
                prompt_speech_16k=prompt_audio,
                stream=stream,
                speed=speed,
            ):
                audio_chunk = wav_dict["tts_speech"]
                # Move to CPU immediately
                audio_chunk = audio_chunk.detach().cpu()
                chunk_queue.put(audio_chunk)
            # Signal completion
            chunk_queue.put(None)
        except Exception as e:
            error_holder.append(e)
            chunk_queue.put(None)

    # Start inference in background thread
    inference_thread = threading.Thread(target=_streaming_inference, daemon=True)
    inference_thread.start()

    # Yield chunks as they become available
    while True:
        # Use asyncio-friendly polling to avoid blocking event loop
        chunk = await run_in_threadpool(chunk_queue.get)

        if chunk is None:
            # Check for errors
            if error_holder:
                raise RuntimeError(f"Model inference failed: {error_holder[0]}")
            break

        yield chunk


async def synthesize_speech_cached(
    model,
    tts_text: str,
    speaker_id: str,
    speed: float = 1.0,
) -> AsyncIterator[torch.Tensor]:
    """Synthesize speech using cached prompt for reduced TTFT.

    This function uses a pre-cached prompt to skip prompt preprocessing,
    significantly reducing time to first audio chunk.

    Args:
        model: TTSModel instance with cached prompt
        tts_text: Text to synthesize
        speaker_id: ID of the cached prompt to use
        speed: Speech speed multiplier (0.5-2.0)

    Yields:
        torch.Tensor: Audio chunks as they are generated (on CPU)
    """
    import queue
    import threading

    chunk_queue: queue.Queue = queue.Queue()
    error_holder: list = []

    def _streaming_inference():
        """Run synchronous streaming inference in background thread."""
        try:
            for wav_dict in model.inference_zero_shot_cached(
                tts_text=tts_text,
                speaker_id=speaker_id,
                stream=True,
                speed=speed,
            ):
                audio_chunk = wav_dict["tts_speech"]
                audio_chunk = audio_chunk.detach().cpu()
                chunk_queue.put(audio_chunk)
            chunk_queue.put(None)
        except Exception as e:
            error_holder.append(e)
            chunk_queue.put(None)

    inference_thread = threading.Thread(target=_streaming_inference, daemon=True)
    inference_thread.start()

    while True:
        chunk = await run_in_threadpool(chunk_queue.get)
        if chunk is None:
            if error_holder:
                raise RuntimeError(f"Cached inference failed: {error_holder[0]}")
            break
        yield chunk


# HTTP Endpoints
@router.post("/", summary="Synthesize speech from text")
async def tts_synthesis(
    tts_text: str = Form(..., description="Text to synthesize"),
    prompt_text: str = Form(..., description="Prompt audio transcription"),
    prompt_wav: UploadFile = File(..., description="Prompt audio file"),
    speed: float = Query(default=1.0, ge=0.5, le=2.0, description="Speech speed"),
    stream: bool = Query(default=False, description="Enable streaming"),
    format: str = Query(default="wav", regex="^(wav|pcm)$", description="Output format"),
    model_state: dict = Depends(get_model_state),
):
    """Synthesize speech from text using zero-shot voice cloning.

    This endpoint accepts a text prompt, a reference audio sample, and synthesizes
    speech in the voice of the reference audio. Supports both streaming and
    non-streaming modes.

    Uses prompt embedding cache for faster inference when synthesizing multiple
    segments from the same prompt.

    Args:
        tts_text: Text to be synthesized into speech
        prompt_text: Transcription of the prompt audio
        prompt_wav: Audio file for voice cloning (WAV, MP3, etc.)
        speed: Speech speed multiplier (0.5-2.0)
        stream: If True, return streaming audio chunks
        format: Output format - "wav" (with headers) or "pcm" (raw audio)
        model_state: Injected model state dependency

    Returns:
        StreamingResponse or Response: Audio data in requested format

    Raises:
        HTTPException: 503 if model not loaded, 400 for invalid input

    Example:
        ```bash
        # Non-streaming WAV
        curl -X POST "http://localhost:8000/v1/tts?speed=1.0&format=wav" \\
             -F "tts_text=Hello world" \\
             -F "prompt_text=Hi there" \\
             -F "prompt_wav=@prompt.wav" \\
             -o output.wav

        # Streaming PCM
        curl -X POST "http://localhost:8000/v1/tts?stream=true&format=pcm" \\
             -F "tts_text=Hello world" \\
             -F "prompt_text=Hi there" \\
             -F "prompt_wav=@prompt.wav" \\
             --output - | play -t raw -r 22050 -e signed -b 16 -c 1 -
        ```
    """
    # Check model availability
    if not model_state.get("loaded"):
        raise HTTPException(
            status_code=503,
            detail="TTS model not loaded",
        )

    model = model_state["model_instance"]

    # Streaming only supports PCM format (WAV requires knowing total length)
    if stream and format == "wav":
        raise HTTPException(
            status_code=400,
            detail="Streaming mode only supports 'pcm' format. Use stream=false for WAV output.",
        )

    # Use semaphore for concurrency control
    async with _tts_semaphore:
        speaker_id = None
        use_cache = False
        try:
            # Load and preprocess prompt audio (16kHz for CosyVoice2 input)
            prompt_audio = await load_and_preprocess_audio(prompt_wav, PROMPT_SAMPLE_RATE)

            # Trim silence from prompt
            prompt_audio = await run_in_threadpool(
                trim_audio_vad, prompt_audio, PROMPT_SAMPLE_RATE
            )

            # Segment text for optimal processing
            segments = segment_text(tts_text)
            use_cache = len(segments) > 1

            if use_cache:
                # Cache prompt embedding for faster multi-segment synthesis
                speaker_id = f"http_{uuid.uuid4().hex[:8]}"
                await run_in_threadpool(
                    model.cache_prompt, prompt_text, prompt_audio, speaker_id
                )
                logger.info(f"Cached prompt for {len(segments)} segments (speaker_id={speaker_id})")

            # Streaming response (PCM only)
            if stream:
                async def audio_stream():
                    """Generate audio chunks for streaming with cache."""
                    try:
                        for seg_idx, segment in enumerate(segments):
                            if use_cache:
                                async for audio_chunk in synthesize_speech_cached(
                                    model, segment, speaker_id, speed
                                ):
                                    pcm_array = (audio_chunk.squeeze(0).numpy() * 32767).astype(
                                        np.int16
                                    )
                                    yield pcm_array.tobytes()
                            else:
                                async for audio_chunk in synthesize_speech(
                                    model, segment, prompt_text, prompt_audio, speed
                                ):
                                    pcm_array = (audio_chunk.squeeze(0).numpy() * 32767).astype(
                                        np.int16
                                    )
                                    yield pcm_array.tobytes()
                    finally:
                        # Cleanup cache after streaming completes
                        if use_cache and speaker_id:
                            try:
                                await run_in_threadpool(model.remove_cached_prompt, speaker_id)
                            except Exception as cleanup_err:
                                logger.warning(f"Failed to cleanup cache: {cleanup_err}")

                return StreamingResponse(
                    audio_stream(),
                    media_type="audio/pcm",
                    headers={
                        "X-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
                        "X-Bit-Depth": "16",
                        "X-Channels": "1",
                    },
                )

            # Non-streaming response
            else:
                audio_chunks = []
                for seg_idx, segment in enumerate(segments):
                    if use_cache:
                        async for audio_chunk in synthesize_speech_cached(
                            model, segment, speaker_id, speed
                        ):
                            audio_chunks.append(audio_chunk)
                    else:
                        async for audio_chunk in synthesize_speech(
                            model, segment, prompt_text, prompt_audio, speed, stream=False
                        ):
                            audio_chunks.append(audio_chunk)

                # Concatenate chunks (already on CPU)
                full_audio = torch.cat(audio_chunks, dim=-1)

                # Return based on format
                if format == "wav":
                    # Save as WAV file in memory
                    wav_buffer = io.BytesIO()
                    await run_in_threadpool(
                        torchaudio.save,
                        wav_buffer,
                        full_audio,
                        OUTPUT_SAMPLE_RATE,
                        format="wav",
                        encoding="PCM_S",
                        bits_per_sample=16,
                    )
                    wav_buffer.seek(0)

                    return Response(
                        content=wav_buffer.read(),
                        media_type="audio/wav",
                    )

                elif format == "pcm":
                    # Return raw PCM data (already on CPU)
                    pcm_array = (full_audio.squeeze(0).numpy() * 32767).astype(np.int16)
                    return Response(
                        content=pcm_array.tobytes(),
                        media_type="audio/pcm",
                        headers={
                            "X-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
                            "X-Bit-Depth": "16",
                            "X-Channels": "1",
                        },
                    )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"TTS synthesis failed: {str(e)}",
            )
        finally:
            # Cleanup cache for non-streaming requests
            if not stream and use_cache and speaker_id:
                try:
                    await run_in_threadpool(model.remove_cached_prompt, speaker_id)
                except Exception as cleanup_err:
                    logger.warning(f"Failed to cleanup cache: {cleanup_err}")


# WebSocket Endpoint
@router.websocket("/ws")
async def tts_websocket(
    websocket: WebSocket,
    model_state: dict = Depends(get_model_state),
):
    """WebSocket endpoint for real-time TTS synthesis.

    This endpoint accepts both binary audio data (for prompts) and JSON
    configuration messages. It returns synthesized audio as binary PCM chunks.

    Protocol:
        - Client sends binary: Prompt audio data (16kHz, mono, int16 PCM, max 10s)
        - Client sends JSON: {"prompt_text": "...", "speed": 1.0} or {"text": "..."}
        - Server sends binary: Synthesized audio PCM chunks (22050Hz, mono, int16)
        - Server sends JSON: {"status": "done"} when synthesis completes

    Audio Processing:
        - Input prompt audio MUST be 16kHz mono int16 PCM (no resampling performed)
        - Input prompt audio is automatically trimmed using VAD
        - Maximum prompt duration: 10 seconds (160000 samples at 16kHz)
        - Output audio sample rate: 22050Hz mono int16 PCM

    Args:
        websocket: WebSocket connection
        model_state: Injected model state dependency

    Raises:
        WebSocketDisconnect: When client disconnects

    Example:
        ```python
        import asyncio
        import json
        import websockets

        async def tts_client():
            uri = "ws://localhost:8000/v1/tts/ws"
            async with websockets.connect(uri) as ws:
                # Send prompt audio
                with open("prompt.pcm", "rb") as f:
                    await ws.send(f.read())

                # Send configuration
                config = {"prompt_text": "Hello", "speed": 1.0}
                await ws.send(json.dumps(config))

                # Send text to synthesize
                await ws.send(json.dumps({"text": "Hi there"}))

                # Receive audio chunks
                while True:
                    data = await ws.recv()
                    if isinstance(data, bytes):
                        # Audio chunk
                        process_audio(data)
                    else:
                        msg = json.loads(data)
                        if msg.get("status") == "done":
                            break

        asyncio.run(tts_client())
        ```
    """
    await websocket.accept()

    # Check model availability
    if not model_state.get("loaded"):
        await websocket.send_json({"error": "Model not loaded"})
        await websocket.close(code=1011, reason="Model not loaded")
        return

    model = model_state["model_instance"]

    # Session state
    session = {
        "prompt_audio": None,
        "prompt_text": None,
        "speed": 1.0,
        "speaker_id": str(uuid.uuid4()),
        "prompt_cached": False,
        "inference_in_progress": False,
    }

    # Maximum prompt audio duration (samples at 16kHz)
    MAX_PROMPT_SAMPLES = 16000 * 10  # 10 seconds max

    try:
        while True:
            # Receive message (binary or text)
            message = await websocket.receive()

            # Handle binary message (prompt audio)
            if "bytes" in message:
                audio_bytes = message["bytes"]

                try:
                    # Convert raw PCM to torch tensor (assumed 16kHz int16)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

                    # Size limit check to prevent memory issues
                    if len(audio_array) > MAX_PROMPT_SAMPLES:
                        await websocket.send_json({
                            "error": f"Prompt audio too long. Max {MAX_PROMPT_SAMPLES // PROMPT_SAMPLE_RATE}s allowed.",
                            "received_samples": len(audio_array),
                            "max_samples": MAX_PROMPT_SAMPLES,
                        })
                        continue

                    # Copy array to make it writable for PyTorch
                    audio_tensor = (
                        torch.from_numpy(audio_array.copy()).unsqueeze(0).float() / 32768.0
                    )

                    # Apply VAD trimming (same as HTTP endpoint)
                    audio_tensor = await run_in_threadpool(
                        trim_audio_vad, audio_tensor, PROMPT_SAMPLE_RATE
                    )

                    # Store prompt audio
                    session["prompt_audio"] = audio_tensor

                    await websocket.send_json({
                        "status": "prompt_received",
                        "samples": audio_tensor.shape[-1],
                        "duration_sec": audio_tensor.shape[-1] / PROMPT_SAMPLE_RATE,
                    })

                except Exception as e:
                    await websocket.send_json(
                        {"error": f"Failed to process audio: {str(e)}"}
                    )

            # Handle text message (JSON configuration or TTS request)
            elif "text" in message:
                try:
                    data = json.loads(message["text"])

                    # Update configuration
                    if "prompt_text" in data:
                        session["prompt_text"] = data["prompt_text"]

                        # If we have both prompt_audio and prompt_text, create cache
                        if session["prompt_audio"] is not None and not session["prompt_cached"]:
                            try:
                                await run_in_threadpool(
                                    model.cache_prompt,
                                    session["prompt_text"],
                                    session["prompt_audio"],
                                    session["speaker_id"]
                                )
                                session["prompt_cached"] = True
                                await websocket.send_json({
                                    "status": "prompt_cached",
                                    "speaker_id": session["speaker_id"],
                                })
                            except Exception as e:
                                await websocket.send_json({
                                    "error": f"Failed to cache prompt: {str(e)}"
                                })
                    if "speed" in data:
                        session["speed"] = max(0.5, min(2.0, data["speed"]))

                    # Synthesize speech if text is provided
                    if "text" in data:
                        tts_text = data["text"]

                        # Validate session has required data
                        if session["prompt_audio"] is None:
                            await websocket.send_json(
                                {"error": "No prompt audio provided"}
                            )
                            continue

                        if session["prompt_text"] is None:
                            await websocket.send_json(
                                {"error": "No prompt text provided"}
                            )
                            continue

                        # Segment text for optimal RTF (keeps seq_len small)
                        segments = segment_text(tts_text)
                        logger.info(f"Text segmented into {len(segments)} segments: {segments}")

                        # Acquire semaphore only during inference (not entire connection)
                        async with _tts_semaphore:
                            try:
                                # Mark inference as in progress
                                session["inference_in_progress"] = True

                                # Get event loop for cross-thread communication
                                loop = asyncio.get_running_loop()

                                # Process each segment sequentially
                                for seg_idx, segment in enumerate(segments):
                                    logger.debug(f"Processing segment {seg_idx + 1}/{len(segments)}: {segment[:30]}...")

                                    # Create pipeline for this segment
                                    pipeline = AsyncStreamingPipeline(websocket, loop, speed=session["speed"])
                                    await pipeline.start_sender()

                                    # Use cached inference if available
                                    if session["prompt_cached"]:
                                        # Capture segment in closure
                                        def _make_cached_worker(seg_text):
                                            def _inference_worker_cached():
                                                try:
                                                    for wav_dict in model.inference_zero_shot_cached(
                                                        tts_text=seg_text,
                                                        speaker_id=session["speaker_id"],
                                                        stream=True,
                                                        speed=session["speed"],
                                                    ):
                                                        chunk = wav_dict["tts_speech"].detach().cpu()
                                                        pipeline.put_chunk_sync(chunk)
                                                except Exception as e:
                                                    logger.error(f"Cached inference error: {e}")
                                                finally:
                                                    pipeline.mark_done()
                                            return _inference_worker_cached

                                        inference_thread = threading.Thread(
                                            target=_make_cached_worker(segment), daemon=True
                                        )
                                    else:
                                        # Use standard inference (no cache)
                                        def _make_standard_worker(seg_text):
                                            def _inference_worker_standard():
                                                try:
                                                    for wav_dict in model.inference_zero_shot(
                                                        tts_text=seg_text,
                                                        prompt_text=session["prompt_text"],
                                                        prompt_speech_16k=session["prompt_audio"],
                                                        stream=True,
                                                        speed=session["speed"],
                                                    ):
                                                        chunk = wav_dict["tts_speech"].detach().cpu()
                                                        pipeline.put_chunk_sync(chunk)
                                                except Exception as e:
                                                    logger.error(f"Standard inference error: {e}")
                                                finally:
                                                    pipeline.mark_done()
                                            return _inference_worker_standard

                                        inference_thread = threading.Thread(
                                            target=_make_standard_worker(segment), daemon=True
                                        )

                                    # Start inference in background
                                    inference_thread.start()

                                    # Wait for pipeline to complete (no timeout - wait for inference)
                                    try:
                                        await pipeline.wait_complete()
                                    except Exception as e:
                                        logger.warning(f"Pipeline error for segment {seg_idx + 1}: {e}")

                                    # Wait for inference thread to finish
                                    inference_thread.join(timeout=5.0)
                                    if inference_thread.is_alive():
                                        logger.warning(f"Inference thread for segment {seg_idx + 1} still running after join timeout")

                            finally:
                                # Mark inference as complete
                                session["inference_in_progress"] = False

                        # Send completion message
                        await websocket.send_json({"status": "done"})

                    else:
                        # Configuration update only
                        await websocket.send_json({"status": "config_updated"})

                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON"})
                except Exception as e:
                    await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        # Client disconnected - cleanup will be handled in finally
        pass
    except Exception as e:
        # Try to send error message, but ignore if connection is already closed
        try:
            await websocket.send_json({"error": f"WebSocket error: {str(e)}"})
            await websocket.close(code=1011, reason=str(e))
        except RuntimeError:
            # Connection already closed, ignore
            pass
    finally:
        # Common cleanup logic for all exit paths
        if session.get("prompt_cached"):
            # Wait for inference to complete before cleanup (max 5 seconds)
            cleanup_timeout = 5.0
            cleanup_start = asyncio.get_event_loop().time()
            while session.get("inference_in_progress", False):
                if asyncio.get_event_loop().time() - cleanup_start > cleanup_timeout:
                    logger.warning(
                        "Cleanup timeout for speaker_id %s, inference still in progress",
                        session["speaker_id"]
                    )
                    break
                await asyncio.sleep(0.1)

            # Attempt to remove cached prompt
            # Note: remove_cached_prompt uses reference counting and will safely
            # skip deletion if the cache is still in use by another inference
            try:
                removed = await run_in_threadpool(
                    model.remove_cached_prompt, session["speaker_id"]
                )
                if not removed:
                    logger.debug(
                        "Cache for speaker_id %s not removed (may still be in use)",
                        session["speaker_id"]
                    )
            except Exception as cleanup_error:
                logger.warning("Failed to cleanup cache: %s", cleanup_error)
