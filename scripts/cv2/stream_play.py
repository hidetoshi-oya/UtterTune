#!/usr/bin/env python3
"""
CosyVoice 2 + LoRA real-time streaming playback script
======================================================
Load a pretrained LoRA adapter and synthesize speech with real-time playback.
Audio is played chunk by chunk as it is generated for low latency.

Usage:
    python -m scripts.cv2.stream_play \
        --base_model pretrained_models/CosyVoice2-0.5B \
        --lora_dir lora_weights/UtterTune-CosyVoice2-jp-JSUTJVS \
        --texts "魑魅魍魎が跋扈する。" \
        --prompt_wav prompts/wav/common_voice_ja_41758953.wav \
        --prompt_text prompts/trans/common_voice_ja_41758953.txt

Requirements:
    pip install sounddevice
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
import warnings
from logging import INFO, StreamHandler, getLogger
from pathlib import Path
from typing import List

import huggingface_hub
import numpy as np
import safetensors.torch as st
import sounddevice as sd
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from peft import PeftConfig, PeftModel

from scripts.cv2.patch import apply_patch

apply_patch()

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

huggingface_hub.cached_download = hf_hub_download

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False


def load_wav(path: Path, sr_out: int) -> np.ndarray:
    """Load a wav file and resample to the target sample rate."""
    wav, sr = torchaudio.load(path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sr_out:
        wav = torchaudio.functional.resample(wav, sr, sr_out)

    return wav


def trim_wav(wav: torch.Tensor, sr: int, trigger_level: float = 7.0) -> torch.Tensor:
    """Trim silence from the beginning and end of the audio."""
    trimmed = torchaudio.functional.vad(wav, sr, trigger_level=trigger_level)

    if trimmed.shape[-1] > 0:
        trimmed_rev = torchaudio.functional.vad(
            trimmed.flip(-1), sr, trigger_level=trigger_level
        )
        trimmed = trimmed_rev.flip(-1)

    return trimmed


class StreamingPlayer:
    """Real-time audio player that plays chunks as they arrive."""

    def __init__(self, sample_rate: int, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self.stream: sd.OutputStream | None = None
        self.is_playing = False
        self._playback_thread: threading.Thread | None = None

    def _playback_worker(self) -> None:
        """Worker thread that plays audio chunks from the queue."""
        while self.is_playing:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is None:
                    break
                if self.stream is not None:
                    self.stream.write(chunk)
            except queue.Empty:
                continue

    def start(self) -> None:
        """Start the audio stream and playback thread."""
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
        )
        self.stream.start()
        self.is_playing = True
        self._playback_thread = threading.Thread(target=self._playback_worker)
        self._playback_thread.start()

    def play_chunk(self, audio: np.ndarray) -> None:
        """Add an audio chunk to the playback queue."""
        self.audio_queue.put(audio)

    def stop(self) -> None:
        """Stop playback and close the stream."""
        self.audio_queue.put(None)
        self.is_playing = False

        if self._playback_thread is not None:
            self._playback_thread.join(timeout=2.0)

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def wait_until_done(self) -> None:
        """Wait until all queued audio has been played."""
        self.audio_queue.join()


def load_model(args) -> tuple:
    """Load CosyVoice2 model with optional LoRA adapter."""
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cv2 = CosyVoice2(model_dir=args.base_model, fp16=False)

    if args.lora_dir is not None:
        base_model = cv2.model.llm

        tok = get_qwen_tokenizer(
            token_path=f"{args.base_model}/CosyVoice-BlankEN", skip_special_tokens=True
        )

        new_tokens = ["<PHON_START>", "<PHON_END>"]
        added = tok.tokenizer.add_special_tokens(
            {"additional_special_tokens": new_tokens}
        )
        logger.info("Number of tokens added: %s", added)

        tok.special_tokens["additional_special_tokens"].extend(
            [
                t
                for t in new_tokens
                if t not in tok.special_tokens["additional_special_tokens"]
            ]
        )
        base_model.llm.model.resize_token_embeddings(len(tok.tokenizer))
        new_ids = tok.tokenizer.convert_tokens_to_ids(new_tokens)

        logger.info("Loading LoRA from %s", args.lora_dir)

        peft_config = PeftConfig.from_pretrained(args.lora_dir)
        peft_config.task_type = None
        hf_model = PeftModel.from_pretrained(
            base_model,
            args.lora_dir,
            config=peft_config,
            is_trainable=False,
            torch_dtype=torch.float32,
        )
        hf_model.to(device).eval()

        rows = st.load_file(args.lora_dir / "embed_patch.safetensors")["embed_rows"].to(
            device
        )

        with torch.no_grad():
            hf_model.base_model.llm.model.get_input_embeddings().weight[new_ids] = rows

        cv2.model.llm = hf_model
        logger.info("LoRA loaded successfully")

    return cv2, device


def main():
    ap = argparse.ArgumentParser(
        description="Real-time streaming TTS with CosyVoice 2 + LoRA"
    )
    ap.add_argument(
        "--base_model",
        type=str,
        required=True,
        default="pretrained_models/CosyVoice2-0.5B",
        help="CosyVoice2 base model directory",
    )
    ap.add_argument(
        "--lora_dir",
        type=Path,
        default=None,
        help="LoRA adapter directory",
    )
    ap.add_argument(
        "--texts",
        required=True,
        help="Synthesise these sentences; separated by | or text file path",
    )
    ap.add_argument(
        "--prompt_wav",
        type=Path,
        required=True,
        help="Prompt wav file path; less than 4-second duration is recommended",
    )
    ap.add_argument(
        "--prompt_text",
        required=True,
        help="Transcription for the prompt wav",
    )
    ap.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed (default: 1.0)",
    )
    ap.add_argument(
        "--save",
        action="store_true",
        help="Also save the synthesized audio to files",
    )
    ap.add_argument("--out_dir", type=Path, default="wavs_out")
    ap.add_argument(
        "--cpu", action="store_true", help="Force CPU inference (for debug)"
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logger.info("Loading model...")
    cv2, device = load_model(args)

    PROMPT_SAMPLE_RATE = 16000

    if Path(args.texts).is_file():
        sentences: List[str] = [
            ln.strip()
            for ln in Path(args.texts).read_text("utf-8").splitlines()
            if ln.strip()
        ]
    else:
        sentences = [s.strip() for s in args.texts.split("|") if s.strip()]

    prompt_speech_16k = load_wav(args.prompt_wav, PROMPT_SAMPLE_RATE)
    prompt_speech_16k = trim_wav(prompt_speech_16k, PROMPT_SAMPLE_RATE)

    if Path(args.prompt_text).is_file():
        prompt_text = Path(args.prompt_text).read_text("utf_8").strip()
    else:
        prompt_text = args.prompt_text

    if args.save:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    player = StreamingPlayer(sample_rate=cv2.sample_rate)

    for idx, sentence in enumerate(sentences):
        logger.info(f"[{idx + 1:03d}] Synthesizing: '{sentence}'")

        t0 = time.perf_counter()
        chunk_count = 0
        total_samples = 0
        all_chunks: List[torch.Tensor] = []

        player.start()

        try:
            for wav_dict in cv2.inference_zero_shot(
                tts_text=sentence,
                prompt_text=prompt_text,
                prompt_speech_16k=prompt_speech_16k,
                stream=True,
                speed=args.speed,
            ):
                chunk_count += 1
                wav_chunk = wav_dict["tts_speech"]
                total_samples += wav_chunk.shape[-1]

                audio_np = wav_chunk.squeeze().cpu().numpy().astype(np.float32)
                player.play_chunk(audio_np)

                if args.save:
                    all_chunks.append(wav_chunk)

                chunk_duration = wav_chunk.shape[-1] / cv2.sample_rate
                logger.info(
                    f"  Chunk {chunk_count}: {chunk_duration:.2f}s"
                )

        finally:
            player.stop()

        dt = time.perf_counter() - t0
        total_duration = total_samples / cv2.sample_rate
        rtf = dt / total_duration if total_duration > 0 else 0

        logger.info(
            f"  Done: {total_duration:.2f}s audio in {dt:.2f}s (RTF: {rtf:.2f})"
        )

        if args.save and all_chunks:
            full_wav = torch.cat(all_chunks, dim=-1)
            out_path = out_dir / f"{idx + 1:03d}_stream.wav"
            torchaudio.save(
                str(out_path),
                full_wav,
                cv2.sample_rate,
                format="wav",
                encoding="PCM_S",
            )
            logger.info(f"  Saved: {out_path}")

    logger.info("All sentences have been synthesised and played.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    main()
