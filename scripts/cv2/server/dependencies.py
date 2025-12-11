#!/usr/bin/env python3
"""
FastAPI Dependencies for CosyVoice2 TTS Model
==============================================

This module provides dependency injection for the CosyVoice2 TTS model with optional LoRA adapter support.
It manages model initialization, loading, and inference through a global singleton pattern.

Key Features:
    - CosyVoice2 base model loading with FP16 support
    - Optional LoRA adapter loading with special token expansion
    - Singleton pattern for efficient memory usage
    - Thread-safe model access for FastAPI
    - Comprehensive error handling and logging

Usage Example:
    ```python
    from fastapi import FastAPI, Depends
    from scripts.cv2.server.dependencies import init_model, get_model, TTSModel

    # Initialize during startup
    config = {
        "base_model": "pretrained_models/CosyVoice2-0.5B",
        "lora_dir": "lora_weights/cv2/ja/checkpoint-20000",
        "use_cpu": False,
        "fp16": False,
    }
    init_model(config)

    # Use in endpoints
    @app.post("/synthesize")
    async def synthesize(model: TTSModel = Depends(get_model)):
        result = model.inference_zero_shot(...)
        return result
    ```

Author: UtterTune Team
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np
import safetensors.torch as st
import torch
import torchaudio
from peft import PeftConfig, PeftModel

# CRITICAL: Apply patch BEFORE importing CosyVoice
from scripts.cv2.patch import apply_patch

apply_patch()

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for CosyVoice2 model initialization.

    This configuration specifies all parameters needed to load the base model
    and optional LoRA adapter with special tokens.

    Attributes:
        base_model: Path to CosyVoice2 base model directory.
        lora_dir: Optional path to LoRA adapter directory. If None, only base model is used.
        use_cpu: Force CPU inference even if CUDA is available. Useful for debugging.
        fp16: Use FP16 precision for base model. Note: LoRA always uses FP32.
        load_jit: Load JIT-compiled flow encoder for faster inference.
        load_trt: Load TensorRT-optimized flow decoder (requires .plan file).
        load_vllm: Use vLLM for LLM inference acceleration.
        trt_concurrent: Number of concurrent TensorRT streams (default: 1).
        sample_rate: Audio sample rate for synthesis output (default: 22050 for CosyVoice2).

    Example:
        >>> config = ModelConfig(
        ...     base_model="pretrained_models/CosyVoice2-0.5B",
        ...     lora_dir="lora_weights/cv2/ja/checkpoint-20000",
        ...     fp16=True,
        ...     load_jit=True,
        ...     load_vllm=True
        ... )
    """

    base_model: str
    lora_dir: Optional[str] = None
    use_cpu: bool = False
    fp16: bool = False
    load_jit: bool = False  # JIT-compiled flow encoder
    load_trt: bool = False  # TensorRT flow decoder
    load_vllm: bool = False  # vLLM for LLM inference
    vllm_model_dir: Optional[str] = None  # Custom vLLM model directory (for merged LoRA)
    trt_concurrent: int = 1  # TensorRT concurrent streams
    sample_rate: int = 22050  # CosyVoice2 output sample rate


class TTSModel:
    """
    Text-to-Speech model wrapper for CosyVoice2 with optional LoRA adapter.

    This class encapsulates the CosyVoice2 model and handles:
    - Base model loading with device placement
    - Optional LoRA adapter integration with special tokens
    - Inference methods for zero-shot voice cloning
    - Audio preprocessing and postprocessing

    The LoRA loading process follows these critical steps:
    1. Load CosyVoice2 base model
    2. Expand tokenizer with special tokens (<PHON_START>, <PHON_END>)
    3. Resize model embeddings to accommodate new tokens
    4. Load LoRA weights with task_type=None (CRITICAL for CosyVoice2)
    5. Patch embeddings for new tokens from saved weights

    Attributes:
        config: Model configuration containing paths and settings.
        device: PyTorch device (cuda or cpu).
        model: CosyVoice2 model instance with optional LoRA adapter.
        sample_rate: Audio sample rate for synthesis.

    Example:
        >>> config = ModelConfig(base_model="path/to/model")
        >>> tts = TTSModel(config)
        >>> audio = tts.inference_zero_shot(
        ...     tts_text="Hello world",
        ...     prompt_text="Sample text",
        ...     prompt_speech_16k=audio_tensor
        ... )

    Note:
        This class is designed to be used as a singleton through the dependency
        injection functions init_model() and get_model().
    """

    def __init__(self, config: Union[ModelConfig, Dict[str, Any]]):
        """
        Initialize the TTS model with the given configuration.

        This constructor loads the CosyVoice2 base model and optionally applies
        a LoRA adapter with special token expansion.

        Args:
            config: Model configuration as ModelConfig instance or dictionary.
                   Dictionary will be converted to ModelConfig automatically.

        Raises:
            RuntimeError: If model loading fails.
            FileNotFoundError: If base_model or lora_dir paths don't exist.
            ValueError: If configuration is invalid.

        Example:
            >>> config = {"base_model": "path/to/model", "lora_dir": "path/to/lora"}
            >>> model = TTSModel(config)
        """
        # Convert dict to ModelConfig if needed
        if isinstance(config, dict):
            config = ModelConfig(**config)

        self.config = config
        self.device = self._setup_device()
        self.sample_rate = config.sample_rate
        self._cache_lock = threading.Lock()  # キャッシュ操作用ロック
        self._cache_ref_count: Dict[str, int] = {}  # 参照カウンタ（使用中のキャッシュを追跡）

        logger.info("Initializing CosyVoice2 model from %s", config.base_model)

        # Validate base model path
        base_model_path = Path(config.base_model)
        if not base_model_path.exists():
            raise FileNotFoundError(f"Base model directory not found: {config.base_model}")

        try:
            # Load base CosyVoice2 model
            self.model = self._load_base_model()

            # Optionally load LoRA adapter
            # Skip if vllm_model_dir is specified (LoRA is pre-merged into vLLM model)
            if config.lora_dir is not None and config.vllm_model_dir is None:
                self._load_lora_adapter()
            elif config.vllm_model_dir is not None:
                logger.info("Using pre-merged vLLM model, skipping LoRA adapter loading")
                # Load tokenizer from vllm_model_dir (includes special tokens)
                self._load_vllm_tokenizer()

            # Apply torch.compile only when NOT using vLLM (vLLM has its own optimizations)
            # Must be done AFTER all weights are loaded (LoRA included)
            if not config.load_vllm and not config.use_cpu and torch.cuda.is_available():
                logger.info("Applying torch.compile to LLM (mode=reduce-overhead)...")
                self.model.model.llm = torch.compile(
                    self.model.model.llm,
                    mode="reduce-overhead",
                )
                logger.info("torch.compile applied successfully")

            logger.info("Model initialization complete on device: %s", self.device)

        except Exception as e:
            logger.error("Failed to initialize TTS model: %s", e)
            raise RuntimeError(f"Model initialization failed: {e}") from e

    def _setup_device(self) -> torch.device:
        """
        Determine the appropriate PyTorch device for model inference.

        Returns:
            torch.device: Either 'cuda' or 'cpu' based on configuration and availability.

        Note:
            Respects the use_cpu flag even if CUDA is available.
        """
        if self.config.use_cpu or not torch.cuda.is_available():
            logger.info("Using CPU for inference")
            return torch.device("cpu")

        logger.info("Using CUDA for inference")
        return torch.device("cuda")

    def _load_base_model(self) -> CosyVoice2:
        """
        Load the CosyVoice2 base model with optional optimizations.

        Supports the following optimizations:
        - JIT: Compiled flow encoder for faster inference
        - TensorRT: GPU-optimized flow decoder
        - vLLM: High-performance LLM inference (supports custom model directory for merged LoRA)

        Returns:
            CosyVoice2: Loaded base model instance.

        Raises:
            RuntimeError: If base model loading fails.
        """
        try:
            # Log optimization settings
            opts = []
            if self.config.load_jit:
                opts.append("JIT")
            if self.config.load_trt:
                opts.append("TensorRT")
            if self.config.load_vllm:
                if self.config.vllm_model_dir:
                    opts.append(f"vLLM (custom: {self.config.vllm_model_dir})")
                else:
                    opts.append("vLLM")
            if self.config.fp16:
                opts.append("FP16")

            if opts:
                logger.info("Loading model with optimizations: %s", ", ".join(opts))
            else:
                logger.info("Loading model without optimizations")

            # If custom vllm_model_dir is specified, we load vLLM manually after init
            # to use the custom path instead of the default {model_dir}/vllm
            use_default_vllm = self.config.load_vllm and self.config.vllm_model_dir is None

            model = CosyVoice2(
                model_dir=self.config.base_model,
                load_jit=self.config.load_jit,
                load_trt=self.config.load_trt,
                load_vllm=use_default_vllm,  # Only use default vLLM if no custom path
                fp16=self.config.fp16,
                trt_concurrent=self.config.trt_concurrent,
            )

            # Load vLLM from custom directory (for merged LoRA models)
            if self.config.load_vllm and self.config.vllm_model_dir is not None:
                logger.info("Loading vLLM from custom directory: %s", self.config.vllm_model_dir)
                # Note: No custom model registration needed - merged models use Qwen2ForCausalLM
                # which is built into vLLM. This avoids the multiprocessing spawn issue in vLLM 0.12.x.
                model.model.load_vllm(self.config.vllm_model_dir)
                logger.info("vLLM loaded from custom directory successfully")

            # Move model to correct device if CPU mode is requested
            if self.config.use_cpu and hasattr(model, 'model'):
                logger.info("Moving model components to CPU")
                inner_model = model.model
                # Move individual components (CosyVoice2Model is not a nn.Module)
                if hasattr(inner_model, 'llm'):
                    inner_model.llm.to(self.device)
                if hasattr(inner_model, 'flow'):
                    inner_model.flow.to(self.device)
                if hasattr(inner_model, 'hift'):
                    inner_model.hift.to(self.device)
                # Update internal device reference
                inner_model.device = self.device

            logger.info("Base model loaded successfully")
            return model

        except Exception as e:
            logger.error("Failed to load base model: %s", e)
            raise RuntimeError(f"Base model loading failed: {e}") from e

    def _load_vllm_tokenizer(self) -> None:
        """
        Load tokenizer from vllm_model_dir for pre-merged vLLM model.

        This method loads the tokenizer that was saved together with the merged model,
        which already includes the special tokens (<PHON_START>, <PHON_END>) with
        correct token IDs matching the model embeddings.

        Note:
            The tokenizer must have been saved by merge_lora_for_vllm.py.
            This ensures token IDs match between tokenizer and model embeddings.
        """
        logger.info("Loading tokenizer from vLLM model directory: %s", self.config.vllm_model_dir)

        try:
            from transformers import AutoTokenizer

            # Load tokenizer from vllm_model_dir (includes special tokens)
            hf_tokenizer = AutoTokenizer.from_pretrained(
                self.config.vllm_model_dir,
                trust_remote_code=True
            )

            # Wrap in QwenTokenizer-compatible interface
            # Get base tokenizer first, then replace internal tokenizer
            tokenizer_path = f"{self.config.base_model}/CosyVoice-BlankEN"
            tok = get_qwen_tokenizer(
                token_path=tokenizer_path,
                skip_special_tokens=True
            )

            # Replace internal tokenizer with the one from vllm_model_dir
            tok.tokenizer = hf_tokenizer

            # Update special tokens metadata
            new_tokens = ["<PHON_START>", "<PHON_END>"]
            tok.special_tokens["additional_special_tokens"].extend(
                [t for t in new_tokens if t not in tok.special_tokens["additional_special_tokens"]]
            )

            # Update frontend tokenizer
            self.model.frontend.tokenizer = tok

            new_ids = hf_tokenizer.convert_tokens_to_ids(new_tokens)
            logger.info("Loaded tokenizer with special token IDs: %s", new_ids)
            logger.info("Tokenizer vocab size: %d", len(hf_tokenizer))
            logger.info("vLLM tokenizer loaded successfully")

        except Exception as e:
            logger.error("Failed to load vLLM tokenizer: %s", e)
            raise RuntimeError(f"vLLM tokenizer loading failed: {e}") from e

    def _load_lora_adapter(self) -> None:
        """
        Load LoRA adapter and apply special token expansion.

        This method performs the complete LoRA integration process:
        1. Validates LoRA directory exists
        2. Expands tokenizer with special tokens (<PHON_START>, <PHON_END>)
        3. Resizes model embeddings to accommodate new tokens
        4. Loads LoRA weights with modified config (task_type=None)
        5. Patches embeddings for new tokens from safetensors file

        Raises:
            FileNotFoundError: If lora_dir or embed_patch.safetensors not found.
            RuntimeError: If LoRA loading or token expansion fails.

        Note:
            CRITICAL: peft_config.task_type must be set to None for CosyVoice2.
            This prevents PeftModelForCausalLM from requiring prepare_inputs_for_generation.
        """
        lora_dir = Path(self.config.lora_dir)

        # Validate LoRA directory
        if not lora_dir.exists():
            raise FileNotFoundError(f"LoRA directory not found: {lora_dir}")

        logger.info("Loading LoRA adapter from %s", lora_dir)

        try:
            # Get base LLM for LoRA attachment
            base_llm = self.model.model.llm

            # Step 1: Expand vocabulary with special tokens
            logger.info("Expanding tokenizer with special tokens")
            tokenizer_path = f"{self.config.base_model}/CosyVoice-BlankEN"
            tok = get_qwen_tokenizer(
                token_path=tokenizer_path,
                skip_special_tokens=True
            )

            # Step 2: Register new special tokens
            new_tokens = ["<PHON_START>", "<PHON_END>"]
            num_added = tok.tokenizer.add_special_tokens(
                {"additional_special_tokens": new_tokens}
            )
            logger.info("Added %d special tokens to tokenizer", num_added)

            # Step 3: Update tokenizer metadata
            tok.special_tokens["additional_special_tokens"].extend(
                [t for t in new_tokens if t not in tok.special_tokens["additional_special_tokens"]]
            )

            # Step 4: Resize model embeddings to accommodate new tokens
            base_llm.llm.model.resize_token_embeddings(len(tok.tokenizer))
            new_ids = tok.tokenizer.convert_tokens_to_ids(new_tokens)
            logger.info("New token IDs: %s", new_ids)

            # Step 5: Load LoRA weights with modified config
            # CRITICAL: Override task_type to None for CosyVoice2 compatibility
            logger.info("Loading LoRA weights")
            peft_config = PeftConfig.from_pretrained(lora_dir)
            peft_config.task_type = None  # CRITICAL: Must be None for CosyVoice2

            hf_model = PeftModel.from_pretrained(
                base_llm,
                lora_dir,
                config=peft_config,
                is_trainable=False,
                torch_dtype=torch.float32  # LoRA uses FP32
            )
            hf_model.to(self.device).eval()

            # Step 6: Patch embeddings for new tokens
            logger.info("Patching embeddings for special tokens")
            embed_path = lora_dir / "embed_patch.safetensors"

            if not embed_path.exists():
                raise FileNotFoundError(
                    f"Embedding patch file not found: {embed_path}. "
                    "This file is required for LoRA adapter."
                )

            rows = st.load_file(str(embed_path))["embed_rows"].to(self.device)

            with torch.no_grad():
                # Convert rows to match embedding dtype (FP16 compatibility)
                embed_weight = hf_model.base_model.llm.model.get_input_embeddings().weight
                rows = rows.to(dtype=embed_weight.dtype)
                embed_weight[new_ids] = rows

            # Step 7: Replace base LLM with LoRA-enhanced version
            self.model.model.llm = hf_model

            logger.info("LoRA adapter loaded successfully")
            logger.debug("Special token embeddings: %s",
                        self.model.model.llm.llm.model.model.embed_tokens.weight[new_ids])

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to load LoRA adapter: %s", e)
            raise RuntimeError(f"LoRA adapter loading failed: {e}") from e

    def _register_vllm_model(self) -> None:
        """
        Register CosyVoice2ForCausalLM with vLLM's ModelRegistry.

        This registration is required before loading vLLM with a CosyVoice2 model
        that has been exported with CosyVoice2ForCausalLM architecture.

        Note:
            This only needs to be called once per process. Subsequent calls are no-ops.
        """
        try:
            from vllm import ModelRegistry
            from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

            # Check if already registered
            if not hasattr(self, '_vllm_model_registered'):
                logger.info("Registering CosyVoice2ForCausalLM with vLLM ModelRegistry")
                ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
                self._vllm_model_registered = True
                logger.info("CosyVoice2ForCausalLM registered successfully")

        except ImportError as e:
            logger.error("Failed to import vLLM or CosyVoice2ForCausalLM: %s", e)
            raise RuntimeError(f"vLLM model registration failed: {e}") from e
        except Exception as e:
            logger.error("Failed to register vLLM model: %s", e)
            raise RuntimeError(f"vLLM model registration failed: {e}") from e

    def inference_zero_shot(
        self,
        tts_text: str,
        prompt_text: str,
        prompt_speech_16k: Union[torch.Tensor, np.ndarray],
        stream: bool = False,
        speed: float = 1.0,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Perform zero-shot voice cloning inference.

        Synthesizes speech in the target voice using a prompt audio sample.
        This method supports phonetic annotations when LoRA is loaded (e.g., <PHON_START>...<PHON_END>).

        Args:
            tts_text: Text to synthesize. May contain phonetic annotations if LoRA is active.
                     Example: "Hello <PHON_START>wɜːrld<PHON_END>"
            prompt_text: Transcription of the prompt audio. Should match prompt_speech_16k content.
            prompt_speech_16k: Prompt audio tensor or numpy array at 16kHz sample rate.
                              Shape: (1, T) or (T,) where T is number of samples.
            stream: Enable true streaming mode for low-latency playback.
                   When True, yields audio chunks as they are generated.
                   When False, yields the complete audio in one chunk.
            speed: Speech speed multiplier (0.5-2.0). Default: 1.0.

        Returns:
            Iterator yielding dictionaries with 'tts_speech' key containing generated audio tensors.
            Each tensor has shape (1, T) where T is the number of samples.

        Raises:
            RuntimeError: If inference fails.
            ValueError: If input formats are invalid.

        Example:
            >>> prompt_audio = load_audio("prompt.wav")  # Shape: (1, T)
            >>> results = model.inference_zero_shot(
            ...     tts_text="Hello world",
            ...     prompt_text="Sample prompt",
            ...     prompt_speech_16k=prompt_audio,
            ...     stream=True,
            ...     speed=1.0
            ... )
            >>> for result in results:
            ...     audio = result["tts_speech"]  # Shape: (1, T)
            ...     play_audio_chunk(audio)

        Note:
            When stream=True, multiple audio chunks are yielded for low-latency playback.
            When stream=False, only one chunk containing the complete audio is yielded.
        """
        try:
            # Validate and convert prompt audio to tensor
            if isinstance(prompt_speech_16k, np.ndarray):
                prompt_speech_16k = torch.from_numpy(prompt_speech_16k)

            if not isinstance(prompt_speech_16k, torch.Tensor):
                raise ValueError(
                    f"prompt_speech_16k must be torch.Tensor or np.ndarray, "
                    f"got {type(prompt_speech_16k)}"
                )

            # Ensure correct shape (1, T)
            if prompt_speech_16k.ndim == 1:
                prompt_speech_16k = prompt_speech_16k.unsqueeze(0)

            logger.debug(
                "Starting inference: text_len=%d, prompt_len=%d samples, stream=%s, speed=%.2f",
                len(tts_text),
                prompt_speech_16k.shape[-1],
                stream,
                speed
            )

            # Perform inference with stream and speed parameters
            result_iter = self.model.inference_zero_shot(
                tts_text=tts_text,
                prompt_text=prompt_text,
                prompt_speech_16k=prompt_speech_16k,
                stream=stream,
                speed=speed,
            )

            return result_iter

        except Exception as e:
            logger.error("Inference failed: %s", e)
            raise RuntimeError(f"Zero-shot inference failed: {e}") from e

    def cache_prompt(
        self,
        prompt_text: str,
        prompt_speech_16k: Union[torch.Tensor, np.ndarray],
        speaker_id: str,
    ) -> None:
        """
        Cache prompt audio for faster subsequent inference.

        Pre-processes the prompt audio and stores it for reuse with the given speaker_id.
        This significantly reduces TTFT for subsequent TTS requests using the same prompt.

        Args:
            prompt_text: Transcription of the prompt audio.
            prompt_speech_16k: Prompt audio tensor or numpy array at 16kHz sample rate.
            speaker_id: Unique identifier for this cached prompt.

        Note:
            Thread-safe. Uses internal lock for concurrent access protection.
        """
        # Validate and convert prompt audio to tensor
        if isinstance(prompt_speech_16k, np.ndarray):
            prompt_speech_16k = torch.from_numpy(prompt_speech_16k)

        if prompt_speech_16k.ndim == 1:
            prompt_speech_16k = prompt_speech_16k.unsqueeze(0)

        # Important: normalize text before caching (same as inference_zero_shot does internally)
        normalized_text = self.model.frontend.text_normalize(
            prompt_text, split=False, text_frontend=True
        )

        logger.info("Caching prompt for speaker_id: %s", speaker_id)

        with self._cache_lock:
            self.model.add_zero_shot_spk(normalized_text, prompt_speech_16k, speaker_id)
            self._cache_ref_count[speaker_id] = 0  # Initialize reference count

        logger.info("Prompt cached successfully for speaker_id: %s", speaker_id)

    def remove_cached_prompt(self, speaker_id: str, force: bool = False) -> bool:
        """
        Remove a cached prompt and free associated memory.

        Args:
            speaker_id: The speaker_id of the cached prompt to remove.
            force: If True, remove even if in use (not recommended).

        Returns:
            bool: True if removed, False if still in use or not found.

        Note:
            Thread-safe. Will not remove if cache is currently in use (ref_count > 0)
            unless force=True. Also triggers CUDA cache cleanup if GPU is used.
        """
        with self._cache_lock:
            if speaker_id not in self.model.frontend.spk2info:
                logger.warning("No cached prompt found for speaker_id: %s", speaker_id)
                return False

            ref_count = self._cache_ref_count.get(speaker_id, 0)
            if ref_count > 0 and not force:
                logger.warning(
                    "Cannot remove cached prompt for speaker_id %s: still in use (ref_count=%d)",
                    speaker_id, ref_count
                )
                return False

            del self.model.frontend.spk2info[speaker_id]
            if speaker_id in self._cache_ref_count:
                del self._cache_ref_count[speaker_id]
            logger.info("Removed cached prompt for speaker_id: %s", speaker_id)

        # Free VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared after removing prompt")

        return True

    def _acquire_cache_ref(self, speaker_id: str) -> bool:
        """Increment reference count for a cached prompt. Returns True if successful."""
        with self._cache_lock:
            if speaker_id not in self.model.frontend.spk2info:
                return False
            self._cache_ref_count[speaker_id] = self._cache_ref_count.get(speaker_id, 0) + 1
            return True

    def _release_cache_ref(self, speaker_id: str) -> None:
        """Decrement reference count for a cached prompt."""
        with self._cache_lock:
            if speaker_id in self._cache_ref_count:
                self._cache_ref_count[speaker_id] = max(0, self._cache_ref_count[speaker_id] - 1)

    def inference_zero_shot_cached(
        self,
        tts_text: str,
        speaker_id: str,
        stream: bool = False,
        speed: float = 1.0,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Perform zero-shot inference using a cached prompt.

        This method skips prompt preprocessing by using a previously cached prompt,
        significantly reducing TTFT (Time To First Token).

        Args:
            tts_text: Text to synthesize.
            speaker_id: The speaker_id of the cached prompt to use.
            stream: Enable streaming mode for low-latency playback.
            speed: Speech speed multiplier (0.5-2.0).

        Returns:
            Iterator yielding dictionaries with 'tts_speech' key.

        Raises:
            RuntimeError: If inference fails or speaker_id not found.

        Note:
            Uses reference counting to prevent cache deletion during inference.
        """
        # Acquire reference to prevent deletion during inference
        if not self._acquire_cache_ref(speaker_id):
            raise RuntimeError(f"No cached prompt found for speaker_id: {speaker_id}")

        logger.debug(
            "Starting cached inference: text_len=%d, speaker_id=%s, stream=%s, speed=%.2f",
            len(tts_text),
            speaker_id,
            stream,
            speed
        )

        try:
            # Use cached prompt via zero_shot_spk_id
            result_iter = self.model.inference_zero_shot(
                tts_text=tts_text,
                prompt_text="",  # Not used when speaker_id is provided
                prompt_speech_16k=torch.zeros(1, 1),  # Dummy, not used
                zero_shot_spk_id=speaker_id,
                stream=stream,
                speed=speed,
            )

            # Wrap iterator to release reference when done
            def _wrapped_iterator():
                try:
                    for item in result_iter:
                        yield item
                finally:
                    self._release_cache_ref(speaker_id)
                    logger.debug("Released cache reference for speaker_id: %s", speaker_id)

            return _wrapped_iterator()

        except Exception as e:
            # Release reference on error
            self._release_cache_ref(speaker_id)
            logger.error("Cached inference failed: %s", e)
            raise RuntimeError(f"Cached inference failed: {e}") from e

    def synthesize_to_file(
        self,
        tts_text: str,
        prompt_text: str,
        prompt_speech_16k: Union[torch.Tensor, np.ndarray],
        output_path: Union[str, Path],
    ) -> Path:
        """
        Synthesize speech and save directly to file.

        Convenience method that combines inference and file saving in one call.

        Args:
            tts_text: Text to synthesize.
            prompt_text: Transcription of the prompt audio.
            prompt_speech_16k: Prompt audio tensor or numpy array at 16kHz.
            output_path: Path where the synthesized audio will be saved.

        Returns:
            Path: Absolute path to the saved audio file.

        Raises:
            RuntimeError: If synthesis or file saving fails.

        Example:
            >>> output = model.synthesize_to_file(
            ...     tts_text="Hello world",
            ...     prompt_text="Sample",
            ...     prompt_speech_16k=prompt_audio,
            ...     output_path="output.wav"
            ... )
            >>> print(f"Saved to {output}")
        """
        try:
            # Perform inference
            result_iter = self.inference_zero_shot(
                tts_text=tts_text,
                prompt_text=prompt_text,
                prompt_speech_16k=prompt_speech_16k,
            )

            # Get first result
            result = next(result_iter)
            audio = result["tts_speech"]  # Shape: (1, T)

            # Save to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torchaudio.save(
                str(output_path),
                audio,
                self.sample_rate,
                format="wav",
                encoding="PCM_S"
            )

            logger.info("Synthesized audio saved to %s", output_path)
            return output_path.absolute()

        except Exception as e:
            logger.error("Failed to synthesize to file: %s", e)
            raise RuntimeError(f"Synthesis to file failed: {e}") from e

    def get_device(self) -> torch.device:
        """Get the device being used for inference."""
        return self.device

    def has_lora(self) -> bool:
        """Check if LoRA adapter is loaded."""
        return self.config.lora_dir is not None


# Global singleton instance
_model: Optional[TTSModel] = None


def init_model(config: Union[ModelConfig, Dict[str, Any]]) -> TTSModel:
    """
    Initialize the global TTS model singleton.

    This function should be called once during application startup to load
    the model into memory. Subsequent calls will replace the existing model.

    Args:
        config: Model configuration as ModelConfig instance or dictionary.

    Returns:
        TTSModel: Initialized model instance.

    Raises:
        RuntimeError: If model initialization fails.

    Example:
        >>> # During FastAPI startup
        >>> config = {
        ...     "base_model": "pretrained_models/CosyVoice2-0.5B",
        ...     "lora_dir": "lora_weights/cv2/ja/checkpoint-20000",
        ...     "use_cpu": False,
        ...     "fp16": False,
        ... }
        >>> init_model(config)

    Warning:
        This function is NOT thread-safe during initialization.
        Ensure it's called before any concurrent requests.
    """
    global _model

    logger.info("Initializing global TTS model singleton")
    _model = TTSModel(config)
    logger.info("Global model initialized successfully")

    return _model


def get_model() -> TTSModel:
    """
    Dependency injection function to get the global TTS model instance.

    This function is designed to be used with FastAPI's Depends() for
    dependency injection in route handlers.

    Returns:
        TTSModel: The global model instance.

    Raises:
        RuntimeError: If model has not been initialized with init_model().

    Example:
        >>> from fastapi import Depends
        >>>
        >>> @app.post("/synthesize")
        >>> async def synthesize_endpoint(
        ...     text: str,
        ...     model: TTSModel = Depends(get_model)
        ... ):
        ...     result = model.inference_zero_shot(...)
        ...     return result

    Note:
        This function is thread-safe and can be called concurrently
        from multiple FastAPI workers.
    """
    if _model is None:
        logger.error("Model not initialized. Call init_model() first.")
        raise RuntimeError(
            "TTS model not initialized. Call init_model() during application startup."
        )

    return _model


def reset_model() -> None:
    """
    Reset the global model singleton (primarily for testing).

    This function clears the global model instance, freeing GPU memory.
    Mainly useful for testing or hot-reloading scenarios.

    Warning:
        This will cause get_model() to fail until init_model() is called again.

    Example:
        >>> # In test teardown
        >>> reset_model()
        >>> torch.cuda.empty_cache()
    """
    global _model

    if _model is not None:
        logger.info("Resetting global TTS model")
        _model = None

        # Free CUDA memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
