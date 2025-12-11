"""Configuration management for UtterTune FastAPI server.

This module provides Pydantic Settings-based configuration for the TTS server,
supporting environment variable overrides with the UTTERTUNE_ prefix.

Classes:
    ServerConfig: Main configuration class for server settings.

Example:
    >>> from scripts.cv2.server.config import ServerConfig
    >>> config = ServerConfig()
    >>> print(config.base_model)
    pretrained_models/CosyVoice2-0.5B

    # Override via environment variables:
    # export UTTERTUNE_PORT=8080
    # export UTTERTUNE_FP16=true
    >>> config = ServerConfig()
    >>> print(config.port)
    8080

Note:
    Environment variables must be prefixed with UTTERTUNE_ to be recognized.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseSettings):
    """Configuration settings for the UtterTune TTS server.

    This class manages all server configuration parameters including model paths,
    server settings, and inference options. Values can be overridden using
    environment variables with the UTTERTUNE_ prefix.

    Attributes:
        base_model: Path to the base CosyVoice2 model directory.
        lora_dir: Optional path to LoRA adaptation weights directory.
        host: Server bind address (0.0.0.0 for all interfaces).
        port: Server port number for HTTP connections.
        fp16: Whether to use half-precision (FP16) inference for speed.
        use_cpu: Force CPU inference even if GPU is available.
        seed: Random seed for reproducible audio generation.
        default_speed: Default speech speed multiplier (1.0 = normal).
        max_prompt_duration: Maximum allowed prompt audio duration in seconds.
        max_concurrent_requests: Maximum number of concurrent TTS requests.

    Example:
        >>> config = ServerConfig(port=8080, fp16=True)
        >>> print(config.port, config.fp16)
        8080 True

        >>> # From environment
        >>> import os
        >>> os.environ["UTTERTUNE_HOST"] = "127.0.0.1"
        >>> config = ServerConfig()
        >>> print(config.host)
        127.0.0.1
    """

    model_config = SettingsConfigDict(
        env_prefix="UTTERTUNE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model configuration
    base_model: str = "pretrained_models/CosyVoice2-0.5B"
    lora_dir: Optional[Path] = None

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 50000

    # Inference configuration
    fp16: bool = False
    use_cpu: bool = False
    seed: int = 42

    # Model optimizations (for low-latency streaming)
    load_jit: bool = False  # JIT-compiled flow encoder
    load_trt: bool = False  # TensorRT flow decoder
    load_vllm: bool = False  # vLLM for LLM inference
    vllm_model_dir: Optional[Path] = None  # Custom vLLM model directory (for merged LoRA)
    trt_concurrent: int = 1  # TensorRT concurrent streams

    # TTS parameters
    default_speed: float = 1.0
    max_prompt_duration: float = 30.0

    # Performance configuration
    max_concurrent_requests: int = 2

    def get_model_path(self) -> Path:
        """Get the resolved absolute path to the base model.

        Converts the base_model string to an absolute Path object,
        resolving relative paths from the current working directory.

        Returns:
            Absolute path to the base model directory.

        Example:
            >>> config = ServerConfig()
            >>> model_path = config.get_model_path()
            >>> print(model_path.is_absolute())
            True
        """
        return Path(self.base_model).resolve()

    def get_lora_path(self) -> Optional[Path]:
        """Get the resolved absolute path to the LoRA directory if configured.

        Converts the lora_dir to an absolute Path object if set,
        otherwise returns None.

        Returns:
            Absolute path to LoRA directory, or None if not configured.

        Example:
            >>> config = ServerConfig(lora_dir=Path("./lora"))
            >>> lora_path = config.get_lora_path()
            >>> print(lora_path.is_absolute() if lora_path else None)
            True
        """
        if self.lora_dir is None:
            return None
        return self.lora_dir.resolve()

    def validate_speed(self, speed: float) -> float:
        """Validate and clamp speech speed to acceptable range.

        Ensures the speed multiplier is within safe operational bounds
        to prevent audio quality degradation or processing errors.

        Args:
            speed: Requested speech speed multiplier.

        Returns:
            Clamped speed value between 0.5 and 2.0.

        Raises:
            ValueError: If speed is not a positive number.

        Example:
            >>> config = ServerConfig()
            >>> print(config.validate_speed(1.5))
            1.5
            >>> print(config.validate_speed(5.0))
            2.0
            >>> print(config.validate_speed(0.1))
            0.5
        """
        if speed <= 0:
            raise ValueError(f"Speed must be positive, got {speed}")
        return max(0.5, min(2.0, speed))

    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is enabled based on configuration.

        Returns:
            True if GPU should be used (not forced to CPU), False otherwise.

        Example:
            >>> config = ServerConfig(use_cpu=False)
            >>> print(config.is_gpu_enabled())
            True
            >>> config = ServerConfig(use_cpu=True)
            >>> print(config.is_gpu_enabled())
            False
        """
        return not self.use_cpu
