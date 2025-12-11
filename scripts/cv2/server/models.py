"""API request and response models for UtterTune FastAPI server.

This module defines Pydantic models for validating API requests and
structuring responses in the TTS server. All models include comprehensive
validation and type safety.

Classes:
    AudioFormat: Enum for supported audio output formats.
    TTSRequest: Request model for text-to-speech synthesis.
    ModelInfo: Response model containing model and system information.
    HealthResponse: Response model for health check endpoints.
    ErrorResponse: Standardized error response structure.

Example:
    >>> from scripts.cv2.server.models import TTSRequest, AudioFormat
    >>> request = TTSRequest(
    ...     tts_text="Hello world",
    ...     prompt_text="Reference voice sample"
    ... )
    >>> print(request.format)
    AudioFormat.wav
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AudioFormat(str, Enum):
    """Supported audio output formats for TTS responses.

    Attributes:
        wav: Standard WAV format with headers (most compatible).
        pcm: Raw PCM audio data without headers (streaming-friendly).

    Example:
        >>> format = AudioFormat.wav
        >>> print(format.value)
        wav
        >>> print(AudioFormat.pcm)
        AudioFormat.pcm
    """

    wav = "wav"
    pcm = "pcm"


class TTSRequest(BaseModel):
    """Request model for text-to-speech synthesis endpoint.

    This model validates and structures incoming TTS requests, ensuring
    all required parameters are present and valid before processing.

    Attributes:
        tts_text: The text content to synthesize into speech.
        prompt_text: Reference text that describes the target voice style.
        speed: Speech speed multiplier (0.5-2.0, default from config).
        stream: Whether to stream audio chunks incrementally.
        format: Output audio format (wav or pcm).

    Example:
        >>> request = TTSRequest(
        ...     tts_text="Hello, this is a test.",
        ...     prompt_text="A clear, professional voice",
        ...     speed=1.2,
        ...     stream=False
        ... )
        >>> print(request.tts_text)
        Hello, this is a test.
        >>> print(request.format)
        AudioFormat.wav
    """

    tts_text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to synthesize into speech",
        examples=["Hello world!", "This is a voice cloning test."],
    )

    prompt_text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Reference text describing the target voice",
        examples=["A professional male voice", "Friendly female narrator"],
    )

    speed: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier (0.5-2.0)",
        examples=[1.0, 1.2, 0.8],
    )

    stream: bool = Field(
        default=False,
        description="Enable streaming response for real-time playback",
        examples=[False, True],
    )

    format: AudioFormat = Field(
        default=AudioFormat.wav,
        description="Output audio format",
        examples=["wav", "pcm"],
    )

    @field_validator("tts_text", "prompt_text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Validate that text fields are not just whitespace.

        Args:
            v: The text value to validate.

        Returns:
            The validated text value.

        Raises:
            ValueError: If text is only whitespace.

        Example:
            >>> request = TTSRequest(
            ...     tts_text="  ",  # Only whitespace
            ...     prompt_text="Valid text"
            ... )
            Traceback (most recent call last):
            ValueError: Text cannot be empty or only whitespace
        """
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()

    class Config:
        """Pydantic model configuration."""

        use_enum_values = True
        json_schema_extra = {
            "example": {
                "tts_text": "Welcome to UtterTune voice synthesis.",
                "prompt_text": "A clear and professional voice",
                "speed": 1.0,
                "stream": False,
                "format": "wav",
            }
        }


class ModelInfo(BaseModel):
    """Model and system information response.

    Provides detailed information about the loaded model, LoRA weights,
    and system configuration for debugging and monitoring.

    Attributes:
        base_model: Name or path of the base CosyVoice2 model.
        lora_loaded: Whether LoRA adaptation weights are currently loaded.
        lora_path: Path to loaded LoRA weights, if any.
        sample_rate: Audio sample rate in Hz.
        device: Compute device being used (cuda:0, cpu, etc.).
        fp16: Whether half-precision inference is enabled.

    Example:
        >>> info = ModelInfo(
        ...     base_model="CosyVoice2-0.5B",
        ...     lora_loaded=True,
        ...     lora_path="/path/to/lora",
        ...     sample_rate=22050,
        ...     device="cuda:0",
        ...     fp16=True
        ... )
        >>> print(info.device)
        cuda:0
    """

    base_model: str = Field(
        ...,
        description="Base model name or path",
        examples=["CosyVoice2-0.5B", "pretrained_models/CosyVoice2-0.5B"],
    )

    lora_loaded: bool = Field(
        ...,
        description="Whether LoRA weights are loaded",
        examples=[True, False],
    )

    lora_path: Optional[str] = Field(
        default=None,
        description="Path to loaded LoRA weights",
        examples=["/models/lora/custom_voice", None],
    )

    sample_rate: int = Field(
        ...,
        description="Audio sample rate in Hz",
        examples=[22050, 24000, 44100],
    )

    device: str = Field(
        ...,
        description="Compute device (cuda:0, cpu, etc.)",
        examples=["cuda:0", "cpu", "mps"],
    )

    fp16: bool = Field(
        ...,
        description="Whether FP16 inference is enabled",
        examples=[True, False],
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "base_model": "CosyVoice2-0.5B",
                "lora_loaded": False,
                "lora_path": None,
                "sample_rate": 22050,
                "device": "cuda:0",
                "fp16": True,
            }
        }


class HealthResponse(BaseModel):
    """Health check response model.

    Provides service health status and basic operational information
    for load balancers and monitoring systems.

    Attributes:
        status: Service health status (healthy, unhealthy, starting).
        model_loaded: Whether the TTS model is loaded and ready.
        version: API version string.

    Example:
        >>> health = HealthResponse(
        ...     status="healthy",
        ...     model_loaded=True,
        ...     version="1.0.0"
        ... )
        >>> print(health.status)
        healthy
    """

    status: str = Field(
        ...,
        description="Service health status",
        examples=["healthy", "unhealthy", "starting"],
    )

    model_loaded: bool = Field(
        ...,
        description="Whether TTS model is loaded",
        examples=[True, False],
    )

    version: str = Field(
        ...,
        description="API version",
        examples=["1.0.0", "1.1.0"],
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0",
            }
        }


class ErrorResponse(BaseModel):
    """Standardized error response structure.

    Provides consistent error reporting across all API endpoints with
    machine-readable error codes and human-friendly messages.

    Attributes:
        error: Short error code or type.
        message: Human-readable error description.
        detail: Optional detailed error information or stack trace.

    Example:
        >>> error = ErrorResponse(
        ...     error="ValidationError",
        ...     message="Invalid speed parameter",
        ...     detail="Speed must be between 0.5 and 2.0"
        ... )
        >>> print(error.error)
        ValidationError
    """

    error: str = Field(
        ...,
        description="Error code or type",
        examples=["ValidationError", "ModelError", "RuntimeError"],
    )

    message: str = Field(
        ...,
        description="Human-readable error message",
        examples=[
            "Invalid request parameters",
            "Model inference failed",
            "Service temporarily unavailable",
        ],
    )

    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information",
        examples=[
            "Speed parameter must be between 0.5 and 2.0",
            "CUDA out of memory error occurred",
            None,
        ],
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request parameters",
                "detail": "tts_text field cannot be empty",
            }
        }
