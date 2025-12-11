"""FastAPI server package for UtterTune voice synthesis.

This package provides a RESTful API server for the UtterTune text-to-speech
system, enabling voice cloning and synthesis through HTTP endpoints.

Modules:
    config: Configuration management using Pydantic Settings.
    models: API request/response models and data validation.
    app: FastAPI application and route handlers.

Example:
    >>> from scripts.cv2.server.config import ServerConfig
    >>> from scripts.cv2.server.models import TTSRequest
    >>> config = ServerConfig()
    >>> print(config.host, config.port)
    0.0.0.0 50000

Note:
    This package requires FastAPI, Pydantic, and the CosyVoice2 model
    to be properly installed and configured.
"""

__version__ = "1.0.0"
__all__ = ["config", "models"]
