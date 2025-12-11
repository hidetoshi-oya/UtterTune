"""
Health check and model information endpoints.

This module provides endpoints for monitoring the health and readiness
of the TTS server, as well as retrieving model metadata.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field


# Response Models
class HealthResponse(BaseModel):
    """Response model for health check endpoint.

    Attributes:
        status: Health status of the service ("healthy" or "unhealthy")
        version: API version string
        uptime_seconds: Number of seconds the service has been running
    """
    status: str = Field(..., description="Service health status")
    version: str = Field(default="1.0.0", description="API version")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")


class ModelInfo(BaseModel):
    """Response model for model information endpoint.

    Attributes:
        model_name: Name of the loaded TTS model
        model_loaded: Whether the model is currently loaded in memory
        base_model_path: Path to the base model directory
        lora_adapter_path: Path to the LoRA adapter directory (if loaded)
        sample_rate: Audio sample rate in Hz
        device: Device the model is running on (cuda/cpu)
    """
    model_name: str = Field(..., description="Name of the TTS model")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    base_model_path: Optional[str] = Field(None, description="Base model directory path")
    lora_adapter_path: Optional[str] = Field(None, description="LoRA adapter path")
    sample_rate: int = Field(default=22050, description="Audio output sample rate in Hz")
    device: str = Field(default="cpu", description="Device model is running on")


# Router instance
router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)


# Global model state tracking
_model_state = {
    "loaded": False,
    "model_instance": None,
    "base_model_path": None,
    "lora_adapter_path": None,
    "device": "cpu",
    "start_time": None,
}


def get_model_state():
    """Dependency to retrieve current model state.

    Returns:
        dict: Current model state dictionary containing model metadata

    Example:
        >>> state = get_model_state()
        >>> print(state["loaded"])
        True
    """
    return _model_state


def set_model_state(
    loaded: bool,
    model_instance=None,
    base_model_path: Optional[str] = None,
    lora_adapter_path: Optional[str] = None,
    device: str = "cpu",
    start_time: Optional[float] = None,
):
    """Update the global model state.

    This function should be called after model initialization to track
    the model's status and metadata for health check endpoints.

    Args:
        loaded: Whether the model is successfully loaded
        model_instance: The loaded model instance (CosyVoice2 object)
        base_model_path: Path to the base model directory
        lora_adapter_path: Path to the LoRA adapter (if used)
        device: Device the model is running on (e.g., "cuda:0", "cpu")
        start_time: Server start timestamp (from time.time())

    Example:
        >>> import time
        >>> set_model_state(
        ...     loaded=True,
        ...     model_instance=cv2_model,
        ...     base_model_path="/models/CosyVoice2-0.5B",
        ...     device="cuda:0",
        ...     start_time=time.time()
        ... )
    """
    _model_state["loaded"] = loaded
    _model_state["model_instance"] = model_instance
    _model_state["base_model_path"] = base_model_path
    _model_state["lora_adapter_path"] = lora_adapter_path
    _model_state["device"] = device
    _model_state["start_time"] = start_time


@router.get("/", response_model=HealthResponse, summary="Health check endpoint")
async def health_check(model_state: dict = Depends(get_model_state)) -> HealthResponse:
    """Check the overall health status of the TTS service.

    This endpoint returns a simple health status indicating whether the
    service is running and responsive. It always returns 200 OK if the
    service is reachable.

    Args:
        model_state: Injected model state dependency

    Returns:
        HealthResponse: Health status information including uptime

    Example:
        ```bash
        curl http://localhost:8000/health
        ```

        Response:
        ```json
        {
            "status": "healthy",
            "version": "1.0.0",
            "uptime_seconds": 3600.5
        }
        ```
    """
    import time

    uptime = None
    if model_state.get("start_time"):
        uptime = time.time() - model_state["start_time"]

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=uptime,
    )


@router.get("/ready", summary="Readiness check endpoint")
async def ready_check(model_state: dict = Depends(get_model_state)):
    """Check if the service is ready to handle TTS requests.

    This endpoint returns HTTP 200 if the model is loaded and ready,
    or HTTP 503 (Service Unavailable) if the model is not yet loaded.
    Use this for Kubernetes readiness probes.

    Args:
        model_state: Injected model state dependency

    Returns:
        dict: Readiness status with model loading state

    Raises:
        HTTPException: 503 if model is not loaded and ready

    Example:
        ```bash
        # When ready
        curl http://localhost:8000/health/ready
        # Response: {"ready": true, "model_loaded": true}

        # When not ready (returns 503)
        curl http://localhost:8000/health/ready
        # Response: {"ready": false, "model_loaded": false, "message": "Model not loaded"}
        ```
    """
    if not model_state.get("loaded"):
        raise HTTPException(
            status_code=503,
            detail={
                "ready": False,
                "model_loaded": False,
                "message": "Model not loaded",
            },
        )

    return {
        "ready": True,
        "model_loaded": True,
    }


@router.get("/model/info", response_model=ModelInfo, summary="Get model information")
async def model_info(model_state: dict = Depends(get_model_state)) -> ModelInfo:
    """Retrieve detailed information about the loaded TTS model.

    This endpoint provides metadata about the currently loaded model,
    including paths, device, and configuration details.

    Args:
        model_state: Injected model state dependency

    Returns:
        ModelInfo: Detailed model metadata

    Raises:
        HTTPException: 503 if model is not loaded

    Example:
        ```bash
        curl http://localhost:8000/health/model/info
        ```

        Response:
        ```json
        {
            "model_name": "CosyVoice2-0.5B",
            "model_loaded": true,
            "base_model_path": "/models/CosyVoice2-0.5B",
            "lora_adapter_path": "/models/lora/ja/checkpoint-20000",
            "sample_rate": 22050,
            "device": "cuda:0"
        }
        ```
    """
    if not model_state.get("loaded"):
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    # Extract model name from base path
    base_path = model_state.get("base_model_path", "")
    model_name = base_path.split("/")[-1] if base_path else "CosyVoice2"

    # Get actual sample rate from model if available
    model_instance = model_state.get("model_instance")
    sample_rate = 22050  # CosyVoice2 default output sample rate
    if model_instance is not None:
        sample_rate = getattr(model_instance, "sample_rate", 22050)

    return ModelInfo(
        model_name=model_name,
        model_loaded=model_state["loaded"],
        base_model_path=model_state.get("base_model_path"),
        lora_adapter_path=model_state.get("lora_adapter_path"),
        sample_rate=sample_rate,
        device=model_state.get("device", "cpu"),
    )
