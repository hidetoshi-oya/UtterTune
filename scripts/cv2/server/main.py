#!/usr/bin/env python3
"""
UtterTune FastAPI Server - Main Entry Point
============================================

This module provides the main FastAPI application for the UtterTune text-to-speech
system with CosyVoice2 and optional LoRA adapter support.

The server provides RESTful endpoints for:
    - Zero-shot voice cloning (TTS synthesis)
    - Health checks and readiness probes
    - Model information and status

Key Features:
    - CosyVoice2 base model with optional LoRA fine-tuning
    - Zero-shot voice cloning from audio prompts
    - Phonetic annotation support (<PHON_START>...<PHON_END>)
    - CORS support for cross-origin requests
    - Proper lifecycle management with startup/shutdown hooks
    - Comprehensive logging and error handling

Architecture:
    - FastAPI for async HTTP server
    - Lifespan context manager for resource management
    - Dependency injection for model access
    - Pydantic for configuration and validation

Usage Examples:
    Basic usage with base model only:
        ```bash
        python -m scripts.cv2.server.main \\
            --base_model pretrained_models/CosyVoice2-0.5B \\
            --host 0.0.0.0 \\
            --port 50000
        ```

    With LoRA adapter for fine-tuned voice:
        ```bash
        python -m scripts.cv2.server.main \\
            --base_model pretrained_models/CosyVoice2-0.5B \\
            --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \\
            --port 50000 \\
            --fp16
        ```

    CPU-only mode for debugging:
        ```bash
        python -m scripts.cv2.server.main \\
            --base_model pretrained_models/CosyVoice2-0.5B \\
            --cpu \\
            --port 50000
        ```

    Using environment variables:
        ```bash
        export UTTERTUNE_BASE_MODEL=pretrained_models/CosyVoice2-0.5B
        export UTTERTUNE_PORT=8080
        export UTTERTUNE_FP16=true
        python -m scripts.cv2.server.main
        ```

API Endpoints:
    Health Checks:
        GET  /health/        - Basic health check (always returns 200 if server is up)
        GET  /health/ready   - Readiness probe (returns 200 only if model is loaded)
        GET  /health/model/info - Detailed model information

    Text-to-Speech:
        POST /v1/tts/        - Zero-shot voice cloning synthesis
            Form parameters:
                - tts_text: Text to synthesize
                - prompt_text: Transcription of prompt audio
                - prompt_wav: Audio file (< 4s recommended)
            Query parameters:
                - speed: Speech speed (0.5-2.0, default: 1.0)
                - stream: Enable streaming (true/false, default: false)
                - format: Output format (wav/pcm, default: wav)
        WS   /v1/tts/ws      - WebSocket for real-time TTS

Client Examples:
    Using curl:
        ```bash
        # Health check
        curl http://localhost:50000/health/

        # TTS synthesis (non-streaming WAV)
        curl -X POST "http://localhost:50000/v1/tts/?format=wav" \\
            -F "tts_text=Hello, this is a test." \\
            -F "prompt_text=Sample prompt text" \\
            -F "prompt_wav=@prompt.wav" \\
            --output output.wav

        # TTS synthesis (streaming PCM)
        curl -X POST "http://localhost:50000/v1/tts/?stream=true&format=pcm" \\
            -F "tts_text=Hello, this is a test." \\
            -F "prompt_text=Sample prompt text" \\
            -F "prompt_wav=@prompt.wav" \\
            --output - | play -t raw -r 22050 -e signed -b 16 -c 1 -
        ```

    Using Python requests:
        ```python
        import requests

        # TTS synthesis
        with open("prompt.wav", "rb") as f:
            files = {"prompt_wav": f}
            data = {
                "tts_text": "Hello world",
                "prompt_text": "Sample prompt"
            }
            response = requests.post(
                "http://localhost:50000/v1/tts/",
                files=files,
                data=data,
                params={"format": "wav"}
            )
            with open("output.wav", "wb") as out:
                out.write(response.content)
        ```

Configuration:
    All settings can be configured via:
        1. Command-line arguments (highest priority)
        2. Environment variables with UTTERTUNE_ prefix
        3. .env file in the working directory
        4. Default values (lowest priority)

    Key settings:
        - base_model: Path to CosyVoice2 base model
        - lora_dir: Optional path to LoRA adapter
        - host: Server bind address (default: 0.0.0.0)
        - port: Server port (default: 50000)
        - fp16: Use half-precision inference (faster but less accurate)
        - cpu: Force CPU inference (useful for debugging)
        - seed: Random seed for reproducibility (default: 42)

Performance:
    - GPU inference: ~0.5-2s per sentence (depending on length)
    - CPU inference: ~5-20s per sentence (significantly slower)
    - Memory requirements: ~2-4GB GPU VRAM for base model
    - Concurrent requests: Configurable max_concurrent_requests

Author: UtterTune Team
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scripts.cv2.server.config import ServerConfig
from scripts.cv2.server.dependencies import init_model, reset_model
from scripts.cv2.server.routers import health, tts
from scripts.cv2.server.routers.health import set_model_state
from scripts.cv2.server.routers.tts import set_max_concurrent_requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress noisy matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan context manager for FastAPI application.

    Handles initialization and cleanup of resources during server startup
    and shutdown. This ensures proper model loading and resource cleanup.

    Lifecycle:
        Startup:
            1. Log server startup with configuration
            2. Initialize CosyVoice2 model with optional LoRA adapter
            3. Update model state for health check endpoints
            4. Log model information (device, paths, sample rate)

        Shutdown:
            1. Log shutdown initiation
            2. Reset model and free GPU memory
            3. Update model state to reflect unloaded status
            4. Log cleanup completion

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control returns to FastAPI to handle requests.

    Raises:
        RuntimeError: If model initialization fails during startup.

    Example:
        This function is used internally by create_app() and should not
        be called directly:
            >>> app = create_app(config)
            >>> # lifespan is automatically managed by FastAPI
    """
    # --- STARTUP ---
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 UtterTune FastAPI Server Starting")
    logger.info("=" * 80)

    config: ServerConfig = app.state.config

    logger.info("Configuration:")
    logger.info("  Base Model: %s", config.base_model)
    logger.info("  LoRA Dir:   %s", config.lora_dir or "None (base model only)")
    logger.info("  Host:       %s", config.host)
    logger.info("  Port:       %s", config.port)
    logger.info("  FP16:       %s", config.fp16)
    logger.info("  CPU Mode:   %s", config.use_cpu)
    logger.info("  Seed:       %s", config.seed)
    logger.info("  Max Concurrent: %s", config.max_concurrent_requests)
    logger.info("Model Optimizations:")
    logger.info("  JIT Flow Encoder:  %s", "Enabled" if config.load_jit else "Disabled")
    logger.info("  TensorRT Decoder:  %s", "Enabled" if config.load_trt else "Disabled")
    logger.info("  vLLM Inference:    %s", "Enabled" if config.load_vllm else "Disabled")
    if config.vllm_model_dir:
        logger.info("  vLLM Model Dir:    %s", config.vllm_model_dir)
    if config.load_trt:
        logger.info("  TRT Concurrent:    %s", config.trt_concurrent)

    # Configure concurrency limit from config
    set_max_concurrent_requests(config.max_concurrent_requests)

    try:
        # Initialize the TTS model
        logger.info("-" * 80)
        logger.info("📦 Initializing CosyVoice2 model...")

        model_config = {
            "base_model": config.base_model,
            "lora_dir": str(config.lora_dir) if config.lora_dir else None,
            "use_cpu": config.use_cpu,
            "fp16": config.fp16,
            # Model optimizations for low-latency streaming
            "load_jit": config.load_jit,
            "load_trt": config.load_trt,
            "load_vllm": config.load_vllm,
            "vllm_model_dir": str(config.vllm_model_dir) if config.vllm_model_dir else None,
            "trt_concurrent": config.trt_concurrent,
        }

        model = init_model(model_config)

        # Update health check state
        set_model_state(
            loaded=True,
            model_instance=model,
            base_model_path=config.base_model,
            lora_adapter_path=str(config.lora_dir) if config.lora_dir else None,
            device=str(model.get_device()),
            start_time=start_time,
        )

        logger.info("-" * 80)
        logger.info("✅ Model initialization complete")
        logger.info("  Device:      %s", model.get_device())
        logger.info("  Sample Rate: %d Hz", model.sample_rate)
        logger.info("  LoRA Active: %s", "Yes" if model.has_lora() else "No")
        logger.info("=" * 80)
        logger.info("🎤 Server ready to accept requests")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("❌ Failed to initialize model: %s", e, exc_info=True)
        logger.error("=" * 80)
        raise RuntimeError(f"Model initialization failed: {e}") from e

    # --- YIELD TO APPLICATION ---
    yield

    # --- SHUTDOWN ---
    logger.info("=" * 80)
    logger.info("🛑 UtterTune FastAPI Server Shutting Down")
    logger.info("=" * 80)

    try:
        logger.info("Cleaning up model resources...")
        reset_model()

        # Update health check state
        set_model_state(
            loaded=False,
            model_instance=None,
            base_model_path=None,
            lora_adapter_path=None,
            device="cpu",
            start_time=None,
        )

        logger.info("✅ Cleanup complete")

    except Exception as e:
        logger.error("⚠️  Error during cleanup: %s", e, exc_info=True)

    logger.info("=" * 80)
    logger.info("👋 Server shutdown complete")
    logger.info("=" * 80)


def create_app(config: ServerConfig) -> FastAPI:
    """
    Create and configure the FastAPI application.

    This factory function creates a FastAPI instance with all necessary
    middleware, routers, and configuration. It uses the lifespan context
    manager for proper resource management.

    Configuration:
        - CORS middleware with permissive settings (allow all origins)
        - Health check router at /health/*
        - TTS router at /tts/*
        - Lifespan management for model loading/cleanup
        - Server configuration stored in app.state

    Args:
        config: Server configuration containing model paths and settings.

    Returns:
        FastAPI: Configured FastAPI application instance ready to run.

    Example:
        >>> config = ServerConfig(
        ...     base_model="pretrained_models/CosyVoice2-0.5B",
        ...     port=50000
        ... )
        >>> app = create_app(config)
        >>> uvicorn.run(app, host=config.host, port=config.port)
    """
    # Create FastAPI app with lifespan management
    app = FastAPI(
        title="UtterTune TTS API",
        description="Text-to-speech synthesis with CosyVoice2 and LoRA adaptation",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Store config in app state for lifespan access
    app.state.config = config

    # Add CORS middleware - allow all origins for development
    # TODO: Configure allowed origins for production deployment
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    # Include routers
    app.include_router(health.router)
    app.include_router(tts.router)

    logger.info("FastAPI application created with routes:")
    logger.info("  - Health checks:  /health/*")
    logger.info("  - TTS synthesis:  /tts/*")
    logger.info("  - API docs:       /docs")
    logger.info("  - ReDoc:          /redoc")

    return app


def parse_args() -> ServerConfig:
    """
    Parse command-line arguments and create server configuration.

    This function parses CLI arguments and creates a ServerConfig instance.
    Environment variables can override default values, and CLI arguments
    take highest priority.

    Command-line Arguments:
        --base_model: Path to CosyVoice2 base model directory
        --lora_dir: Optional path to LoRA adapter directory
        --host: Server bind address (default: 0.0.0.0)
        --port: Server port number (default: 50000)
        --fp16: Enable FP16 precision (flag, default: False)
        --cpu: Force CPU inference (flag, default: False)
        --seed: Random seed for reproducibility (default: 42)

    Returns:
        ServerConfig: Configuration object with parsed arguments.

    Example:
        >>> # From command line:
        >>> # python -m scripts.cv2.server.main --port 8080 --fp16
        >>> config = parse_args()
        >>> print(config.port)
        8080
        >>> print(config.fp16)
        True
    """
    parser = argparse.ArgumentParser(
        description="UtterTune FastAPI server for CosyVoice2 + LoRA TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with base model only
  python -m scripts.cv2.server.main \\
      --base_model pretrained_models/CosyVoice2-0.5B

  # With LoRA adapter for fine-tuned voice
  python -m scripts.cv2.server.main \\
      --base_model pretrained_models/CosyVoice2-0.5B \\
      --lora_dir lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS \\
      --port 50000

  # CPU mode with custom port
  python -m scripts.cv2.server.main \\
      --base_model pretrained_models/CosyVoice2-0.5B \\
      --cpu \\
      --port 8080

Environment Variables:
  All settings can be overridden with UTTERTUNE_ prefix:
    export UTTERTUNE_BASE_MODEL=pretrained_models/CosyVoice2-0.5B
    export UTTERTUNE_PORT=8080
    export UTTERTUNE_FP16=true
        """,
    )

    # Model configuration
    parser.add_argument(
        "--base_model",
        type=str,
        default="pretrained_models/CosyVoice2-0.5B",
        help="Path to CosyVoice2 base model directory (default: pretrained_models/CosyVoice2-0.5B)",
    )

    parser.add_argument(
        "--lora_dir",
        type=Path,
        default=None,
        help="Optional path to LoRA adapter directory for fine-tuned voice",
    )

    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server bind address (default: 0.0.0.0, use 127.0.0.1 for localhost only)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=50000,
        help="Server port number (default: 50000)",
    )

    # Inference configuration
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 precision for faster inference (may reduce quality slightly)",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if GPU is available (useful for debugging)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Model optimization options (for low-latency streaming)
    parser.add_argument(
        "--jit",
        action="store_true",
        help="Load JIT-compiled flow encoder for faster inference (requires flow.encoder.*.zip)",
    )

    parser.add_argument(
        "--trt",
        action="store_true",
        help="Load TensorRT-optimized flow decoder (requires *.plan file)",
    )

    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use vLLM for LLM inference acceleration (requires vllm directory)",
    )

    parser.add_argument(
        "--vllm_model_dir",
        type=Path,
        default=None,
        help="Path to merged vLLM model directory (use with LoRA-merged models)",
    )

    args = parser.parse_args()

    # Create ServerConfig from arguments
    # Note: Pydantic Settings will also load from environment variables
    # If vllm_model_dir is specified, auto-enable vllm
    load_vllm = args.vllm or (args.vllm_model_dir is not None)

    config = ServerConfig(
        base_model=args.base_model,
        lora_dir=args.lora_dir,
        host=args.host,
        port=args.port,
        fp16=args.fp16,
        use_cpu=args.cpu,
        seed=args.seed,
        # Model optimization options
        load_jit=args.jit,
        load_trt=args.trt,
        load_vllm=load_vllm,
        vllm_model_dir=args.vllm_model_dir,
    )

    return config


def main() -> None:
    """
    Main entry point for the UtterTune FastAPI server.

    This function orchestrates the entire server lifecycle:
        1. Parse command-line arguments and load configuration
        2. Create FastAPI application with lifespan management
        3. Run server with uvicorn (async ASGI server)

    The server will:
        - Load the CosyVoice2 model on startup
        - Accept TTS synthesis requests
        - Provide health check endpoints
        - Clean up resources on shutdown

    Configuration can be provided via:
        - Command-line arguments (highest priority)
        - Environment variables (UTTERTUNE_ prefix)
        - .env file
        - Default values (lowest priority)

    Raises:
        SystemExit: If model initialization fails or invalid configuration.

    Example:
        Run directly:
            >>> python -m scripts.cv2.server.main --port 50000

        Run with custom configuration:
            >>> python -m scripts.cv2.server.main \\
            ...     --base_model pretrained_models/CosyVoice2-0.5B \\
            ...     --lora_dir lora_weights/custom \\
            ...     --port 8080 \\
            ...     --fp16
    """
    try:
        # Parse command-line arguments
        config = parse_args()

        # Create FastAPI application
        app = create_app(config)

        # Run server with uvicorn
        logger.info("Starting uvicorn server at http://%s:%d", config.host, config.port)

        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level="info",
            access_log=True,
            # Enable auto-reload in development (disable in production)
            # reload=True,
        )

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")

    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
