"""
FastAPI routers package for UtterTune TTS server.

This package contains the API route handlers for:
- Health checks and model information
- Text-to-Speech synthesis endpoints (HTTP and WebSocket)
"""

from scripts.cv2.server.routers import health, tts

__all__ = ["health", "tts"]
