"""Local FastAPI service for SovereignAI Workbench (SIH26117 / MRPL)."""
from __future__ import annotations

from backend.config import BackendSettings, get_settings, settings
from backend.app import app

__all__ = ["app", "settings", "BackendSettings", "get_settings"]
