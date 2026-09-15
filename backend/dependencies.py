"""FastAPI dependency injection module for Sovereign AI Workbench (Phase 1 Foundation).

Provides shared dependencies for settings, authentication, and database access
designed to be expanded in subsequent implementation phases.
"""
from __future__ import annotations

from typing import Generator
from fastapi import Request

from backend.config import BackendSettings, get_settings


def get_app_settings() -> BackendSettings:
    """Dependency provider for application settings."""
    return get_settings()


def current_user(request: Request) -> str:
    """Dependency provider resolving current authenticated user from session token.

    Note: In Phase 1 this delegates to the existing session check implementation in app.py
    for backwards compatibility.
    """
    from backend.app import current_user as _app_current_user
    return _app_current_user(request)
