"""Centralized backend configuration for Sovereign AI Workbench (SIH26117 / MRPL).

Provides single source of truth for runtime directories, database paths,
session lifecycle parameters, and security policies for offline/air-gapped operation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List

from rag_engine.config.runtime_paths import runtime_file, runtime_root

# Root directory of the repository
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent


@dataclass(frozen=True)
class BackendSettings:
    """Immutable backend configuration settings."""

    # Application metadata
    app_title: str = "SovereignAI Local API"
    app_version: str = "1.0"
    offline_mode: bool = True

    # Runtime and storage paths (reuses rag_engine runtime_paths mechanism)
    runtime_root_dir: Path = field(default_factory=runtime_root)
    db_path: Path = field(default_factory=lambda: runtime_file("backend", "sovereignai.db"))
    upload_dir: Path = field(default_factory=lambda: runtime_file("backend", "uploads"))
    artifact_dir: Path = field(default_factory=lambda: runtime_file("backend", "artifacts"))
    sandbox_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SOVEREIGNAI_SANDBOX_DIR", str(_PROJECT_ROOT / "workspace_sandbox"))
        ).resolve()
    )

    # Authentication & Security
    bootstrap_token_env_var: str = "SOVEREIGNAI_BOOTSTRAP_TOKEN"
    session_duration_hours: int = 8
    max_upload_size_bytes: int = 50 * 1024 * 1024  # 50 MB default ceiling

    # CORS configuration - strict local origins only (no wildcard allowed)
    cors_allowed_origins: List[str] = field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    @property
    def bootstrap_token(self) -> str:
        """Fetch bootstrap token from environment at access time."""
        return os.environ.get(self.bootstrap_token_env_var, "")


_settings: BackendSettings | None = None


def get_settings() -> BackendSettings:
    """Return the singleton BackendSettings instance."""
    global _settings
    if _settings is None:
        _settings = BackendSettings()
    return _settings


# Export standard singleton
settings = get_settings()
