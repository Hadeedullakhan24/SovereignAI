"""Writable runtime locations for local, air-gapped state.

Source checkouts and packaged model/data directories may be read-only.  Runtime
databases, caches, journals and locks therefore live outside those locations by
default.  Set ``SOVEREIGNAI_RUNTIME_DIR`` to place all mutable state on an
approved local volume; no network location or cloud service is involved.
"""
from __future__ import annotations

import os
from pathlib import Path


def runtime_root() -> Path:
    configured = os.environ.get("SOVEREIGNAI_RUNTIME_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "SovereignAI" / "runtime"
    return Path.cwd() / ".sovereignai_runtime"


def runtime_file(*parts: str) -> Path:
    return runtime_root().joinpath(*parts)
