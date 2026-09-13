"""Pytest configuration and environment fixtures for member3_ocr."""

# Pre-load torch on Windows before paddle to prevent DLL symbol conflict
try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pass
