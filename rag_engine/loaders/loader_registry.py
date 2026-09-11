"""
Loader Registry — Thread-Safe Dynamic Inversion of Control Registry.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Provides extension and MIME-based loader registration without hardcoded if/else chains.
Supports decorator registration and runtime plugin loading.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional, Type

from rag_engine.config.logging_config import get_logger
from rag_engine.loaders.base_loader import BaseLoader

logger = get_logger("loaders.loader_registry")


class LoaderRegistry:
    """Thread-safe registry for managing document loaders."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._extension_map: dict[str, Type[BaseLoader]] = {}
        self._mime_map: dict[str, Type[BaseLoader]] = {}
        self._loader_classes: set[Type[BaseLoader]] = set()

    def register(self, extension: str, loader_cls: Type[BaseLoader]) -> None:
        """
        Register a loader class for a specific file extension.

        Args:
            extension: Lowercase extension with dot (e.g. '.pdf').
            loader_cls: Concrete BaseLoader subclass.
        """
        clean_ext = extension.strip().lower()
        if not clean_ext.startswith("."):
            clean_ext = f".{clean_ext}"

        with self._lock:
            self._extension_map[clean_ext] = loader_cls
            self._loader_classes.add(loader_cls)
            logger.debug(f"Registered loader {loader_cls.__name__} for extension {clean_ext}")

    def register_mime(self, mime_type: str, loader_cls: Type[BaseLoader]) -> None:
        """
        Register a loader class for a specific MIME type.

        Args:
            mime_type: MIME type string (e.g. 'application/pdf').
            loader_cls: Concrete BaseLoader subclass.
        """
        clean_mime = mime_type.strip().lower()
        with self._lock:
            self._mime_map[clean_mime] = loader_cls
            self._loader_classes.add(loader_cls)
            logger.debug(f"Registered loader {loader_cls.__name__} for MIME {clean_mime}")

    def register_loader_class(self, loader_cls: Type[BaseLoader]) -> None:
        """
        Register a loader class for all formats returned by its supported_formats().
        """
        instance = loader_cls()
        for ext in instance.supported_formats():
            self.register(ext, loader_cls)

    def get_by_extension(self, extension: str) -> Optional[Type[BaseLoader]]:
        """Lookup loader class by file extension."""
        clean_ext = extension.strip().lower()
        if not clean_ext.startswith("."):
            clean_ext = f".{clean_ext}"
        with self._lock:
            return self._extension_map.get(clean_ext)

    def get_by_mime(self, mime_type: str) -> Optional[Type[BaseLoader]]:
        """Lookup loader class by MIME type."""
        clean_mime = mime_type.strip().lower()
        with self._lock:
            return self._mime_map.get(clean_mime)

    def get_supported_extensions(self) -> set[str]:
        """Return all currently registered extensions."""
        with self._lock:
            return set(self._extension_map.keys())

    def get_all_loaders(self) -> list[Type[BaseLoader]]:
        """Return list of all registered loader classes."""
        with self._lock:
            return list(self._loader_classes)

    def clear(self) -> None:
        """Clear all registered loaders."""
        with self._lock:
            self._extension_map.clear()
            self._mime_map.clear()
            self._loader_classes.clear()


# Global thread-safe loader registry
global_loader_registry = LoaderRegistry()


def register_loader(*formats: str) -> Callable[[Type[BaseLoader]], Type[BaseLoader]]:
    """
    Class decorator to automatically register a loader with the global registry.

    Usage:
        @register_loader(".pdf")
        class PDFLoader(BaseLoader):
            ...
    """
    def decorator(cls: Type[BaseLoader]) -> Type[BaseLoader]:
        for fmt in formats:
            global_loader_registry.register(fmt, cls)
        return cls

    return decorator
