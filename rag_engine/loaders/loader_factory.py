"""
Loader Factory — Automated Loader Resolution & Document Loading Dispatcher.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Inspects file MIME types first, falls back to extension inspection,
and returns initialized BaseLoader instances.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Optional

from rag_engine.config.logging_config import get_logger
from rag_engine.loaders.base_loader import BaseLoader
from rag_engine.loaders.exceptions import UnsupportedFormatError
from rag_engine.loaders.loader_events import UnsupportedDocument, global_event_bus
from rag_engine.loaders.loader_registry import LoaderRegistry, global_loader_registry
from rag_engine.loaders.loader_utils import detect_mime_type
from rag_engine.loaders.processing_context import ProcessingContext
from rag_engine.schemas.document import Document

logger = get_logger("loaders.loader_factory")


class LoaderFactory:
    """Thread-safe factory for dynamically resolving and creating document loaders."""

    def __init__(self, registry: Optional[LoaderRegistry] = None) -> None:
        """
        Initialize LoaderFactory.

        Args:
            registry: LoaderRegistry instance (defaults to global_loader_registry).
        """
        self.registry = registry or global_loader_registry
        self._lock = threading.RLock()
        self._instances: dict[type, BaseLoader] = {}

    def get_loader(self, file_path: Path | str) -> BaseLoader:
        """
        Resolve the appropriate loader class for a file:
        1. MIME type verification
        2. Extension fallback
        3. Instantiate loader

        Args:
            file_path: Target document path.

        Returns:
            Instantiated BaseLoader subclass.

        Raises:
            UnsupportedFormatError: If no matching loader is registered.
        """
        path = Path(file_path).resolve()
        loader_cls = None

        with self._lock:
            # 1. Attempt MIME type resolution
            mime = detect_mime_type(path)
            if mime:
                loader_cls = self.registry.get_by_mime(mime)

            # 2. Fallback to extension resolution
            if loader_cls is None:
                ext = path.suffix.lower()
                loader_cls = self.registry.get_by_extension(ext)

            # 3. Handle unregistered format
            if loader_cls is None:
                global_event_bus.publish(
                    UnsupportedDocument(
                        file_path=str(path),
                        loader_name="LoaderFactory",
                        extension=path.suffix.lower(),
                        details={"detected_mime": mime},
                    )
                )
                raise UnsupportedFormatError(
                    f"No loader registered for format '{path.suffix}' (detected MIME: {mime})",
                    str(path),
                )

            # Reuse cached instance or instantiate new
            if loader_cls not in self._instances:
                self._instances[loader_cls] = loader_cls()

            return self._instances[loader_cls]

    def load(
        self,
        file_path: Path | str,
        context: Optional[ProcessingContext] = None,
    ) -> Document:
        """
        Convenience entrypoint to resolve the loader and load a document in one call.
        """
        loader = self.get_loader(file_path)
        return loader.load(file_path, context=context)


# Global factory singleton
global_loader_factory = LoaderFactory()
