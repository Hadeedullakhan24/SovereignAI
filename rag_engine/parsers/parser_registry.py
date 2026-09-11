"""Thread-safe registry for document parsers."""

from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional, Set, Type

from rag_engine.parsers.base_parser import BaseParser
from rag_engine.parsers.exceptions import ParserError


class ParserRegistry:
    """Thread-safe registry mapping file extensions and categories to parser implementations."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._parsers_by_ext: Dict[str, Type[BaseParser]] = {}
        self._parsers_by_category: Dict[str, Type[BaseParser]] = {}
        self._registered_classes: Set[Type[BaseParser]] = set()

    def register(
        self,
        parser_cls: Type[BaseParser],
        extensions: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        override: bool = False,
    ) -> None:
        """Register a parser class for specific extensions and/or categories."""
        with self._lock:
            temp_instance = parser_cls()
            exts = extensions if extensions is not None else temp_instance.supported_formats()
            cats = categories if categories is not None else temp_instance.supported_categories()

            for ext in exts:
                clean_ext = ext.lower().strip()
                if not clean_ext.startswith("."):
                    clean_ext = f".{clean_ext}"
                if clean_ext in self._parsers_by_ext and not override:
                    # Keep existing unless override
                    pass
                else:
                    self._parsers_by_ext[clean_ext] = parser_cls

            for cat in cats:
                clean_cat = cat.lower().strip()
                if clean_cat in self._parsers_by_category and not override:
                    pass
                else:
                    self._parsers_by_category[clean_cat] = parser_cls

            self._registered_classes.add(parser_cls)

    def get_by_extension(self, extension: str) -> Optional[Type[BaseParser]]:
        """Retrieve parser registered for given file extension."""
        clean_ext = extension.lower().strip()
        if not clean_ext.startswith("."):
            clean_ext = f".{clean_ext}"
        with self._lock:
            return self._parsers_by_ext.get(clean_ext)

    def get_by_category(self, category: str) -> Optional[Type[BaseParser]]:
        """Retrieve parser registered for given document category."""
        clean_cat = category.lower().strip()
        with self._lock:
            return self._parsers_by_category.get(clean_cat)

    def list_parsers(self) -> List[Type[BaseParser]]:
        """Return all uniquely registered parser classes."""
        with self._lock:
            return list(self._registered_classes)

    def clear(self) -> None:
        """Reset registry mappings."""
        with self._lock:
            self._parsers_by_ext.clear()
            self._parsers_by_category.clear()
            self._registered_classes.clear()


# Global singleton registry instance
_GLOBAL_PARSER_REGISTRY = ParserRegistry()


def register_parser(
    extensions: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    override: bool = False,
) -> Callable[[Type[BaseParser]], Type[BaseParser]]:
    """Decorator to register a parser class into the global registry."""

    def decorator(cls: Type[BaseParser]) -> Type[BaseParser]:
        _GLOBAL_PARSER_REGISTRY.register(
            cls, extensions=extensions, categories=categories, override=override
        )
        return cls

    return decorator


def get_global_registry() -> ParserRegistry:
    """Return the global registry instance."""
    return _GLOBAL_PARSER_REGISTRY
