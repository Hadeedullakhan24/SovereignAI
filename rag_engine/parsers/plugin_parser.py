"""Plugin parser dynamic loader and extension manager."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import List, Type

from rag_engine.parsers.base_parser import BaseParser
from rag_engine.parsers.parser_registry import ParserRegistry, get_global_registry


class PluginParserManager:
    """Discovers, validates, and dynamically registers third-party plugin parsers."""

    def __init__(self, registry: ParserRegistry | None = None) -> None:
        self.registry = registry or get_global_registry()
        self._loaded_plugins: List[Type[BaseParser]] = []

    def load_from_directory(self, plugin_dir: Path) -> List[Type[BaseParser]]:
        """Scan directory for python modules defining BaseParser subclasses."""
        if not plugin_dir.is_dir():
            return []

        loaded: List[Type[BaseParser]] = []
        for py_file in plugin_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            parsers = self.load_from_file(py_file)
            loaded.extend(parsers)

        return loaded

    def load_from_file(self, plugin_file: Path) -> List[Type[BaseParser]]:
        """Dynamically import a Python file and register any BaseParser subclasses."""
        module_name = f"rag_engine_parser_plugin_{plugin_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        if not spec or not spec.loader:
            return []

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            return []

        discovered: List[Type[BaseParser]] = []
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseParser) and obj is not BaseParser:
                self.registry.register(obj)
                discovered.append(obj)
                self._loaded_plugins.append(obj)

        return discovered

    def get_loaded_plugins(self) -> List[Type[BaseParser]]:
        """Return all successfully loaded plugin parser classes."""
        return list(self._loaded_plugins)
