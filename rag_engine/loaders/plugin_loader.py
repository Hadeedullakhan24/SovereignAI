"""
Plugin Loader — Dynamic Third-Party & Extension Loader Registration.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Allows extending the loading engine at runtime with plugins from external
modules or directories without modifying core framework code.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Optional, Type

from rag_engine.config.logging_config import get_logger
from rag_engine.loaders.base_loader import BaseLoader
from rag_engine.loaders.loader_registry import LoaderRegistry, global_loader_registry

logger = get_logger("loaders.plugin_loader")


class PluginLoaderManager:
    """Discovers, imports, and registers plugin loaders dynamically."""

    def __init__(self, registry: Optional[LoaderRegistry] = None) -> None:
        self.registry = registry or global_loader_registry

    def register_plugin_class(self, loader_cls: Type[BaseLoader]) -> list[str]:
        """
        Register a concrete BaseLoader class.

        Returns:
            List of extensions registered for this plugin.
        """
        if not inspect.isclass(loader_cls) or not issubclass(loader_cls, BaseLoader):
            raise TypeError(f"Expected BaseLoader subclass, got {loader_cls}")

        instance = loader_cls()
        formats = instance.supported_formats()
        for fmt in formats:
            self.registry.register(fmt, loader_cls)

        logger.info(f"Loaded plugin loader: {loader_cls.__name__} for formats: {formats}")
        return formats

    def load_plugins_from_file(self, file_path: Path | str) -> list[str]:
        """
        Dynamically import a Python file and register any BaseLoader classes within it.
        """
        path = Path(file_path).resolve()
        if not path.is_file() or not path.suffix == ".py":
            return []

        module_name = f"sov_loader_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return []

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        registered_formats: list[str] = []
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseLoader) and obj is not BaseLoader:
                formats = self.register_plugin_class(obj)
                registered_formats.extend(formats)

        return registered_formats

    def load_plugins_from_directory(self, dir_path: Path | str) -> list[str]:
        """
        Scan a directory for plugin Python files and load all BaseLoader implementations.
        """
        directory = Path(dir_path).resolve()
        if not directory.is_dir():
            return []

        all_formats: list[str] = []
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            formats = self.load_plugins_from_file(py_file)
            all_formats.extend(formats)

        return all_formats
