"""Runtime cleaner plugin loader and manager."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
from typing import Optional, Type

from rag_engine.preprocessing.base_cleaner import BaseCleaner
from rag_engine.preprocessing.exceptions import PluginError
from rag_engine.preprocessing.registry import CleanerRegistry, get_cleaner_registry


class PluginCleanerManager:
    """Manages dynamic runtime cleaner plugins without modifying framework code."""

    def __init__(self, registry: Optional[CleanerRegistry] = None) -> None:
        self._lock = threading.RLock()
        self._registry = registry or get_cleaner_registry()
        self._loaded_plugins: dict[str, Type[BaseCleaner]] = {}

    def register_plugin(self, name: str, cleaner_cls: Type[BaseCleaner]) -> None:
        """Register a runtime plugin class directly."""
        if not issubclass(cleaner_cls, BaseCleaner):
            raise PluginError(f"Plugin {cleaner_cls} must subclass BaseCleaner")

        with self._lock:
            self._registry.register(name, cleaner_cls)
            self._loaded_plugins[name] = cleaner_cls

    def load_from_file(self, file_path: str | Path, plugin_name: Optional[str] = None) -> list[str]:
        """Dynamically load and register cleaner plugin classes from a Python file."""
        path = Path(file_path).resolve()
        if not path.is_file() or path.suffix != ".py":
            raise PluginError(f"Invalid plugin file: {path}")

        module_name = f"rag_engine.plugins.{path.stem}_{id(self)}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise PluginError(f"Could not load spec for plugin file: {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise PluginError(f"Failed to execute plugin module {path}: {exc}") from exc

        loaded_names: list[str] = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseCleaner)
                and attr is not BaseCleaner
            ):
                reg_name = plugin_name or getattr(attr, "name", attr_name.lower())
                if callable(reg_name):
                    reg_name = attr().name
                self.register_plugin(str(reg_name), attr)
                loaded_names.append(str(reg_name))

        return loaded_names

    def load_from_directory(self, dir_path: str | Path) -> list[str]:
        """Scan directory for .py plugin files and load all BaseCleaner implementations."""
        path = Path(dir_path).resolve()
        if not path.is_dir():
            raise PluginError(f"Invalid plugin directory: {path}")

        all_loaded: list[str] = []
        for py_file in sorted(path.glob("*.py")):
            if py_file.name.startswith("__"):
                continue
            loaded = self.load_from_file(py_file)
            all_loaded.extend(loaded)

        return all_loaded

    def list_plugins(self) -> dict[str, str]:
        """Return dictionary mapping plugin name to class qualified name."""
        with self._lock:
            return {
                name: f"{cls.__module__}.{cls.__qualname__}"
                for name, cls in self._loaded_plugins.items()
            }


# Global shared plugin manager
_global_plugin_manager = PluginCleanerManager()


def get_plugin_cleaner_manager() -> PluginCleanerManager:
    """Get the singleton thread-safe plugin cleaner manager."""
    return _global_plugin_manager
