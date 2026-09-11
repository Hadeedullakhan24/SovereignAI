"""Unit tests for the loader framework components."""

from pathlib import Path
import pytest

from rag_engine.loaders.base_loader import BaseLoader
from rag_engine.loaders.exceptions import (
    CorruptedDocumentError,
    UnsupportedFormatError,
    ValidationFailedError,
)
from rag_engine.loaders.loader_events import (
    DocumentLoaded,
    EventBus,
    FallbackActivated,
    ValidationFailed,
)
from rag_engine.loaders.loader_factory import LoaderFactory
from rag_engine.loaders.loader_health import LoaderHealthStatus
from rag_engine.loaders.loader_metrics import LoadMetrics, MetricsCollector
from rag_engine.loaders.loader_registry import LoaderRegistry, register_loader
from rag_engine.loaders.plugin_loader import PluginLoaderManager
from rag_engine.loaders.processing_context import ProcessingContext


class DummyLoader(BaseLoader):
    """Test loader implementation."""
    def supported_formats(self) -> list[str]:
        return [".dummy"]

    def is_primary_driver_available(self) -> bool:
        return True

    def _load_primary(self, file_path: Path):
        return "dummy primary content", {"test": True}

    def _load_fallback(self, file_path: Path):
        return "dummy fallback content", {"test": False}


def test_registry_registration():
    reg = LoaderRegistry()
    reg.register(".dummy", DummyLoader)
    assert reg.get_by_extension(".dummy") is DummyLoader
    assert reg.get_by_extension("dummy") is DummyLoader
    assert ".dummy" in reg.get_supported_extensions()


def test_factory_dispatch_and_error(tmp_path):
    reg = LoaderRegistry()
    reg.register(".dummy", DummyLoader)
    factory = LoaderFactory(registry=reg)

    f = tmp_path / "test.dummy"
    f.write_text("sample dummy text", encoding="utf-8")

    loader = factory.get_loader(f)
    assert isinstance(loader, DummyLoader)

    bad_f = tmp_path / "unknown.xyz"
    bad_f.write_text("data", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        factory.get_loader(bad_f)


def test_processing_context_propagation(tmp_path):
    reg = LoaderRegistry()
    reg.register(".dummy", DummyLoader)
    factory = LoaderFactory(registry=reg)

    f = tmp_path / "test.dummy"
    f.write_text("sample content", encoding="utf-8")

    ctx = ProcessingContext(category="safety_docs", subcategory="sop")
    doc = factory.load(f, context=ctx)

    assert doc.metadata.category == "safety_docs"
    assert doc.metadata.subcategory == "sop"
    assert doc.content == "dummy primary content"
    assert len(doc.metadata.processing_history) == 1
    assert doc.metadata.processing_history[0]["loader"] == "DummyLoader"


def test_metrics_collector():
    collector = MetricsCollector()
    m1 = LoadMetrics(file_path="a.pdf", loader_name="PDFLoader", duration_ms=10.0, success=True)
    m2 = LoadMetrics(file_path="b.pdf", loader_name="PDFLoader", duration_ms=20.0, success=False)
    collector.record(m1)
    collector.record(m2)

    summary = collector.get_summary()
    assert summary["total_operations"] == 2
    assert summary["successful_operations"] == 1
    assert summary["success_rate_pct"] == 50.0
    assert summary["avg_duration_ms"] == 15.0


def test_event_bus():
    bus = EventBus()
    received_events = []

    def on_loaded(event):
        received_events.append(event)

    bus.subscribe(DocumentLoaded, on_loaded)
    ev = DocumentLoaded(file_path="doc.pdf", loader_name="PDFLoader", doc_id="123")
    bus.publish(ev)

    assert len(received_events) == 1
    assert received_events[0].doc_id == "123"


def test_loader_health():
    loader = DummyLoader()
    report = loader.health()
    assert report.loader_name == "DummyLoader"
    assert report.status == LoaderHealthStatus.HEALTHY
    assert report.primary_driver_available is True
    assert ".dummy" in report.supported_formats


def test_plugin_loader_registration(tmp_path):
    plugin_py = tmp_path / "custom_plugin.py"
    plugin_py.write_text("""
from rag_engine.loaders.base_loader import BaseLoader

class PluginXML(BaseLoader):
    def supported_formats(self):
        return [".customxml"]
    def is_primary_driver_available(self):
        return True
    def _load_primary(self, path):
        return "<xml>custom</xml>", {}
    def _load_fallback(self, path):
        return "<xml>fallback</xml>", {}
""", encoding="utf-8")

    reg = LoaderRegistry()
    manager = PluginLoaderManager(registry=reg)
    formats = manager.load_plugins_from_file(plugin_py)

    assert ".customxml" in formats
    loader_cls = reg.get_by_extension(".customxml")
    assert loader_cls is not None
    assert loader_cls.__name__ == "PluginXML"
