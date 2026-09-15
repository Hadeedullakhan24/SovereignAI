from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from rag_engine.vector_store.collection_config import CollectionConfig, VectorStoreConfig
from rag_engine.vector_store.exceptions import VectorDimensionMismatchError
from rag_engine.vector_store.qdrant_store import QdrantVectorStore
from rag_engine.vision import (
    OpenCLIPVisionEmbedder,
    VisionCapabilityUnavailable,
    VisionIndexRecord,
    VisionIndexer,
    VisionRetriever,
    vision_collection_config,
)

STAGED_MODEL_PATH = Path("models/vision/openclip-vit-b-32")
HAS_STAGED_MODEL = (STAGED_MODEL_PATH / "open_clip_model.safetensors").is_file()


class FakeVisionEmbedder:
    def get_dimension(self):
        return 3

    def get_model_name(self):
        return "fake-openclip"

    def get_metadata(self):
        return {"model_name": self.get_model_name(), "dimension": 3}

    def embed_image(self, path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        return [1.0, 0.0, 0.0] if "p101" in Path(path).stem.lower() else [0.0, 1.0, 0.0]

    def embed_text(self, text):
        return [1.0, 0.0, 0.0] if "p-101" in text.lower() else [0.0, 1.0, 0.0]


@pytest.fixture
def store():
    return QdrantVectorStore(VectorStoreConfig(storage_path=Path(":memory:")))


@pytest.fixture
def images(tmp_path):
    p101, other = tmp_path / "pid_p101.png", tmp_path / "other.png"
    Image.new("RGB", (16, 16), "red").save(p101)
    Image.new("RGB", (16, 16), "blue").save(other)
    return p101, other


def record(path: Path, tag: str = "P-101") -> VisionIndexRecord:
    return VisionIndexRecord(
        document_id="pid-002",
        image_id=path.stem,
        source_file=path.name,
        file_path=str(path),
        page_number=1,
        image_type="engineering_drawing",
        routing_decision="engineering_drawing",
        drawing_number="PID-002",
        title="Process Example",
        equipment_tags=(tag,),
        ocr_text="P-101 centrifugal pump",
        vlm_description="Visible process drawing",
        visual_observations=({"description": "Pump symbol", "confidence": 0.8},),
        confidence=0.8,
        provenance={"backend": "LocalQwenVisionBackend"},
    )


def test_collection_configuration_and_idempotency(store):
    embedder = FakeVisionEmbedder()
    indexer = VisionIndexer(store, embedder)
    cfg = vision_collection_config(3)
    assert cfg.name == "mrpl_vision_v1" and cfg.vector_size == 3
    assert indexer.ensure_collection() is True
    assert indexer.ensure_collection() is False
    assert store.get_collection_vector_size("mrpl_vision_v1") == 3


def test_existing_wrong_dimension_is_never_recreated(store):
    store.create_collection(CollectionConfig(name="mrpl_vision_v1", vector_size=4))
    with pytest.raises(VectorDimensionMismatchError):
        VisionIndexer(store, FakeVisionEmbedder()).ensure_collection()


def test_image_and_text_retrieval_preserve_metadata_and_text_collections(store, images):
    p101, other = images
    embedder = FakeVisionEmbedder()
    indexer = VisionIndexer(store, embedder)
    store.create_collection(CollectionConfig(name="mrpl_docs_v1", vector_size=384))
    indexer.index_record(record(p101))
    indexer.index_record(record(other, "V-201"))
    retriever = VisionRetriever(store, embedder)
    text_hits = retriever.search_text("What P&ID shows P-101?", filters={"equipment_tags": "P-101"})
    image_hits = retriever.search_image(str(p101))
    assert text_hits[0].payload["image_id"] == "pid_p101"
    assert text_hits[0].payload["provenance"]["backend"] == "LocalQwenVisionBackend"
    assert text_hits[0].payload["ocr_text"] == "P-101 centrifugal pump"
    assert image_hits[0].payload["image_id"] == "pid_p101"
    assert store.get_collection_vector_size("mrpl_docs_v1") == 384
    assert store.count_points("mrpl_docs_v1") == 0


def test_phase1_drawing_and_ocr_only_translation(images):
    p101, _ = images
    equipment = (SimpleNamespace(tag="P-101", bbox=None),)
    drawing = SimpleNamespace(
        drawing_id="pid-002",
        equipment=equipment,
        title_block=SimpleNamespace(drawing_number="PID-002", title="P&ID"),
        extraction_metadata={"source": "ocr"},
    )
    ocr_only = SimpleNamespace(
        document_id="pid-002",
        source_path=str(p101),
        routing_decision="engineering_drawing",
        drawing_analysis=drawing,
        vision_analysis=None,
        processing_metadata={"vision": {"executed": False}},
        get_full_text=lambda: "P-101 from OCR",
    )
    result = VisionIndexRecord.from_multimodal_result(ocr_only)
    assert result is not None and result.ocr_text == "P-101 from OCR" and result.vlm_description == ""
    assert result.equipment_tags == ("P-101",)


def test_vlm_enhanced_visual_translation_and_missing_image(images):
    p101, _ = images
    vision = SimpleNamespace(
        image_id="field-1",
        image_type="visual_inspection",
        caption="Pump field photograph",
        observations=(SimpleNamespace(description="No visible leak", confidence=0.7),),
        equipment=(SimpleNamespace(equipment_type="Pump", name_or_tag="P-101", bbox=None),),
        confidence=0.7,
        extraction_metadata={"model_name": "qwen2.5-vl-3b-instruct"},
    )
    result = SimpleNamespace(
        document_id="field-1",
        source_path=str(p101),
        routing_decision="visual_inspection",
        drawing_analysis=None,
        vision_analysis=vision,
        processing_metadata={},
        get_full_text=lambda: "",
    )
    record_out = VisionIndexRecord.from_multimodal_result(result)
    assert record_out and record_out.vlm_description == "Pump field photograph" and record_out.model_name == "qwen2.5-vl-3b-instruct"
    with pytest.raises(FileNotFoundError):
        FakeVisionEmbedder().embed_image(p101.parent / "missing.png")


def test_openclip_safetensors_checkpoint_detection(tmp_path):
    # 1. Directory with open_clip_model.safetensors
    dir_a = tmp_path / "model_a"
    dir_a.mkdir()
    file_a = dir_a / "open_clip_model.safetensors"
    file_a.touch()
    embedder_a = OpenCLIPVisionEmbedder(dir_a)
    assert embedder_a._weights_path() == file_a

    # 2. Directory with model.safetensors
    dir_b = tmp_path / "model_b"
    dir_b.mkdir()
    file_b = dir_b / "model.safetensors"
    file_b.touch()
    embedder_b = OpenCLIPVisionEmbedder(dir_b)
    assert embedder_b._weights_path() == file_b

    # 3. Direct file path
    direct_file = tmp_path / "custom_checkpoint.safetensors"
    direct_file.touch()
    embedder_c = OpenCLIPVisionEmbedder(direct_file)
    assert embedder_c._weights_path() == direct_file

    # 4. Default resolution when not existing
    dir_d = tmp_path / "model_d"
    embedder_d = OpenCLIPVisionEmbedder(dir_d)
    assert embedder_d._weights_path() == dir_d / "open_clip_model.safetensors"


def test_openclip_is_explicitly_unavailable_without_staged_weights(tmp_path):
    embedder = OpenCLIPVisionEmbedder(tmp_path / "missing")
    with pytest.raises(VisionCapabilityUnavailable, match="Expected local file.*No model download is attempted"):
        embedder.embed_text("P-101")

    dummy_image = tmp_path / "dummy.png"
    dummy_image.touch()
    with pytest.raises(VisionCapabilityUnavailable, match="Expected local file.*No model download is attempted"):
        embedder.embed_image(dummy_image)


def test_openclip_no_runtime_downloads(tmp_path, monkeypatch):
    def fake_download(*args, **kwargs):
        raise AssertionError("Runtime download was attempted!")

    try:
        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", fake_download)
    except Exception:
        pass

    try:
        import importlib
        open_clip_pretrained = importlib.import_module("open_clip.pretrained")
        monkeypatch.setattr(open_clip_pretrained, "download_pretrained", fake_download)
    except Exception:
        pass

    embedder = OpenCLIPVisionEmbedder(tmp_path / "nonexistent")
    with pytest.raises(VisionCapabilityUnavailable, match="No model download is attempted"):
        embedder.embed_text("sample text")


def test_openclip_validation_errors(tmp_path):
    embedder = OpenCLIPVisionEmbedder(tmp_path / "missing")
    with pytest.raises(ValueError, match="Vision text query must not be empty"):
        embedder.embed_text("")
    with pytest.raises(ValueError, match="Vision text query must not be empty"):
        embedder.embed_text("   ")

    with pytest.raises(FileNotFoundError, match="Vision source image not found"):
        embedder.embed_image(tmp_path / "missing_file.jpg")


@pytest.mark.skipif(not HAS_STAGED_MODEL, reason="Staged OpenCLIP safetensors model not available")
def test_openclip_staged_model_loading_and_512d_embeddings(tmp_path):
    embedder = OpenCLIPVisionEmbedder(STAGED_MODEL_PATH)
    assert embedder.get_dimension() == 512
    assert embedder.get_model_name() == "openclip-vit-b-32"

    meta = embedder.get_metadata()
    assert meta["model_name"] == "openclip-vit-b-32"
    assert meta["dimension"] == 512
    assert "device" in meta

    # Test real image embedding
    test_img = tmp_path / "test_drawing.png"
    Image.new("RGB", (64, 64), color=(200, 100, 50)).save(test_img)

    img_vec = embedder.embed_image(test_img)
    assert isinstance(img_vec, list)
    assert len(img_vec) == 512
    assert all(isinstance(x, float) for x in img_vec)
    img_norm = math.sqrt(sum(x * x for x in img_vec))
    assert pytest.approx(img_norm, rel=1e-4) == 1.0

    # Test real text embedding
    text_vec = embedder.embed_text("P-101 centrifugal pump in P&ID diagram")
    assert isinstance(text_vec, list)
    assert len(text_vec) == 512
    assert all(isinstance(x, float) for x in text_vec)
    text_norm = math.sqrt(sum(x * x for x in text_vec))
    assert pytest.approx(text_norm, rel=1e-4) == 1.0

    # Test cosine similarity
    cos_sim = sum(a * b for a, b in zip(img_vec, text_vec))
    assert -1.0 <= cos_sim <= 1.0


@pytest.mark.skipif(not HAS_STAGED_MODEL, reason="Staged OpenCLIP safetensors model not available")
def test_openclip_staged_model_indexer_and_retriever_e2e(store, tmp_path):
    embedder = OpenCLIPVisionEmbedder(STAGED_MODEL_PATH)
    indexer = VisionIndexer(store, embedder)
    retriever = VisionRetriever(store, embedder)

    img_a = tmp_path / "pid_pump_p101.png"
    img_b = tmp_path / "inspection_valve_v201.png"
    Image.new("RGB", (64, 64), "white").save(img_a)
    Image.new("RGB", (64, 64), "black").save(img_b)

    rec_a = record(img_a, "P-101")
    rec_b = record(img_b, "V-201")
    indexer.index_record(rec_a)
    indexer.index_record(rec_b)

    assert store.get_collection_vector_size("mrpl_vision_v1") == 512
    assert store.count_points("mrpl_vision_v1") == 2

    # Retrieval by text
    text_results = retriever.search_text("Centrifugal pump P-101", limit=5)
    assert len(text_results) == 2

    # Retrieval by image
    image_results = retriever.search_image(str(img_a), limit=5)
    assert len(image_results) == 2
    assert image_results[0].payload["image_id"] == "pid_pump_p101"
