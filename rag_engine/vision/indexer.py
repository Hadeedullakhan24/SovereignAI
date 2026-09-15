"""Dedicated collection governance and indexing for visual vectors."""
from __future__ import annotations
from typing import Any
from rag_engine.schemas.vector_store import DistanceMetric, PayloadSchemaType
from rag_engine.vector_store.collection_config import CollectionConfig, HNSWConfig
from rag_engine.vector_store.exceptions import VectorDimensionMismatchError
from .embedder import VisionEmbedder
from .schemas import VisionIndexRecord
VISION_COLLECTION_NAME = "mrpl_vision_v1"
def vision_collection_config(dimension: int) -> CollectionConfig:
    return CollectionConfig(name=VISION_COLLECTION_NAME, vector_size=dimension, distance=DistanceMetric.COSINE, payload_indexes={"document_id": PayloadSchemaType.KEYWORD, "image_id": PayloadSchemaType.KEYWORD, "source_file": PayloadSchemaType.KEYWORD, "page_number": PayloadSchemaType.INTEGER, "image_type": PayloadSchemaType.KEYWORD, "routing_decision": PayloadSchemaType.KEYWORD, "drawing_number": PayloadSchemaType.KEYWORD, "equipment_tags": PayloadSchemaType.KEYWORD}, hnsw_config=HNSWConfig(m=16, ef_construct=100, on_disk=True), on_disk_payload=True)
class VisionIndexer:
    def __init__(self, store: Any, embedder: VisionEmbedder) -> None: self.store, self.embedder = store, embedder
    def ensure_collection(self) -> bool:
        config = vision_collection_config(self.embedder.get_dimension())
        if hasattr(self.store, "ensure_collection"): return self.store.ensure_collection(config)
        if not self.store.collection_exists(config.name): return self.store.create_collection(config)
        actual = self.store.get_collection_vector_size(config.name)
        if actual != config.vector_size: raise VectorDimensionMismatchError(f"Collection '{config.name}' has {actual} dimensions; expected {config.vector_size}.")
        return False
    def index_record(self, record: VisionIndexRecord) -> str:
        self.ensure_collection(); vector = self.embedder.embed_image(record.file_path)
        if len(vector) != self.embedder.get_dimension(): raise VectorDimensionMismatchError("Vision embedder returned an unexpected vector dimension.")
        payload = record.to_payload(); payload.update(model_name=self.embedder.get_model_name(), embedding_dimension=len(vector))
        self.store.upsert_vision_points(VISION_COLLECTION_NAME, [(record.point_id, vector, payload)]); return record.point_id
    def index_multimodal_result(self, result: Any, *, page_number: int | None = None) -> str | None:
        record = VisionIndexRecord.from_multimodal_result(result, page_number=page_number); return self.index_record(record) if record else None
