"""Text-to-image and image-to-image retrieval over the vision collection."""
from __future__ import annotations
from typing import Any
from rag_engine.schemas.vector_store import FieldFilter, FilterOperator, MetadataFilter
from .embedder import VisionEmbedder
from .indexer import VISION_COLLECTION_NAME
class VisionRetriever:
    def __init__(self, store: Any, embedder: VisionEmbedder) -> None: self.store, self.embedder = store, embedder
    @staticmethod
    def _filters(filters: dict[str, Any] | None) -> MetadataFilter | None:
        if not filters: return None
        supported = {"drawing_number", "equipment_tags", "image_type", "source_file", "page_number", "document_id", "image_id"}; clauses = []
        for key, value in filters.items():
            if key not in supported: raise ValueError(f"Unsupported vision filter '{key}'.")
            clauses.append(FieldFilter(field=key, operator=FilterOperator.CONTAINS if key == "equipment_tags" else FilterOperator.EQUALS, value=value))
        return MetadataFilter(must=clauses)
    def search_text(self, query: str, *, limit: int = 10, filters: dict[str, Any] | None = None) -> list[Any]: return self.store.search_vectors(VISION_COLLECTION_NAME, self.embedder.embed_text(query), limit=limit, filters=self._filters(filters))
    def search_image(self, image_path: str, *, limit: int = 10, filters: dict[str, Any] | None = None) -> list[Any]: return self.store.search_vectors(VISION_COLLECTION_NAME, self.embedder.embed_image(image_path), limit=limit, filters=self._filters(filters))
