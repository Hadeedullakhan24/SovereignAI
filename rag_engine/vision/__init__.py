"""Offline visual-vector indexing and retrieval primitives."""
from .embedder import OpenCLIPVisionEmbedder, VisionCapabilityUnavailable, VisionEmbedder
from .image_generator import (
    DiffusionCapabilityUnavailable,
    DiffusionImageGenerator,
    ImageGenerationError,
    ImageGenerationResult,
)
from .indexer import VISION_COLLECTION_NAME, VisionIndexer, vision_collection_config
from .retriever import VisionRetriever
from .schemas import VisionIndexRecord

__all__ = [
    "OpenCLIPVisionEmbedder",
    "VisionCapabilityUnavailable",
    "VisionEmbedder",
    "DiffusionCapabilityUnavailable",
    "DiffusionImageGenerator",
    "ImageGenerationError",
    "ImageGenerationResult",
    "VISION_COLLECTION_NAME",
    "VisionIndexer",
    "VisionRetriever",
    "VisionIndexRecord",
    "vision_collection_config",
]
