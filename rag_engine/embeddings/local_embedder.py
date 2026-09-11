"""Local offline embedding model wrapper using sentence-transformers or direct HuggingFace weights.

Loads models directly from local disk directory:
    models/embeddings/<model-name>/
Ensures 100% offline air-gapped execution with zero network calls.
Also includes DeterministicTestEmbedder for high-speed offline unit testing.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct
from typing import Any, Optional

from rag_engine.embeddings.exceptions import ModelIntegrityError, ModelNotFoundError
from rag_engine.interfaces.base_embedder import BaseEmbedder
from rag_engine.schemas.embedding import EmbeddingVector


# Model metadata registry
MODEL_SPECIFICATIONS: dict[str, dict[str, Any]] = {
    "BAAI/bge-small-en-v1.5": {
        "dimension": 384,
        "local_folder": "bge-small-en-v1.5",
        "passage_prefix": "",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "version": "1.5.0",
    },
    "BAAI/bge-base-en-v1.5": {
        "dimension": 768,
        "local_folder": "bge-base-en-v1.5",
        "passage_prefix": "",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "version": "1.5.0",
    },
    "intfloat/e5-small-v2": {
        "dimension": 384,
        "local_folder": "e5-small-v2",
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "version": "2.0.0",
    },
    "intfloat/e5-base-v2": {
        "dimension": 768,
        "local_folder": "e5-base-v2",
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "version": "2.0.0",
    },
}

# Aliases
MODEL_ALIASES: dict[str, str] = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "e5-small": "intfloat/e5-small-v2",
    "e5-small-v2": "intfloat/e5-small-v2",
    "e5-base": "intfloat/e5-base-v2",
    "e5-base-v2": "intfloat/e5-base-v2",
    "test-embedder": "test-embedder-384",
}


def resolve_model_name(name_or_alias: str) -> str:
    """Normalize model alias or identifier to canonical name."""
    cleaned = name_or_alias.strip()
    return MODEL_ALIASES.get(cleaned, cleaned)


def detect_device() -> str:
    """Auto-detect optimal available hardware acceleration (cuda / cpu)."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class LocalHuggingFaceEmbedder(BaseEmbedder):
    """Embedder implementation wrapping local sentence-transformers models from disk."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        models_dir: str | Path = "models/embeddings",
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        self.raw_name = model_name
        self.model_name = resolve_model_name(model_name)
        self.models_dir = Path(models_dir)
        self.device = device or detect_device()
        self.normalize = normalize

        self.spec = MODEL_SPECIFICATIONS.get(
            self.model_name,
            {
                "dimension": 384,
                "local_folder": self.model_name.replace("/", "_"),
                "passage_prefix": "",
                "query_prefix": "",
                "version": "1.0.0",
            },
        )
        self.dimension = self.spec["dimension"]
        self.passage_prefix = self.spec.get("passage_prefix", "")
        self.query_prefix = self.spec.get("query_prefix", "")
        self.version = self.spec.get("version", "1.0.0")

        self._model = None
        self._load_model()

    def _find_local_path(self) -> Optional[Path]:
        """Locate model weights on local disk."""
        local_folder = self.spec.get("local_folder", "")
        candidate_paths = [
            self.models_dir / local_folder,
            self.models_dir / self.model_name,
            Path(self.raw_name),
        ]
        for path in candidate_paths:
            if path.exists() and (
                (path / "config.json").exists()
                or (path / "model.safetensors").exists()
                or (path / "pytorch_model.bin").exists()
            ):
                return path
        return None

    def _load_model(self) -> None:
        """Load sentence-transformers model instance strictly from local files."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ModelIntegrityError(
                "sentence-transformers is not installed in the environment."
            ) from exc

        local_path = self._find_local_path()
        if not local_path:
            raise ModelNotFoundError(
                f"Failed to load embedding model {self.model_name}: weights not found in {self.models_dir}. "
                f"Ensure weights have been downloaded via scripts/download_embedding_models.py."
            )

        try:
            # Load strictly with local_files_only=True to guarantee air-gapped security
            self._model = SentenceTransformer(
                str(local_path),
                device=self.device,
                local_files_only=True,
            )
        except Exception as e:
            raise ModelIntegrityError(
                f"Failed to load embedding model {self.model_name} from {local_path}. "
                f"Model files may be missing or corrupt. Error: {e}"
            ) from e

        if self.device == "cpu":
            try:
                import os
                import torch

                threads = min(4, max(1, os.cpu_count() or 1))
                torch.set_num_threads(threads)
            except Exception:
                pass

        # Validate dimension matches
        dim_getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        actual_dim = dim_getter() if dim_getter else None
        if actual_dim:
            self.dimension = actual_dim

    def embed(self, text: str) -> list[float]:
        """Embed single text using local model."""
        if not text:
            raise ValueError("Input text to embed must not be empty.")
        if self._model is None:
            raise ModelIntegrityError("Model is not loaded.")

        prefixed = f"{self.passage_prefix}{text}" if self.passage_prefix else text
        embedding = self._model.encode(
            prefixed,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress_bar: Optional[bool] = None,
    ) -> list[list[float]]:
        """Embed batch of texts."""
        if not texts:
            return []
        if self._model is None:
            raise ModelIntegrityError("Model is not loaded.")

        bs = batch_size or (64 if self.device == "cuda" else 32)
        prefixed_texts = (
            [f"{self.passage_prefix}{t}" for t in texts]
            if self.passage_prefix
            else texts
        )

        progress_flag = (
            show_progress_bar
            if show_progress_bar is not None
            else (len(texts) > 32)
        )

        embeddings = self._model.encode(
            prefixed_texts,
            batch_size=bs,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=progress_flag,
        )
        return embeddings.tolist()

    def embed_passage(self, passage: str) -> list[float]:
        return self.embed(passage)

    def embed_query(self, query: str) -> list[float]:
        if not query:
            raise ValueError("Query to embed must not be empty.")
        prefixed = f"{self.query_prefix}{query}" if self.query_prefix else query
        embedding = self._model.encode(
            prefixed,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def get_dimension(self) -> int:
        return self.dimension

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_version(self) -> str:
        return self.version

    def is_normalized(self) -> bool:
        return self.normalize

    def get_device(self) -> str:
        return self.device


class DeterministicTestEmbedder(BaseEmbedder):
    """High-speed deterministic mock embedder for offline test suites and CI.

    Generates stable unit-length pseudo-random vectors using SHA-256 digests over text content.
    Guarantees:
    - Zero network access
    - Zero model weight dependency
    - Exactly deterministic output
    - Valid L2-normalized float32 vectors
    """

    def __init__(
        self,
        model_name: str = "test-embedder-384",
        dimension: int = 384,
        version: str = "1.0.0",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.version = version
        self.device = device

    def embed(self, text: str) -> list[float]:
        if not text:
            raise ValueError("Input text to embed must not be empty.")

        # Generate deterministic float vector from text hash
        values: list[float] = []
        seed = f"{self.model_name}:{text}"
        step = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(f"{seed}:{step}".encode("utf-8")).digest()
            step += 1
            # Unpack 8 floats from 32 bytes
            for i in range(0, 32, 4):
                if len(values) < self.dimension:
                    int_val = struct.unpack("<i", digest[i : i + 4])[0]
                    # Scale to range [-1.0, 1.0]
                    float_val = int_val / 2147483648.0
                    values.append(float_val)

        # L2-normalize vector
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress_bar: Optional[bool] = None,
    ) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def embed_passage(self, passage: str) -> list[float]:
        return self.embed(f"passage: {passage}")

    def embed_query(self, query: str) -> list[float]:
        return self.embed(f"query: {query}")

    def get_dimension(self) -> int:
        return self.dimension

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_version(self) -> str:
        return self.version

    def is_normalized(self) -> bool:
        return True

    def get_device(self) -> str:
        return self.device

