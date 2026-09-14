"""Enterprise Offline Embedding Pipeline orchestrating batching, caching, checkpointing, and validation.

Converts atomic Chunk objects from Milestone 5 into dense EmbeddedChunk objects for Milestone 7 Vector DB.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Optional

from rag_engine.embeddings.checkpoint_manager import CheckpointManager
from rag_engine.embeddings.embedding_cache import EmbeddingCache
from rag_engine.embeddings.embedding_factory import EmbeddingFactory
from rag_engine.embeddings.embedding_metrics import EmbeddingMetricsCollector
from rag_engine.embeddings.vector_validator import VectorValidator
from rag_engine.interfaces.base_embedder import BaseEmbedder
from rag_engine.schemas.chunk import Chunk
from rag_engine.schemas.embedding import (
    EmbeddedChunk,
    EmbeddingBatchRequest,
    EmbeddingBatchResponse,
    EmbeddingMetrics,
    compute_vector_checksum,
)

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """Orchestrates end-to-end local embedding generation, vector caching, and checkpointing."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        models_dir: str | Path = "models/embeddings",
        cache_db_path: str | Path = "cache/embeddings/embedding_cache.db",
        checkpoint_db_path: str | Path = "cache/embeddings/checkpoints.db",
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        enable_cache: bool = True,
        enable_checkpoints: bool = True,
        strict_validation: bool = True,
        use_test_fallback: bool = True,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.device = device
        self.enable_cache = enable_cache
        self.enable_checkpoints = enable_checkpoints
        self.strict_validation = strict_validation
        self.use_test_fallback = use_test_fallback

        self.factory = EmbeddingFactory.get_instance()
        self.cache = EmbeddingCache(db_path=cache_db_path)
        self.checkpoints = CheckpointManager(db_path=checkpoint_db_path)
        self.validator = VectorValidator(strict=strict_validation)

        # Load initial embedder
        self.embedder: BaseEmbedder = self._load_embedder(model_name)
        self.model_name = self.embedder.get_model_name()
        self.dimension = self.embedder.get_dimension()
        self.model_version = self.embedder.get_model_version()

        # Determine optimal batch size: 64 on GPU, 32 on CPU if unspecified
        self.batch_size = batch_size or (64 if self.embedder.get_device() == "cuda" else 32)

        self.metrics_collector = EmbeddingMetricsCollector(
            model_name=self.model_name,
            model_version=self.model_version,
            device=self.embedder.get_device(),
        )

    def _load_embedder(self, model_name: str) -> BaseEmbedder:
        t0 = time.perf_counter()
        embedder = self.factory.get_embedder(
            model_name=model_name,
            models_dir=self.models_dir,
            device=self.device,
            use_test_fallback=self.use_test_fallback,
        )
        duration = time.perf_counter() - t0
        return embedder

    def switch_model(self, new_model_name: str) -> None:
        """Dynamically switch the active embedding model."""
        self.embedder = self._load_embedder(new_model_name)
        self.model_name = self.embedder.get_model_name()
        self.dimension = self.embedder.get_dimension()
        self.model_version = self.embedder.get_model_version()
        self.metrics_collector = EmbeddingMetricsCollector(
            model_name=self.model_name,
            model_version=self.model_version,
            device=self.embedder.get_device(),
        )
        logger.info("Successfully switched embedding model to: %s (dim: %d)", self.model_name, self.dimension)

    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk:
        """Process and embed a single Chunk object."""
        embedded = self.embed_chunks([chunk])
        return embedded[0]

    def embed_chunks(
        self,
        chunks: list[Chunk],
        job_id: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> list[EmbeddedChunk]:
        """Convert a list of Chunk objects into retrieval-optimized EmbeddedChunk objects.

        Preserves original list ordering, leverages persistent SQLite vector caching,
        and optionally updates the job checkpoint.
        """
        if not chunks:
            return []

        start_time = time.perf_counter()
        bs = batch_size or self.batch_size
        results: list[Optional[EmbeddedChunk]] = [None] * len(chunks)

        # 1. Checkpoint Resume check
        unprocessed_indices: list[int] = []
        cache_hits_count = 0
        cache_misses_count = 0

        if job_id and self.enable_checkpoints:
            completed_chunk_ids = self.checkpoints.get_completed_chunk_ids(job_id)
            completed_items = [
                (idx, chunks[idx])
                for idx in range(len(chunks))
                if chunks[idx].chunk_id in completed_chunk_ids
            ]
            if completed_items and self.enable_cache:
                hashes_to_query = [c.chunk_hash for _, c in completed_items]
                cached_vecs = self.cache.get_batch(self.model_name, hashes_to_query)
                for idx, ch in completed_items:
                    if ch.chunk_hash in cached_vecs:
                        results[idx] = self._create_embedded_chunk(
                            chunk=ch,
                            vector=cached_vecs[ch.chunk_hash],
                            validation_status="CACHED",
                        )
                        cache_hits_count += 1
                    else:
                        unprocessed_indices.append(idx)
            else:
                for idx, _ in completed_items:
                    unprocessed_indices.append(idx)

            for idx in range(len(chunks)):
                if chunks[idx].chunk_id not in completed_chunk_ids:
                    unprocessed_indices.append(idx)
            unprocessed_indices.sort()
        else:
            unprocessed_indices = list(range(len(chunks)))

        # 2. Batch cache lookup for remaining chunks
        misses_indices: list[int] = []

        if self.enable_cache and unprocessed_indices:
            logger.info(
                "Querying persistent SQLite vector cache for %d chunks...",
                len(unprocessed_indices),
            )
            hashes_to_query = [chunks[i].chunk_hash for i in unprocessed_indices]
            cached_vectors = self.cache.get_batch(self.model_name, hashes_to_query)

            for idx in unprocessed_indices:
                ch = chunks[idx]
                if ch.chunk_hash in cached_vectors:
                    vec = cached_vectors[ch.chunk_hash]
                    results[idx] = self._create_embedded_chunk(
                        chunk=ch,
                        vector=vec,
                        validation_status="CACHED",
                    )
                    cache_hits_count += 1
                else:
                    misses_indices.append(idx)
                    cache_misses_count += 1

            logger.info(
                "Cache lookup complete: %d hits, %d misses (total evaluated: %d).",
                cache_hits_count,
                len(misses_indices),
                len(unprocessed_indices),
            )
        else:
            misses_indices = unprocessed_indices
            cache_misses_count = len(unprocessed_indices)

        # 3. Embed cache misses in batches with incremental persistence
        generated_count = 0
        total_misses = len(misses_indices)
        total_batches = (total_misses + bs - 1) // bs if total_misses > 0 else 0

        if misses_indices:
            logger.info(
                "Generating embeddings for %d chunks across %d batches (batch_size=%d)...",
                total_misses,
                total_batches,
                bs,
            )

            for batch_num, i in enumerate(range(0, total_misses, bs)):
                batch_start = time.perf_counter()
                batch_indices = misses_indices[i : i + bs]
                batch_texts = [chunks[idx].content for idx in batch_indices]

                # Run inference
                batch_embeddings = self.embedder.embed_batch(batch_texts, batch_size=bs)
                generated_count += len(batch_embeddings)

                batch_cache_entries: dict[str, list[float]] = {}
                batch_completed_records: list[tuple[str, str]] = []

                # Validate, construct output, and prepare cache/checkpoint updates
                for batch_offset, vec in enumerate(batch_embeddings):
                    orig_idx = batch_indices[batch_offset]
                    ch = chunks[orig_idx]

                    # Validate vector
                    self.validator.validate_vector(
                        vec,
                        expected_dimension=self.dimension,
                        check_normalized=self.embedder.is_normalized(),
                    )

                    batch_cache_entries[ch.chunk_hash] = vec
                    batch_completed_records.append((ch.chunk_id, ch.chunk_hash))

                    results[orig_idx] = self._create_embedded_chunk(
                        chunk=ch,
                        vector=vec,
                        validation_status="VALID",
                    )

                # Persist batch immediately to cache in efficient transaction
                if self.enable_cache and batch_cache_entries:
                    self.cache.put_batch(
                        self.model_name,
                        batch_cache_entries,
                        dimension=self.dimension,
                    )

                # Persist batch immediately to checkpoints
                if job_id and self.enable_checkpoints and batch_completed_records:
                    self.checkpoints.record_completed_batch(
                        job_id, batch_completed_records
                    )

                batch_dur = time.perf_counter() - batch_start
                throughput = len(batch_texts) / max(0.001, batch_dur)
                remaining_batches = total_batches - (batch_num + 1)
                est_remaining_sec = remaining_batches * batch_dur
                logger.info(
                    "Batch %d/%d completed (%d chunks in %.2fs, %.1f chunks/sec, est remaining: %.1fs)",
                    batch_num + 1,
                    total_batches,
                    len(batch_texts),
                    batch_dur,
                    throughput,
                    est_remaining_sec,
                )

        duration = time.perf_counter() - start_time
        logger.info(
            "embed_chunks completed: %d total, %d generated, %d cached in %.2fs (%.1f chunks/sec overall).",
            len(chunks),
            generated_count,
            cache_hits_count,
            duration,
            len(chunks) / max(0.001, duration),
        )

        self.metrics_collector.record_batch(
            num_chunks=len(chunks),
            num_generated=generated_count,
            cache_hits=cache_hits_count,
            cache_misses=cache_misses_count,
            duration=duration,
        )

        return [r for r in results if r is not None]

    def embed_batch_request(self, request: EmbeddingBatchRequest) -> EmbeddingBatchResponse:
        """Process an EmbeddingBatchRequest schema."""
        start_time = time.perf_counter()
        embedded = self.embed_chunks(request.chunks, batch_size=request.batch_size)
        duration = time.perf_counter() - start_time
        metrics = self.metrics_collector.get_metrics()
        return EmbeddingBatchResponse(
            embedded_chunks=embedded,
            model_name=self.model_name,
            embedding_dimension=self.dimension,
            total_chunks=len(embedded),
            cache_hits=metrics.cache_hits,
            cache_misses=metrics.cache_misses,
            duration_seconds=duration,
        )

    def _create_embedded_chunk(
        self,
        chunk: Chunk,
        vector: list[float],
        validation_status: str = "VALID",
    ) -> EmbeddedChunk:
        """Construct validated EmbeddedChunk with full metadata inheritance."""
        return EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            chunk_hash=chunk.chunk_hash,
            embedding=vector,
            model_name=self.model_name,
            model_version=self.model_version,
            embedding_dimension=self.dimension,
            embedding_timestamp=datetime.now(timezone.utc),
            vector_checksum=compute_vector_checksum(vector),
            validation_status=validation_status,
            metadata=chunk.metadata,
            content=chunk.content,
            text_preview=chunk.content[:120].strip() if chunk.content else "",
        )

    def get_metrics(self) -> EmbeddingMetrics:
        """Return pipeline telemetry."""
        return self.metrics_collector.get_metrics()

    def get_cache_stats(self) -> dict:
        """Return cache hit/miss statistics."""
        return self.cache.get_stats()
