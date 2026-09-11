"""Context Formatter & Window Builder re-export for top-level generation access."""

from rag_engine.generation.prompt.context_compressor import (
    CompressionStrategy,
    ContextCompressor,
)
from rag_engine.generation.prompt.context_window_builder import (
    ContextWindowBuilder,
)

__all__ = [
    "ContextWindowBuilder",
    "ContextCompressor",
    "CompressionStrategy",
]
