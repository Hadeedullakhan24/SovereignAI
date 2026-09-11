"""Custom exception hierarchy for Milestone 6 Offline Embedding Pipeline."""

class EmbeddingError(Exception):
    """Base exception for all embedding pipeline errors."""
    pass

class ModelNotFoundError(EmbeddingError):
    """Raised when an embedding model cannot be found in the local cache/models directory."""
    pass

class ModelIntegrityError(EmbeddingError):
    """Raised when local model files are corrupt, incomplete, or fail checksum validation."""
    pass

class VectorValidationError(EmbeddingError):
    """Raised when an embedding vector fails dimension, NaN/Inf, or normalization checks."""
    pass

class CacheError(EmbeddingError):
    """Raised when an error occurs while accessing or persisting the vector cache."""
    pass

class CheckpointError(EmbeddingError):
    """Raised when checkpoint loading or saving fails."""
    pass

class DeviceError(EmbeddingError):
    """Raised when an invalid or unsupported hardware device is specified."""
    pass

