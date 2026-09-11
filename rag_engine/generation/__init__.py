"""Milestone 9: Enterprise In-Process Context Assembly & Generation Engine.

Universal sovereign generation platform for SIH26117 / MRPL, integrating
in-process local Hugging Face model inference, multi-turn conversation memory,
deterministic prompt synthesis, citation provenance verification, anti-hallucination
guardrails, and persistent generation caching.
"""

from rag_engine.generation.generation_config import (
    GenerationConfig,
    GuardrailConfig,
    ModelInferenceConfig,
    TokenBudgetConfig,
)
from rag_engine.generation.generation_events import (
    GenerationEvent,
    GenerationEventBus,
    GenerationEventType,
)
from rag_engine.generation.generation_exceptions import (
    BaseGenerationException,
    CitationValidationError,
    ContextAssemblyError,
    ConversationMemoryError,
    GenerationCacheError,
    GenerationInferenceError,
    HallucinationDetectedError,
    LLMModelError,
    ModelLoadingError,
    ModelNotFoundError,
    PromptAssemblyError,
    SafetyViolationError,
    StreamingError,
    TokenLimitExceededError,
)
from rag_engine.generation.generation_health import (
    GenerationHealthMonitor,
    GenerationHealthReport,
)
from rag_engine.generation.generation_metrics import (
    GenerationMetrics,
    GenerationMetricsCollector,
)
from rag_engine.generation.generation_pipeline import (
    GenerationPipeline,
    GenerationResponse,
)
from rag_engine.generation.guardrails import (
    CitationValidationReport,
    CitationValidator,
    ConfidenceBreakdown,
    ConfidenceScorer,
    GroundingVerificationReport,
    HallucinationGuard,
    SafetyCheckResult,
    SafetyValidator,
)
from rag_engine.generation.memory import (
    CachedGeneration,
    ConversationMemory,
    GenerationCache,
    TurnRecord,
)
from rag_engine.generation.models import (
    BaseLocalLLM,
    DeterministicTestLLM,
    HuggingFaceCausalLM,
    LLMFactory,
    LLMGenerationOutput,
    LLMRegistry,
    register_llm,
)
from rag_engine.generation.prompt import (
    BudgetAllocation,
    CitationFormatter,
    ContextCompressor,
    ContextWindowBuilder,
    ConversationFormatter,
    PromptArchetype,
    PromptBuilder,
    PromptConfig,
    PromptContextBuilder,
    PromptFactory,
    PromptHealthMonitor,
    PromptMetrics,
    PromptPayload,
    PromptPipeline,
    PromptRegistry,
    PromptTemplate,
    PromptTemplateRegistry,
    PromptValidator,
    RetrievedPrompt,
    SystemPromptManager,
    TokenBudgetManager,
)
from rag_engine.generation.response_formatter import ResponseFormatter
from rag_engine.generation.streaming_manager import (
    StreamingManager,
    StreamingSessionMetrics,
)

__all__ = [
    # Config & Pipeline
    "GenerationConfig",
    "TokenBudgetConfig",
    "ModelInferenceConfig",
    "GuardrailConfig",
    "GenerationPipeline",
    "GenerationResponse",
    # Models
    "BaseLocalLLM",
    "LLMGenerationOutput",
    "HuggingFaceCausalLM",
    "DeterministicTestLLM",
    "LLMRegistry",
    "LLMFactory",
    "register_llm",
    # Prompt & Context (Milestone 9)
    "PromptArchetype",
    "PromptTemplate",
    "PromptTemplateRegistry",
    "PromptRegistry",
    "PromptFactory",
    "PromptPipeline",
    "PromptConfig",
    "TokenBudgetManager",
    "BudgetAllocation",
    "ContextWindowBuilder",
    "PromptContextBuilder",
    "ContextCompressor",
    "SystemPromptManager",
    "ConversationFormatter",
    "CitationFormatter",
    "PromptValidator",
    "PromptBuilder",
    "PromptPayload",
    "RetrievedPrompt",
    "PromptMetrics",
    "PromptHealthMonitor",
    # Memory & Cache
    "ConversationMemory",
    "TurnRecord",
    "GenerationCache",
    "CachedGeneration",
    # Guardrails
    "CitationValidator",
    "CitationValidationReport",
    "HallucinationGuard",
    "GroundingVerificationReport",
    "SafetyValidator",
    "SafetyCheckResult",
    "ConfidenceScorer",
    "ConfidenceBreakdown",
    # Streaming & Formatting
    "StreamingManager",
    "StreamingSessionMetrics",
    "ResponseFormatter",
    # Observability
    "GenerationMetrics",
    "GenerationMetricsCollector",
    "GenerationEvent",
    "GenerationEventType",
    "GenerationEventBus",
    "GenerationHealthMonitor",
    "GenerationHealthReport",
    # Exceptions
    "BaseGenerationException",
    "PromptAssemblyError",
    "ContextAssemblyError",
    "TokenLimitExceededError",
    "SafetyViolationError",
    "HallucinationDetectedError",
    "CitationValidationError",
    "LLMModelError",
    "ModelNotFoundError",
    "ModelLoadingError",
    "GenerationInferenceError",
    "GenerationCacheError",
    "StreamingError",
    "ConversationMemoryError",
]
