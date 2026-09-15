"""Prompt Engineering Subsystem — Templates, Budgets, Context, Assembly, Validation, and Lineage."""

from rag_engine.generation.prompt.citation_formatter import CitationFormatter
from rag_engine.generation.prompt.context_compressor import (
    CompressionStrategy,
    ContextCompressor,
)
from rag_engine.generation.prompt.context_window_builder import (
    ContextWindowBuilder,
    PromptContextBuilder,
)
from rag_engine.generation.prompt.conversation_formatter import ConversationFormatter
from rag_engine.generation.prompt.prompt_builder import (
    PromptBuilder,
    PromptPayload,
)
from rag_engine.generation.prompt.prompt_config import PromptConfig
from rag_engine.generation.prompt.prompt_events import (
    PromptEvent,
    PromptEventBus,
    PromptEventType,
)
from rag_engine.generation.prompt.prompt_exceptions import (
    CitationFormattingError,
    ContextCompressionError,
    ConversationFormattingError,
    PromptBudgetExceededError,
    PromptException,
    PromptTemplateNotFoundError,
    PromptValidationError,
    SystemPromptError,
)
from rag_engine.generation.prompt.prompt_factory import (
    PromptFactory,
    global_prompt_factory,
)
from rag_engine.generation.prompt.prompt_health import (
    PromptHealthMonitor,
    PromptHealthReport,
)
from rag_engine.generation.prompt.prompt_metrics import (
    PromptMetrics,
    PromptMetricsCollector,
)
from rag_engine.generation.prompt.prompt_pipeline import PromptPipeline
from rag_engine.generation.prompt.prompt_templates import (
    PromptArchetype,
    PromptRegistry,
    PromptTemplate,
    PromptTemplateRegistry,
    detect_task_type,
)
from rag_engine.generation.prompt.task_intent import (
    EmailPurpose,
    OutputFormat,
    TaskIntent,
    TaskIntentClassifier,
    TaskOperation,
)
from rag_engine.generation.prompt.prompt_validator import PromptValidator
from rag_engine.generation.prompt.system_prompt_manager import SystemPromptManager
from rag_engine.generation.prompt.token_budget_manager import (
    BudgetAllocation,
    TokenBudgetManager,
)
from rag_engine.schemas.prompt import (
    ContextWindow,
    ConversationTurn,
    PromptValidationResult,
    RetrievedPrompt,
    SystemPrompt,
)

__all__ = [
    # Core Builders & Gateway
    "PromptBuilder",
    "PromptPayload",
    "RetrievedPrompt",
    "PromptPipeline",
    "PromptFactory",
    "global_prompt_factory",
    "PromptConfig",
    # Templates & Archetypes
    "PromptArchetype",
    "PromptTemplate",
    "PromptTemplateRegistry",
    "PromptRegistry",
    # Context & Token Budget
    "ContextWindowBuilder",
    "PromptContextBuilder",
    "ContextCompressor",
    "CompressionStrategy",
    "ContextWindow",
    "TokenBudgetManager",
    "BudgetAllocation",
    # Personas & Formatting
    "SystemPromptManager",
    "SystemPrompt",
    "ConversationFormatter",
    "ConversationTurn",
    "CitationFormatter",
    # Validation & Diagnostics
    "PromptValidator",
    "PromptValidationResult",
    "PromptHealthMonitor",
    "PromptHealthReport",
    # Metrics & Events
    "PromptMetrics",
    "PromptMetricsCollector",
    "PromptEvent",
    "PromptEventType",
    "PromptEventBus",
    # Exceptions
    "PromptException",
    "PromptValidationError",
    "PromptBudgetExceededError",
    "PromptTemplateNotFoundError",
    "ContextCompressionError",
    "SystemPromptError",
    "ConversationFormattingError",
    "CitationFormattingError",
]
