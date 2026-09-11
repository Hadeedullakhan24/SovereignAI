"""Prompt Templates re-export for top-level generation access."""

from rag_engine.generation.prompt.prompt_templates import (
    PromptArchetype,
    PromptRegistry,
    PromptTemplate,
    PromptTemplateRegistry,
    TEMPLATES,
)

__all__ = [
    "PromptArchetype",
    "PromptTemplate",
    "PromptTemplateRegistry",
    "PromptRegistry",
    "TEMPLATES",
]
