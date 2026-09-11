"""Token Budget Manager re-export for top-level generation access."""

from rag_engine.generation.prompt.token_budget_manager import (
    BudgetAllocation,
    TokenBudgetManager,
)

__all__ = [
    "TokenBudgetManager",
    "BudgetAllocation",
]
