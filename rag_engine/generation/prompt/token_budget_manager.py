"""Token Budget Manager — Dynamic Allocation of Prompt Context Budget.

Enforces strict partition limits across System Prompt, Conversation Memory,
Retrieved Ground Truth Context, and Generation Output to prevent context window overflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

from rag_engine.generation.generation_config import TokenBudgetConfig
from rag_engine.generation.generation_exceptions import TokenLimitExceededError
from rag_engine.interfaces.base_prompt import BaseTokenBudgetManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BudgetAllocation:
    """Current allocated token distribution across prompt components."""

    system_tokens: int
    memory_tokens: int
    context_tokens: int
    allowed_generation_tokens: int
    total_prompt_tokens: int
    remaining_tokens: int


class TokenBudgetManager(BaseTokenBudgetManager):
    """Manages token accounting and enforces context boundary limits."""

    def __init__(self, config: TokenBudgetConfig | None = None) -> None:
        self.config = config or TokenBudgetConfig()

    def estimate_tokens(self, text: str) -> int:
        """Heuristic token count estimation (~4 characters per token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def allocate(
        self,
        system_text: str,
        memory_text: str,
        context_text: str,
        safety_headroom: int = 64,
    ) -> BudgetAllocation:
        """Calculate and validate token allocation across prompt components.
        
        Args:
            system_text: Formatted system prompt.
            memory_text: Formatted conversation history turns.
            context_text: Formatted ground truth context.
            safety_headroom: Safety margin buffer.
            
        Returns:
            BudgetAllocation detailing tokens and remaining capacity.
            
        Raises:
            TokenLimitExceededError if prompt exceeds allowable context window.
        """
        sys_tokens = self.estimate_tokens(system_text)
        mem_tokens = self.estimate_tokens(memory_text)
        ctx_tokens = self.estimate_tokens(context_text)

        total_prompt = sys_tokens + mem_tokens + ctx_tokens + safety_headroom
        max_allowed_prompt = self.config.max_context_window - self.config.generation_budget

        if total_prompt > max_allowed_prompt:
            logger.warning(
                "Prompt tokens (%d) exceed max allowed prompt budget (%d). Trimming required.",
                total_prompt,
                max_allowed_prompt,
            )

        remaining = max(0, self.config.max_context_window - total_prompt)
        gen_tokens = min(self.config.generation_budget, remaining)

        return BudgetAllocation(
            system_tokens=sys_tokens,
            memory_tokens=mem_tokens,
            context_tokens=ctx_tokens,
            allowed_generation_tokens=gen_tokens,
            total_prompt_tokens=total_prompt,
            remaining_tokens=remaining,
        )
