"""Conversation Formatter for Multi-Turn Chat History."""

from __future__ import annotations

from typing import Any, Sequence

from rag_engine.interfaces.base_prompt import BaseConversationFormatter
from rag_engine.schemas.prompt import ConversationTurn


class ConversationFormatter(BaseConversationFormatter):
    """Formats structured multi-turn conversation records into prompt memory blocks."""

    def __init__(self, default_style: str = "markdown") -> None:
        self.default_style = default_style

    def format_history(
        self,
        turns: Sequence[Any],
        max_tokens: int = 500,
        format_style: str = "markdown",
    ) -> str:
        """Render turns into formatted text respecting the token ceiling."""
        if not turns:
            return ""

        style = format_style or self.default_style
        formatted_turns: list[str] = []
        token_sum = 0

        # Process in reverse (most recent turns take priority)
        reversed_turns = list(reversed(turns))

        for turn in reversed_turns:
            role = getattr(turn, "role", "user")
            content = getattr(turn, "content", str(turn))
            tokens = getattr(turn, "token_count", len(content.split()))

            if token_sum + tokens > max_tokens and formatted_turns:
                break

            if style == "chatml":
                turn_str = f"<|im_start|>{role}\n{content.strip()}\n<|im_end|>"
            elif style == "plain":
                turn_str = f"{role.capitalize()}: {content.strip()}"
            else:  # markdown
                role_label = "User" if role.lower() == "user" else "Assistant"
                turn_str = f"**{role_label}:** {content.strip()}"

            formatted_turns.append(turn_str)
            token_sum += tokens

        # Restore chronological order
        formatted_turns.reverse()
        return "\n\n".join(formatted_turns)
