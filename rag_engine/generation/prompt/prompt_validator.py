"""Prompt Validator for safety gating, token capacity, and anchor consistency."""

from __future__ import annotations

import re
from typing import Optional

from rag_engine.interfaces.base_prompt import BasePromptValidator
from rag_engine.schemas.prompt import PromptValidationResult, RetrievedPrompt


class PromptValidator(BasePromptValidator):
    """Validates synthesized prompts prior to dispatching to local LLM."""

    INJECTION_PATTERNS = [
        r"(ignore\s+(all\s+)?(previous|prior)\s+instructions)",
        r"(disregard\s+the\s+above)",
        r"(bypass\s+safety)",
        r"(system\s+override)",
        r"(you\s+are\s+now\s+in\s+developer\s+mode)",
        r"(\bDAN\s+mode\b)",
    ]

    def __init__(self, default_max_tokens: int = 4096) -> None:
        self.default_max_tokens = default_max_tokens
        self._compiled_injection = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def validate_prompt(
        self,
        prompt: RetrievedPrompt | str,
        max_context_length: Optional[int] = None,
    ) -> PromptValidationResult:
        """Inspect prompt for safety violations, token overflows, and format defects."""
        prompt_text = prompt.prompt_text if isinstance(prompt, RetrievedPrompt) else str(prompt)
        max_limit = max_context_length or self.default_max_tokens

        errors: list[str] = []
        warnings: list[str] = []
        injection_found = False

        # 1. Token ceiling check
        tokens = len(prompt_text.split())
        if tokens > max_limit:
            errors.append(f"Prompt token count ({tokens}) exceeds maximum context ceiling ({max_limit}).")

        # 2. Injection detection
        for rgx in self._compiled_injection:
            if rgx.search(prompt_text):
                injection_found = True
                errors.append(f"Security Alert: Potential prompt injection pattern detected: '{rgx.pattern}'.")
                break

        # 3. Citation Anchor presence
        anchors = re.findall(r"\[\d+\]", prompt_text)
        if not anchors:
            warnings.append("No citation anchors ([1], [2]) detected in prompt context.")

        # 4. Empty query or context check
        if not prompt_text.strip():
            errors.append("Prompt is empty.")

        is_valid = len(errors) == 0

        return PromptValidationResult(
            is_valid=is_valid,
            total_tokens=tokens,
            max_allowed_tokens=max_limit,
            errors=errors,
            warnings=warnings,
            contains_injection_attempt=injection_found,
            has_unanchored_citations=len(warnings) > 0,
        )
