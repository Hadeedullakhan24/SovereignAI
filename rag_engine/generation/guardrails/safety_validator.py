"""Safety Validator — Prompt Injection Detection and Credential Leak Redaction.

Guards input queries against adversarial manipulation and inspects output responses
to prevent leakage of sensitive credentials or unauthorized refinery bypass directives.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import List

from rag_engine.generation.generation_exceptions import SafetyViolationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SafetyCheckResult:
    """Safety evaluation outcome."""

    is_safe: bool
    flags: List[str]
    redacted_text: str


class SafetyValidator:
    """Evaluates prompts and responses for security and operational integrity."""

    # Adversarial prompt injection markers
    _INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"disregard\s+(?:all\s+)?(?:system|safety)\s+rules", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
        re.compile(r"bypass\s+(?:all\s+)?(?:security|refinery)\s+protocols", re.IGNORECASE),
        re.compile(r"reveal\s+(?:your\s+)?(?:system\s+prompt|secret\s+key)", re.IGNORECASE),
    ]

    # Sensitive credentials patterns
    _CREDENTIAL_PATTERNS = [
        (re.compile(r"(?:api[_-]?key|secret[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}['\"]?", re.IGNORECASE), "[REDACTED_API_KEY]"),
        (re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----.*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----", re.DOTALL), "[REDACTED_PRIVATE_KEY]"),
        (re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{6,}['\"]?", re.IGNORECASE), "[REDACTED_PASSWORD]"),
    ]

    def validate_input(self, query: str) -> SafetyCheckResult:
        """Evaluate user input for prompt injection attempts."""
        flags: list[str] = []
        for pattern in self._INJECTION_PATTERNS:
            if pattern.search(query):
                flags.append(f"Prompt injection pattern detected: {pattern.pattern}")

        is_safe = len(flags) == 0
        if not is_safe:
            logger.warning("Safety Violation: Adversarial query rejected: %s", flags)

        return SafetyCheckResult(
            is_safe=is_safe,
            flags=flags,
            redacted_text=query,
        )

    def validate_output(self, text: str) -> SafetyCheckResult:
        """Redact sensitive credentials or keys from output text."""
        flags: list[str] = []
        redacted = text

        for pattern, replacement in self._CREDENTIAL_PATTERNS:
            if pattern.search(redacted):
                flags.append("Sensitive credential detected and redacted")
                redacted = pattern.sub(replacement, redacted)

        return SafetyCheckResult(
            is_safe=True,
            flags=flags,
            redacted_text=redacted,
        )
