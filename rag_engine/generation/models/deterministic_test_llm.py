"""Deterministic Test LLM — Reproducible In-Memory Model for CI and Benchmarks.

Provides a predictable, high-fidelity language model implementation for testing
prompt building, context assembly, citation validation, guardrails, and streaming
without requiring gigabytes of GPU VRAM or local weight downloads.
"""

from __future__ import annotations

import re
import time
from typing import Iterator

from rag_engine.generation.generation_config import ModelInferenceConfig
from rag_engine.generation.models.base_model import BaseLocalLLM, LLMGenerationOutput


class DeterministicTestLLM(BaseLocalLLM):
    """Deterministic, repeatable language model for testing and offline CI."""

    def __init__(
        self,
        model_name: str = "deterministic_test",
        context_window_size: int = 4096,
        tokens_per_second: float = 1000.0,
    ) -> None:
        super().__init__(model_name=model_name, context_window_size=context_window_size)
        self.tokens_per_second = tokens_per_second

    def count_tokens(self, text: str) -> int:
        """Heuristic token estimation (~4 chars per token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _extract_query_and_context(self, prompt: str) -> tuple[str, list[str], list[str], str]:
        """Extract user question, context passages, and citation tags from prompt."""
        # Extract question
        question = ""
        q_match = re.search(r"(?:Question|Query|USER QUERY):\s*(.*?)(?:\n\n|\n[A-Z]|\Z)", prompt, re.DOTALL | re.IGNORECASE)
        if q_match:
            question = q_match.group(1).strip()

        # Extract verified context block
        ctx_match = re.search(r"=== VERIFIED MRPL REFINERY GROUND TRUTH CONTEXT ===\s*(.*?)(?:\n===|\Z)", prompt, re.DOTALL)
        ctx = ctx_match.group(1) if ctx_match else prompt

        # Citations strictly from context
        citations = list(dict.fromkeys(re.findall(r"\[([0-9]+)\]", ctx)))
        if not citations:
            citations = list(dict.fromkeys(re.findall(r"\[([0-9]+)\]", prompt)))

        # Extract equipment tags: prioritize question, then context
        tags = list(dict.fromkeys(re.findall(r"\b([A-Z]{1,4}-[0-9]{3,5}[A-Z]?)\b", question)))
        if not tags:
            tags = list(dict.fromkeys(re.findall(r"\b([A-Z]{1,4}-[0-9]{3,5}[A-Z]?)\b", ctx)))

        return question, citations, tags, ctx

    def _synthesize_response(self, prompt: str) -> str:
        """Synthesize a grounded answer based on query and prompt context."""
        question, citations, tags, ctx = self._extract_query_and_context(prompt)

        # Citations string
        cit_str = f" [{citations[0]}]" if citations else ""
        all_cits = " ".join(f"[{c}]" for c in citations[:2]) if citations else ""

        # Check for safety / prompt injection queries
        if "ignore previous instructions" in prompt.lower() or "reveal system prompt" in prompt.lower():
            return "I cannot comply with requests that violate MRPL security protocols or attempt to alter system guidelines."

        # Specialized refinery answers
        if "pressure" in question.lower() or "bar" in ctx.lower():
            p_match = re.search(r"([0-9]+(?:\.[0-9]+)?\s*bar)", ctx, re.IGNORECASE)
            pressure = p_match.group(1) if p_match else "15.2 bar"
            tag = tags[0] if tags else "the equipment"
            return (
                f"Based on the verified operating documentation for {tag}, the design operating pressure "
                f"is specified as {pressure}{cit_str}. All relief valves and pressure transmitters must be "
                f"calibrated according to standard inspection schedules{all_cits}."
            )

        if "temperature" in question.lower() or "degc" in ctx.lower() or "°c" in ctx.lower():
            t_match = re.search(r"([0-9]+(?:\.[0-9]+)?\s*(?:°C|degC))", ctx, re.IGNORECASE)
            temperature = t_match.group(1) if t_match else "65.0 °C"
            tag = tags[0] if tags else "the equipment"
            return (
                f"According to the inspection records for {tag}, the recorded operating temperature is {temperature}{cit_str}. "
                f"Thermal monitoring indicates normal heat dissipation within design limits{all_cits}."
            )

        if "oisd" in question.lower() or "safety" in question.lower():
            return (
                f"In compliance with OISD and refinery safety guidelines{cit_str}, proper personal protective equipment (PPE) "
                f"and lockout-tagout (LOTO) procedures must be enforced prior to maintenance interventions{all_cits}."
            )

        tag_context = f" for {tags[0]}" if tags else ""
        return (
            f"Based on the provided technical documentation{tag_context}, the requested parameters are verified "
            f"and compliant with MRPL standard operating procedures{cit_str}. Maintenance intervals must follow "
            f"standard engineering specifications{all_cits}."
        )

    def generate(
        self,
        prompt: str,
        config: ModelInferenceConfig | None = None,
    ) -> LLMGenerationOutput:
        """Generate response deterministically."""
        response_text = self._synthesize_response(prompt)
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(response_text)

        return LLMGenerationOutput(
            text=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="stop",
        )

    def stream_generate(
        self,
        prompt: str,
        config: ModelInferenceConfig | None = None,
    ) -> Iterator[str]:
        """Stream synthesized response word by word."""
        full_text = self._synthesize_response(prompt)
        words = full_text.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            yield chunk
