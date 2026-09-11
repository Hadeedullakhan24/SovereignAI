"""Prompt Health Monitor and canary diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any

from rag_engine.generation.prompt.prompt_builder import PromptBuilder
from rag_engine.generation.prompt.prompt_templates import PromptArchetype, PromptTemplateRegistry
from rag_engine.generation.prompt.token_budget_manager import TokenBudgetManager


@dataclass(frozen=True)
class PromptHealthReport:
    """Diagnostic report assessing operational health of prompt synthesis."""

    status: str
    is_healthy: bool
    canary_latency_ms: float
    templates_count: int
    archetypes_healthy: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: dict[str, Any] = field(default_factory=dict)


class PromptHealthMonitor:
    """Runs periodic canary syntheses to verify template rendering and budget math."""

    def __init__(
        self,
        builder: PromptBuilder | None = None,
        registry: PromptTemplateRegistry | None = None,
    ) -> None:
        self.builder = builder or PromptBuilder()
        self.registry = registry or PromptTemplateRegistry()

    def check_health(self) -> PromptHealthReport:
        """Execute active canary test across all archetypes."""
        t0 = time.perf_counter()
        archetypes = list(PromptArchetype)
        healthy_archetypes: list[str] = []
        errors: dict[str, str] = {}

        for arc in archetypes:
            try:
                # Test synthesis with dummy candidates
                payload = self.builder.build_prompt(
                    query="Test refinery pressure question",
                    candidates=[],
                    citations=[],
                    archetype=arc,
                )
                assert payload.prompt_hash, "Missing prompt hash"
                assert payload.prompt_text, "Empty prompt text"
                healthy_archetypes.append(arc.value)
            except Exception as e:
                errors[arc.value] = str(e)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        is_healthy = len(errors) == 0

        return PromptHealthReport(
            status="HEALTHY" if is_healthy else "DEGRADED",
            is_healthy=is_healthy,
            canary_latency_ms=round(latency_ms, 2),
            templates_count=len(archetypes),
            archetypes_healthy=healthy_archetypes,
            details={"errors": errors} if errors else {},
        )
