"""Generation Metrics — Telemetry and Performance Aggregator.

Collects high-resolution timings, token accounting, TTFT (Time-To-First-Token),
tokens/second throughput, memory RSS, CPU utilization, cache efficiency, and guardrail statistics.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
import os
import platform
import threading
import time
from typing import Any, Dict, List


def get_current_process_memory_mb() -> float:
    """Retrieve current process Resident Set Size (RSS) in Megabytes across platforms."""
    system = platform.system()
    if system == "Windows":
        try:
            class PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            fn = ctypes.windll.psapi.GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            fn.restype = wintypes.BOOL
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if fn(handle, ctypes.byref(pmc), pmc.cb):
                return round(pmc.WorkingSetSize / (1024 * 1024), 2)
        except Exception:
            pass
    elif system in ("Linux", "Darwin"):
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # Linux: ru_maxrss in kilobytes; macOS: ru_maxrss in bytes
            divisor = 1024.0 if system == "Linux" else 1024.0 * 1024.0
            return round(usage.ru_maxrss / divisor, 2)
        except Exception:
            pass
    return 0.0


@dataclass
class GenerationMetrics:
    """Telemetry payload for a single generation pipeline execution."""

    session_id: str
    query: str
    model_name: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prompt_build_latency_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    model_generation_latency_ms: float = 0.0
    guardrail_latency_ms: float = 0.0
    total_pipeline_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    cache_hit: bool = False
    memory_rss_mb: float = 0.0
    cpu_percent: float = 0.0
    model_load_time_ms: float = 0.0
    streaming_latency_ms: float = 0.0
    total_citations: int = 0
    verified_citations: int = 0
    stripped_citations: int = 0
    grounding_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def completion_tokens(self) -> int:
        """Alias for generated_tokens to adhere to standard LLM telemetry naming."""
        return self.generated_tokens

    @property
    def generation_latency_ms(self) -> float:
        """Alias for model_generation_latency_ms."""
        return self.model_generation_latency_ms


class GenerationMetricsCollector:
    """Thread-safe collector and statistical aggregator for generation metrics."""

    _instance: GenerationMetricsCollector | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._records: List[GenerationMetrics] = []
        self._data_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> GenerationMetricsCollector:
        """Get singleton GenerationMetricsCollector instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def record(self, metrics: GenerationMetrics) -> None:
        """Record a completed generation telemetry record."""
        with self._data_lock:
            self._records.append(metrics)

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate statistical summary across all recorded generation events."""
        with self._data_lock:
            if not self._records:
                return {
                    "total_requests": 0,
                    "cache_hit_ratio": 0.0,
                    "avg_total_latency_ms": 0.0,
                    "avg_ttft_ms": 0.0,
                    "avg_tokens_per_second": 0.0,
                    "avg_grounding_score": 0.0,
                    "total_tokens_generated": 0,
                    "avg_memory_rss_mb": 0.0,
                    "avg_cpu_percent": 0.0,
                    "avg_model_load_time_ms": 0.0,
                    "avg_generation_latency_ms": 0.0,
                }

            n = len(self._records)
            hits = sum(1 for r in self._records if r.cache_hit)
            total_gen_time = sum(r.model_generation_latency_ms for r in self._records if not r.cache_hit)
            gen_runs = sum(1 for r in self._records if not r.cache_hit)

            return {
                "total_requests": n,
                "cache_hit_ratio": round(hits / n, 4),
                "avg_prompt_build_ms": round(sum(r.prompt_build_latency_ms for r in self._records) / n, 2),
                "avg_ttft_ms": round(sum(r.time_to_first_token_ms for r in self._records) / n, 2),
                "avg_model_latency_ms": round(total_gen_time / max(1, gen_runs), 2),
                "avg_generation_latency_ms": round(total_gen_time / max(1, gen_runs), 2),
                "avg_total_latency_ms": round(sum(r.total_pipeline_latency_ms for r in self._records) / n, 2),
                "avg_tokens_per_second": round(
                    sum(r.tokens_per_second for r in self._records if not r.cache_hit) / max(1, gen_runs), 2
                ),
                "avg_grounding_score": round(sum(r.grounding_score for r in self._records) / n, 3),
                "total_tokens_generated": sum(r.generated_tokens for r in self._records),
                "avg_memory_rss_mb": round(sum(r.memory_rss_mb for r in self._records) / n, 2),
                "avg_cpu_percent": round(sum(r.cpu_percent for r in self._records) / n, 2),
                "avg_model_load_time_ms": round(sum(r.model_load_time_ms for r in self._records) / n, 2),
            }

    def clear(self) -> None:
        """Clear all stored telemetry history."""
        with self._data_lock:
            self._records.clear()
