"""Streaming Manager — Unified Streaming and Telemetry Tracking.

Wraps token stream generators, measures Time-To-First-Token (TTFT), accumulates
complete responses, and calculates dynamic token generation throughput.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Generator, Iterator


@dataclass
class StreamingSessionMetrics:
    """Telemetry data captured during token streaming."""

    ttft_ms: float = 0.0
    total_stream_time_ms: float = 0.0
    tokens_streamed: int = 0
    tokens_per_second: float = 0.0
    accumulated_text: str = ""


class StreamingManager:
    """Manages token streaming iteration and performance telemetry."""

    def wrap_stream(
        self,
        token_iterator: Iterator[str],
    ) -> Generator[str, None, StreamingSessionMetrics]:
        """Wrap token stream to track TTFT, throughput, and accumulated response.
        
        Args:
            token_iterator: Upstream model token generator.
            
        Yields:
            Token strings as received.
            
        Returns:
            StreamingSessionMetrics upon stream exhaustion.
        """
        start_time = time.perf_counter()
        first_token_time: float | None = None
        accumulated: list[str] = []
        token_count = 0

        for token in token_iterator:
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now

            token_count += 1
            accumulated.append(token)
            yield token

        end_time = time.perf_counter()

        ttft_ms = ((first_token_time or end_time) - start_time) * 1000.0
        total_time_ms = (end_time - start_time) * 1000.0
        total_time_s = end_time - start_time
        tok_per_sec = token_count / total_time_s if total_time_s > 0 else 0.0

        metrics = StreamingSessionMetrics(
            ttft_ms=round(ttft_ms, 2),
            total_stream_time_ms=round(total_time_ms, 2),
            tokens_streamed=token_count,
            tokens_per_second=round(tok_per_sec, 2),
            accumulated_text="".join(accumulated),
        )
        return metrics
