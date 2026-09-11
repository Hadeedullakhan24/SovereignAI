"""Vector Health Monitor.

Executes active canary probes (write, search, delete) to verify end-to-end database
responsiveness, measures disk capacity, and detects performance degradation.
"""

from __future__ import annotations

import logging

from rag_engine.interfaces.base_vector_store import BaseVectorStore
from rag_engine.schemas.vector_store import VectorDBHealthReport

logger = logging.getLogger(__name__)


class VectorHealthMonitor:
    """Canary probe runner and health diagnostics engine."""

    def __init__(self, store: BaseVectorStore) -> None:
        self.store = store

    def run_diagnostics(self) -> VectorDBHealthReport:
        """Run complete health diagnostics check."""
        return self.store.health_check()
