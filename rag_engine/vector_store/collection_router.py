"""Multi-Collection Router.

Deterministically routes chunks to dedicated, specialized collections based on
document category, metadata attributes, drawing types, and procedure markers.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from rag_engine.schemas.embedding import EmbeddedChunk

logger = logging.getLogger(__name__)


class RoutingRule:
    """A deterministic predicate rule mapping matching chunks to a target collection."""

    def __init__(
        self,
        name: str,
        target_collection: str,
        predicate: Callable[[EmbeddedChunk], bool],
        priority: int = 100,
    ) -> None:
        self.name = name
        self.target_collection = target_collection
        self.predicate = predicate
        self.priority = priority


class CollectionRouter:
    """Evaluates chunk metadata and routes chunks to the appropriate collection."""

    def __init__(self) -> None:
        self._rules: list[RoutingRule] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register deterministic rules for standard MRPL refinery document domains."""
        # 1. P&ID Diagrams (High Priority)
        def is_pid(chunk: EmbeddedChunk) -> bool:
            meta = chunk.metadata
            cat = str(getattr(meta, "category", "")).lower()
            subcat = str(getattr(meta, "subcategory", "")).lower()
            src = str(getattr(meta, "source_file", "")).lower()
            if "p&id" in cat or "pid" in cat or "p&id" in subcat or "pid" in subcat:
                return True
            if "p&id" in src or "_pid" in src or "-pid" in src:
                return True
            extra = getattr(meta, "extra", {}) or {}
            if extra.get("is_pid") or extra.get("drawing_type") == "P&ID":
                return True
            return False

        self.register_rule("pid_rule", "mrpl_pids_v1", is_pid, priority=10)

        # 2. Engineering Drawings
        def is_drawing(chunk: EmbeddedChunk) -> bool:
            meta = chunk.metadata
            cat = str(getattr(meta, "category", "")).lower()
            subcat = str(getattr(meta, "subcategory", "")).lower()
            src = str(getattr(meta, "source_file", "")).lower()
            if "drawing" in cat or "drawing" in subcat:
                return True
            if src.endswith((".dwg", ".dxf", ".svg")):
                return True
            extra = getattr(meta, "extra", {}) or {}
            return bool(extra.get("drawing_metadata") or extra.get("drawing_number"))

        self.register_rule("drawing_rule", "mrpl_drawings_v1", is_drawing, priority=20)

        # 3. Safety Standards & OISD
        def is_safety(chunk: EmbeddedChunk) -> bool:
            meta = chunk.metadata
            cat = str(getattr(meta, "category", "")).lower()
            safety_tags = getattr(meta, "safety_entities", None)
            if "safety" in cat or "oisd" in cat or "hazard" in cat:
                return True
            return bool(safety_tags and len(safety_tags) > 0)

        self.register_rule("safety_rule", "mrpl_safety_v1", is_safety, priority=30)

        # 4. Inspection Reports
        def is_inspection(chunk: EmbeddedChunk) -> bool:
            meta = chunk.metadata
            cat = str(getattr(meta, "category", "")).lower()
            subcat = str(getattr(meta, "subcategory", "")).lower()
            return "inspect" in cat or "ndt" in cat or "inspect" in subcat or "ndt" in subcat

        self.register_rule("inspection_rule", "mrpl_inspection_v1", is_inspection, priority=40)

        # 5. Maintenance Documents
        def is_maintenance(chunk: EmbeddedChunk) -> bool:
            meta = chunk.metadata
            cat = str(getattr(meta, "category", "")).lower()
            subcat = str(getattr(meta, "subcategory", "")).lower()
            return "maint" in cat or "overhaul" in cat or "work_order" in cat or "maint" in subcat

        self.register_rule("maintenance_rule", "mrpl_maintenance_v1", is_maintenance, priority=50)

        # 6. Standard Operating Procedures (SOPs)
        def is_sop(chunk: EmbeddedChunk) -> bool:
            meta = chunk.metadata
            cat = str(getattr(meta, "category", "")).lower()
            subcat = str(getattr(meta, "subcategory", "")).lower()
            is_list = getattr(meta, "is_list_chunk", False)
            if "sop" in cat or "procedure" in cat or "sop" in subcat:
                return True
            return False

        self.register_rule("sop_rule", "mrpl_sops_v1", is_sop, priority=60)

        # 7. Technical Emails & Correspondence
        def is_email(chunk: EmbeddedChunk) -> bool:
            meta = chunk.metadata
            cat = str(getattr(meta, "category", "")).lower()
            src = str(getattr(meta, "source_file", "")).lower()
            return "email" in cat or src.endswith((".eml", ".msg"))

        self.register_rule("email_rule", "mrpl_emails_v1", is_email, priority=70)

        # 8. Engineering Manuals
        def is_manual(chunk: EmbeddedChunk) -> bool:
            meta = chunk.metadata
            cat = str(getattr(meta, "category", "")).lower()
            return "manual" in cat or "handbook" in cat or "guide" in cat

        self.register_rule("manual_rule", "mrpl_manuals_v1", is_manual, priority=80)

    def register_rule(
        self,
        name: str,
        target_collection: str,
        predicate: Callable[[EmbeddedChunk], bool],
        priority: int = 100,
    ) -> None:
        """Register a custom routing rule."""
        rule = RoutingRule(
            name=name,
            target_collection=target_collection,
            predicate=predicate,
            priority=priority,
        )
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)

    def get_target_collection(self, chunk: EmbeddedChunk) -> str:
        """Determine target collection for a single chunk using rule precedence."""
        for rule in self._rules:
            try:
                if rule.predicate(chunk):
                    return rule.target_collection
            except Exception as e:
                logger.warning("Error evaluating rule '%s' on chunk '%s': %s", rule.name, chunk.chunk_id, e)

        # Deterministic fallback
        return "mrpl_general_v1"

    def route_chunk(self, chunk: EmbeddedChunk) -> str:
        """Alias for get_target_collection."""
        return self.get_target_collection(chunk)

    def route_chunks(self, chunks: list[EmbeddedChunk]) -> dict[str, list[EmbeddedChunk]]:
        """Group a batch of EmbeddedChunk objects by their target collections."""
        grouped: dict[str, list[EmbeddedChunk]] = {}
        for chunk in chunks:
            target = self.get_target_collection(chunk)
            if target not in grouped:
                grouped[target] = []
            grouped[target].append(chunk)
        return grouped
