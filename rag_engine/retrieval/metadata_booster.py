"""Metadata Booster.

Applies domain-aware contextual boosting to candidate chunks matching critical
refinery parameters (equipment IDs, plant units, safety standards, page proximity).
"""

from __future__ import annotations

from typing import Optional

from rag_engine.retrieval.base_retriever import ScoredRetrievalChunk
from rag_engine.retrieval.query_analyzer import QueryIntent


class MetadataBooster:
    """Calculates and applies additive or multiplicative score boosts based on metadata matches."""

    def __init__(
        self,
        equipment_boost: float = 0.20,
        plant_unit_boost: float = 0.15,
        safety_standard_boost: float = 0.20,
        document_boost: float = 0.10,
        revision_boost: float = 0.05,
        page_proximity_boost: float = 0.10,
    ) -> None:
        self.equipment_boost = equipment_boost
        self.plant_unit_boost = plant_unit_boost
        self.safety_standard_boost = safety_standard_boost
        self.document_boost = document_boost
        self.revision_boost = revision_boost
        self.page_proximity_boost = page_proximity_boost

    def boost(
        self,
        candidates: list[ScoredRetrievalChunk],
        intent: QueryIntent,
        target_document_id: Optional[str] = None,
        target_page: Optional[int] = None,
    ) -> list[ScoredRetrievalChunk]:
        """Apply contextual boosts to candidate chunks and re-sort."""
        if not candidates:
            return []

        entities = intent.entities
        target_equips = set(entities.equipment_tags)
        target_units = set(entities.plant_units)
        target_standards = set(entities.standards)

        boosted: list[ScoredRetrievalChunk] = []

        for item in candidates:
            chunk = item.chunk
            meta = chunk.metadata
            chunk_equips = set(meta.equipment_entities or [])
            if meta.equipment_id:
                chunk_equips.add(meta.equipment_id)
            chunk_unit = (meta.plant_unit or "").upper()
            chunk_standards = set(meta.safety_entities or [])

            total_boost = 0.0
            reasons = []

            # Equipment match
            if target_equips and (target_equips & chunk_equips):
                total_boost += self.equipment_boost
                reasons.append(f"+{self.equipment_boost:.2f} equipment match {list(target_equips & chunk_equips)}")

            # Plant unit match
            if target_units and chunk_unit in target_units:
                total_boost += self.plant_unit_boost
                reasons.append(f"+{self.plant_unit_boost:.2f} plant unit match ({chunk_unit})")

            # Safety standard match
            if target_standards and (target_standards & chunk_standards):
                total_boost += self.safety_standard_boost
                reasons.append(f"+{self.safety_standard_boost:.2f} safety standard match")

            # Document ID match
            if target_document_id and meta.document_id == target_document_id:
                total_boost += self.document_boost
                reasons.append(f"+{self.document_boost:.2f} target document match")

            # Page proximity match
            if target_page is not None and meta.page_number is not None:
                diff = abs(meta.page_number - target_page)
                if diff == 0:
                    total_boost += self.page_proximity_boost
                    reasons.append(f"+{self.page_proximity_boost:.2f} exact page match")
                elif diff <= 2:
                    prox = self.page_proximity_boost * 0.5
                    total_boost += prox
                    reasons.append(f"+{prox:.2f} adjacent page ({meta.page_number})")

            # Update score
            new_score = item.score + total_boost
            explain = item.explainability
            if reasons:
                explain += " Boosts: " + "; ".join(reasons)

            updated = ScoredRetrievalChunk(
                chunk=chunk,
                score=new_score,
                rank=item.rank,
                dense_score=item.dense_score,
                bm25_score=item.bm25_score,
                fusion_score=item.fusion_score,
                rerank_score=item.rerank_score,
                boost_applied=item.boost_applied + total_boost,
                explainability=explain,
            )
            boosted.append(updated)

        # Re-sort descending by score
        boosted.sort(key=lambda x: x.score, reverse=True)
        for r, item in enumerate(boosted):
            item.rank = r

        return boosted
