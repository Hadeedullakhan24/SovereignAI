"""Metadata Filter Planner & Builder.

Synthesizes structured Qdrant MetadataFilter objects from natural language queries,
extracted domain entities, and explicit programmatic filter constraints.
"""

from __future__ import annotations

from typing import Any, Optional

from rag_engine.retrieval.query_analyzer import ExtractedEntities, IntentType, QueryIntent
from rag_engine.schemas.vector_store import (
    FieldFilter,
    FilterOperator,
    MetadataFilter,
)


class MetadataFilterBuilder:
    """Fluent builder for programmatic MetadataFilter construction."""

    def __init__(self) -> None:
        self._must: list[FieldFilter] = []
        self._should: list[FieldFilter] = []
        self._must_not: list[FieldFilter] = []

    def equals(self, field: str, value: Any) -> MetadataFilterBuilder:
        self._must.append(FieldFilter(field=field, operator=FilterOperator.EQUALS, value=value))
        return self

    def contains(self, field: str, value: Any) -> MetadataFilterBuilder:
        self._must.append(FieldFilter(field=field, operator=FilterOperator.CONTAINS, value=value))
        return self

    def in_list(self, field: str, values: list[Any]) -> MetadataFilterBuilder:
        self._must.append(FieldFilter(field=field, operator=FilterOperator.IN, value=values))
        return self

    def page_number(self, page: int) -> MetadataFilterBuilder:
        return self.equals("page_number", page)

    def document_id(self, doc_id: str) -> MetadataFilterBuilder:
        return self.equals("document_id", doc_id)

    def equipment(self, tag: str) -> MetadataFilterBuilder:
        return self.contains("equipment_entities", tag)

    def plant_unit(self, unit: str) -> MetadataFilterBuilder:
        return self.equals("plant_unit", unit)

    def category(self, cat: str) -> MetadataFilterBuilder:
        return self.equals("category", cat)

    def build(self) -> Optional[MetadataFilter]:
        if not self._must and not self._should and not self._must_not:
            return None
        return MetadataFilter(
            must=self._must,
            should=self._should,
            must_not=self._must_not,
        )


class MetadataFilterPlanner:
    """Plans and derives Qdrant payload filters from QueryIntent and query text."""

    @classmethod
    def plan_filters(
        cls,
        intent: QueryIntent,
        explicit_filters: Optional[dict[str, Any]] = None,
    ) -> Optional[MetadataFilter]:
        """Synthesize MetadataFilter combining inferred intent filters and explicit constraints."""
        builder = MetadataFilterBuilder()

        # 1. Apply explicit filters first if provided
        if explicit_filters:
            for k, v in explicit_filters.items():
                if v is None:
                    continue
                if isinstance(v, list):
                    builder.in_list(k, v)
                else:
                    builder.equals(k, v)

        # 2. Add inferred filters from entity extraction
        entities = intent.entities

        # If equipment tags are present
        if entities.equipment_tags:
            # Match chunks containing any of the detected equipment tags
            if len(entities.equipment_tags) == 1:
                builder.contains("equipment_entities", entities.equipment_tags[0])
            else:
                # We can add a should filter for multiple tags
                for tag in entities.equipment_tags:
                    builder._should.append(
                        FieldFilter(
                            field="equipment_entities",
                            operator=FilterOperator.CONTAINS,
                            value=tag,
                        )
                    )

        # If plant unit is detected
        if entities.plant_units:
            builder.equals("plant_unit", entities.plant_units[0])

        # Note: Do not impose hard category 'must' filters from intent alone;
        # allow open multi-channel semantic and lexical retrieval across all corpus categories.


        # Page reference filter
        if entities.page_references:
            try:
                pg = int(entities.page_references[0])
                builder.page_number(pg)
            except ValueError:
                pass

        # Revision filter
        if entities.revision_numbers:
            builder.equals("revision", entities.revision_numbers[0])

        return builder.build()
