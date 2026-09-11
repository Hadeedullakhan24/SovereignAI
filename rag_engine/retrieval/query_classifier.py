"""Query Intent Classifier.

Provides fine-grained classification of user queries into operational categories:
equipment lookup, safety query, inspection report, maintenance history, procedure lookup,
or general semantic questions.
"""

from __future__ import annotations

from typing import Optional

from rag_engine.retrieval.query_analyzer import (
    ExtractedEntities,
    IntentType,
    QueryAnalyzer,
    QueryIntent,
)


class QueryClassifier:
    """Classifies user queries into refinery operational archetypes."""

    def __init__(self, analyzer: Optional[QueryAnalyzer] = None) -> None:
        self.analyzer = analyzer or QueryAnalyzer()

    def classify(self, query: str) -> QueryIntent:
        """Analyze and classify query into intent categories."""
        return self.analyzer.analyze(query)

    def get_intent_category(self, query: str) -> str:
        """Convenience method returning string identifier of primary intent category."""
        intent = self.classify(query)
        return intent.primary_intent.value
