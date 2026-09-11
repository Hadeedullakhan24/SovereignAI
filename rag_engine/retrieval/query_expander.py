"""Deterministic Query Expander.

Generates deterministic synonyms, related engineering keywords, and acronym variants
to maximize BM25 lexical recall in technical refinery documentation.
"""

from __future__ import annotations

from rag_engine.retrieval.query_normalizer import QueryNormalizer


class QueryExpander:
    """Domain-specific deterministic keyword and synonym expander."""

    # Engineering synonym mappings
    _SYNONYMS: dict[str, list[str]] = {
        "pump": ["centrifugal pump", "impeller", "casing", "suction", "discharge"],
        "valve": ["actuator", "seat", "bonnet", "stem", "disk", "closure"],
        "heat exchanger": ["cooler", "condenser", "reboiler", "tube bundle", "shell"],
        "compressor": ["rotor", "stator", "vane", "stage", "seal", "reciprocating"],
        "inspection": ["ndt", "ultrasonic testing", "visual inspection", "thickness", "wall loss"],
        "pressure": ["discharge pressure", "suction pressure", "differential pressure", "head"],
        "temperature": ["operating temperature", "skin temperature", "inlet", "outlet"],
        "maintenance": ["overhaul", "preventive maintenance", "work order", "downtime", "spares"],
        "vibration": ["amplitude", "frequency", "bearing vibration", "misalignment", "unbalance"],
        "leak": ["flange leak", "packing leak", "gasket failure", "emission"],
    }

    @classmethod
    def expand(cls, query: str, max_terms: int = 10) -> list[str]:
        """Generate expanded keyword tokens from query."""
        if not query:
            return []

        norm = QueryNormalizer.normalize(query).lower()
        expanded: list[str] = []

        # 1. Check acronym expansion
        for acr, full in QueryNormalizer.REFINERY_ACRONYMS.items():
            if acr.lower() in norm:
                expanded.append(full.lower())

        # 2. Check synonyms
        for key, syn_list in cls._SYNONYMS.items():
            if key in norm:
                for syn in syn_list:
                    if syn not in norm and syn not in expanded:
                        expanded.append(syn)
                        if len(expanded) >= max_terms:
                            break

        return expanded[:max_terms]

    @classmethod
    def create_expanded_query(cls, query: str) -> str:
        """Create an augmented query string combining original query and top expansion terms."""
        expansions = cls.expand(query)
        if not expansions:
            return query
        return f"{query} {' '.join(expansions)}"
