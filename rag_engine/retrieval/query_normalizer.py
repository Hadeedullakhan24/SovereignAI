"""Query Normalizer.

Performs deterministic query normalization: NFKC unicode normalization,
equipment tag standard hyphenation (P203 -> P-203), engineering unit spacing
(150bar -> 150 bar), and noise removal.
"""

from __future__ import annotations

import re
import unicodedata

from rag_engine.retrieval.retrieval_utils import normalize_whitespace


class QueryNormalizer:
    """Deterministic normalizer for technical queries in petroleum refining."""

    # Common unhyphenated equipment tags: P203 -> P-203, MOV101 -> MOV-101, HX01 -> HX-01
    _EQUIP_TAG_NORM = re.compile(
        r"\b(P|MOV|HX|V|C|E|K|T|TK|PU|B|R)([0-9]{2,4}[A-Z]?)\b", re.IGNORECASE
    )

    # Unit spacing: 150bar -> 150 bar, 250C -> 250 °C, 10kg/cm2 -> 10 kg/cm2
    _UNIT_SPACING_BAR = re.compile(r"([0-9]+(?:\.[0-9]+)?)(bar|psi|mpa|kpa)\b", re.IGNORECASE)
    _UNIT_SPACING_TEMP = re.compile(r"([0-9]+(?:\.[0-9]+)?)(degc|c|degf|f)\b", re.IGNORECASE)
    _UNIT_SPACING_KG = re.compile(r"([0-9]+(?:\.[0-9]+)?)(kg/cm2|kg/cm²)\b", re.IGNORECASE)
    _UNIT_SPACING_RPM = re.compile(r"([0-9]+(?:\.[0-9]+)?)(rpm|kw|mw)\b", re.IGNORECASE)

    # Noise / filler prefix patterns: e.g. "please tell me", "can you show", "what is", "give me"
    _NOISE_PREFIXES = re.compile(
        r"^(?:please\s+|can\s+you\s+(?:please\s+)?(?:tell|show|give|find)\s+(?:me\s+)?|what\s+is\s+|tell\s+me\s+about\s+|give\s+me\s+|show\s+me\s+|where\s+can\s+i\s+find\s+)",
        re.IGNORECASE,
    )

    # Common refinery acronym expansions dictionary
    REFINERY_ACRONYMS: dict[str, str] = {
        "MOV": "Motor Operated Valve",
        "PRV": "Pressure Relief Valve",
        "PSV": "Pressure Safety Valve",
        "CDU": "Crude Distillation Unit",
        "VDU": "Vacuum Distillation Unit",
        "DCU": "Delayed Coker Unit",
        "NHT": "Naphtha Hydrotreater",
        "CCR": "Continuous Catalytic Reformer",
        "FCCU": "Fluidized Catalytic Cracking Unit",
        "DHDS": "Diesel Hydrodesulfurization",
        "HGU": "Hydrogen Generation Unit",
        "SRU": "Sulfur Recovery Unit",
        "SOP": "Standard Operating Procedure",
        "NDT": "Non-Destructive Testing",
        "UT": "Ultrasonic Testing",
        "MPT": "Magnetic Particle Testing",
        "DPT": "Dye Penetrant Testing",
        "P&ID": "Piping and Instrumentation Diagram",
        "PID": "Piping and Instrumentation Diagram",
        "PFD": "Process Flow Diagram",
        "ESD": "Emergency Shutdown",
        "DCS": "Distributed Control System",
        "PLC": "Programmable Logic Controller",
        "OISD": "Oil Industry Safety Directorate",
        "PNGRB": "Petroleum and Natural Gas Regulatory Board",
    }

    @classmethod
    def normalize(cls, query: str, strip_noise: bool = False) -> str:
        """Apply sequential normalization to query string."""
        if not query:
            return ""

        # 1. Unicode NFKC normalization
        text = unicodedata.normalize("NFKC", query)

        # 2. Equipment tag standard hyphenation (e.g. P203 -> P-203, MOV101 -> MOV-101)
        text = cls._EQUIP_TAG_NORM.sub(r"\1-\2", text)

        # 3. Unit spacing (e.g. 150bar -> 150 bar, 250C -> 250 °C)
        text = cls._UNIT_SPACING_BAR.sub(r"\1 \2", text)
        text = cls._UNIT_SPACING_KG.sub(r"\1 \2", text)
        text = cls._UNIT_SPACING_RPM.sub(r"\1 \2", text)
        text = cls._UNIT_SPACING_TEMP.sub(r"\1 °\2", text)
        # Normalize double degree symbol if any
        text = text.replace("°°", "°")

        # 4. Strip noise filler prefix if requested
        if strip_noise:
            text = cls._NOISE_PREFIXES.sub("", text)

        # 5. Whitespace collapse
        return normalize_whitespace(text)

    @classmethod
    def expand_abbreviations(cls, query: str) -> str:
        """Expand known refinery acronyms in parentheses alongside original terms."""
        normalized = cls.normalize(query)
        words = normalized.split()
        expanded_words = []
        for word in words:
            clean_w = word.strip(".,;:?!()[]").upper()
            if clean_w in cls.REFINERY_ACRONYMS:
                expanded_words.append(f"{word} ({cls.REFINERY_ACRONYMS[clean_w]})")
            else:
                expanded_words.append(word)
        return " ".join(expanded_words)
