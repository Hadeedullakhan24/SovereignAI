"""Deterministic extraction utilities for refinery documents.

Contains rule-based tokenizers, regex matchers, table parsers, entity extractors,
relationship graph builders, and citation coordinate generators.
NO AI / NO LLM / NO EMBEDDINGS.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, List, Optional, Tuple

from rag_engine.parsers.profiles.base_profile import RefineryProfile
from rag_engine.schemas.parsed_document import (
    CitationCoordinates,
    CrossReference,
    EntityGraph,
    EntityRelationship,
    EquipmentEntity,
    EquipmentType,
    RelationType,
    SafetyWarning,
    SafetyWarningSeverity,
    Section,
    Table,
    TableCell,
)


class ParserUtils:
    """Universal deterministic text analysis and entity extraction utility."""

    # Common regex patterns
    HEADING_NUMERIC = re.compile(r"^(\d+(?:\.\d+)*)\s+(.*)$")
    HEADING_MD = re.compile(r"^(#{1,6})\s+(.*)$")
    BULLET_PATTERN = re.compile(r"^\s*[-*•–—]\s+(.*)$")
    NUMBERED_LIST_PATTERN = re.compile(r"^\s*(\d+[.)])\s+(.*)$")

    DATE_PATTERN = re.compile(
        r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    )
    REVISION_PATTERN = re.compile(r"\b(?:Rev|Revision)[\s.:#-]*([A-Z0-9]+)\b", re.IGNORECASE)
    ENGINEER_PATTERN = re.compile(
        r"\b(?:Engineer|Inspected by|Tested by|Prepared by|Approved by|Auditor)[\s.:-]+([A-Z][a-zA-Z\s.]+?)(?=\n|,|;|$)",
        re.IGNORECASE,
    )

    CROSS_REF_PATTERN = re.compile(
        r"\b(Fig(?:ure)?\.?\s*\d+(?:\.\d+)*|Table\s*\d+(?:\.\d+)*|Section\s*\d+(?:\.\d+)*|"
        r"OISD(?:-STD)?-\d+|API-\d+|ASME\s+[A-Z0-9.]+)\b",
        re.IGNORECASE,
    )

    PRESSURE_PATTERN = re.compile(
        r"(\d+(?:\.\d+)?)\s*(bar[g]?|bar\(g\)|psi[g]?|kPa|MPa|kg/cm[2²]|atm)\b",
        re.IGNORECASE,
    )
    TEMPERATURE_PATTERN = re.compile(
        r"(-?\d+(?:\.\d+)?)\s*(°\s*C|deg\s*C|degC|\bC\b|°\s*F|deg\s*F|degF|\bF\b|\bK\b)",
        re.IGNORECASE,
    )
    LOOP_PATTERN = re.compile(r"\b([A-Z]{2,4}-\d{2,4}[A-Z]?)\b")
    PIPE_PATTERN = re.compile(r"\b(LINE-\d{2,4}-[A-Z0-9-]+|PIPE-\d{2,4}-[A-Z0-9-]+)\b")

    # Relationship keywords in sentence context
    RELATION_MAP = [
        (re.compile(r"\b(?:discharges\s+to|feeds|pumps\s+to|supplies|delivers\s+to)\b", re.I), RelationType.FEEDS),
        (re.compile(r"\b(?:connected\s+to|linked\s+to|piped\s+to|joined\s+to)\b", re.I), RelationType.CONNECTED_TO),
        (re.compile(r"\b(?:regulates|throttles|controls\s+flow\s+to)\b", re.I), RelationType.REGULATES),
        (re.compile(r"\b(?:monitors|measures|senses|detects)\b", re.I), RelationType.MONITORS),
        (re.compile(r"\b(?:powers|drives|actuates)\b", re.I), RelationType.POWERS),
        (re.compile(r"\b(?:bypasses|diverts\s+around)\b", re.I), RelationType.BYPASSES),
        (re.compile(r"\b(?:isolates|blocks|shuts\s+off)\b", re.I), RelationType.ISOLATES),
    ]

    @staticmethod
    def normalize_text(text: str) -> str:
        """Sanitize whitespace, standardize quotes, and remove null/control bytes."""
        if not text:
            return ""
        # Remove null and non-printable control chars except \n \r \t
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Standardize quotes and dashes
        sanitized = sanitized.replace("“", '"').replace("”", '"')
        sanitized = sanitized.replace("‘", "'").replace("’", "'")
        sanitized = sanitized.replace("–", "-").replace("—", "-")
        # Standardize linebreaks
        sanitized = sanitized.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse excessive consecutive blank lines
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()

    @classmethod
    def extract_sections(cls, text: str, default_page: Optional[int] = 1) -> list[Section]:
        """Extract hierarchical sections based on headings and paragraph layout."""
        normalized = cls.normalize_text(text)
        if not normalized:
            return []

        lines = normalized.split("\n")
        sections: list[Section] = []
        current_section: Optional[Section] = None
        current_paragraphs: list[str] = []
        current_bullets: list[str] = []
        current_numbered: list[str] = []
        char_offset = 0

        def flush_current() -> None:
            nonlocal current_section, current_paragraphs, current_bullets, current_numbered
            if current_section is not None:
                body = "\n".join(current_paragraphs).strip()
                current_section.paragraphs = list(current_paragraphs)
                current_section.bullet_points = list(current_bullets)
                current_section.numbered_items = list(current_numbered)
                current_section.content = body
                current_section.normalized_text = body
                sections.append(current_section)
                current_paragraphs = []
                current_bullets = []
                current_numbered = []

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            # Check markdown heading
            md_match = cls.HEADING_MD.match(trimmed)
            # Check numeric heading (e.g. 1.1 Introduction)
            num_match = cls.HEADING_NUMERIC.match(trimmed)

            is_heading = False
            heading_title = ""
            heading_level = 1

            if md_match:
                is_heading = True
                heading_level = len(md_match.group(1))
                heading_title = md_match.group(2).strip()
            elif num_match and len(trimmed) < 100 and not trimmed.endswith("."):
                is_heading = True
                depth = num_match.group(1).count(".") + 1
                heading_level = min(depth, 6)
                heading_title = trimmed
            elif (
                trimmed.isupper()
                and 3 < len(trimmed) < 80
                and not trimmed.endswith(".")
                and sum(1 for c in trimmed if c.isalpha()) >= 3
                and not re.search(r"[#@\$%\^&*~+=<>{}\[\]|\\]", trimmed)
            ):
                is_heading = True
                heading_level = 1
                heading_title = trimmed

            if is_heading:
                flush_current()
                sec_id = f"sec_{uuid.uuid4().hex[:8]}"
                current_section = Section(
                    section_id=sec_id,
                    title=heading_title,
                    level=heading_level,
                    raw_text=line,
                    normalized_text="",
                    page_number=default_page,
                    citation_coords=CitationCoordinates(
                        page_number=default_page,
                        section_title=heading_title,
                        char_offset_start=char_offset,
                        citation_id=f"cit_{sec_id}",
                    ),
                    confidence=0.95,
                )
            else:
                # Check for bullet points
                bullet_match = cls.BULLET_PATTERN.match(trimmed)
                num_list_match = cls.NUMBERED_LIST_PATTERN.match(trimmed)

                if bullet_match:
                    current_bullets.append(bullet_match.group(1).strip())
                elif num_list_match:
                    current_numbered.append(num_list_match.group(2).strip())
                else:
                    current_paragraphs.append(trimmed)

            char_offset += len(line) + 1

        flush_current()

        # If no sections were identified, wrap all text into a default Root section
        if not sections:
            sec_id = f"sec_{uuid.uuid4().hex[:8]}"
            body = "\n".join(lines).strip()
            sections.append(
                Section(
                    section_id=sec_id,
                    title="General Document Content",
                    level=1,
                    content=body,
                    raw_text=text,
                    normalized_text=body,
                    paragraphs=[p.strip() for p in lines if p.strip()],
                    page_number=default_page,
                    citation_coords=CitationCoordinates(
                        page_number=default_page,
                        section_title="General Document Content",
                        char_offset_start=0,
                        char_offset_end=len(body),
                        citation_id=f"cit_{sec_id}",
                    ),
                    confidence=0.90,
                )
            )

        return sections

    @classmethod
    def parse_tables_from_text(cls, text: str, page_number: Optional[int] = 1) -> list[Table]:
        """Parse pipe-delimited markdown or ASCII tables from text."""
        tables: list[Table] = []
        lines = text.split("\n")
        table_lines: list[str] = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            # Detect pipe-delimited table line
            if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
                table_lines.append(stripped)
                in_table = True
            elif in_table:
                # End of current table
                tbl = cls._convert_lines_to_table(table_lines, page_number)
                if tbl:
                    tables.append(tbl)
                table_lines = []
                in_table = False

        if in_table and table_lines:
            tbl = cls._convert_lines_to_table(table_lines, page_number)
            if tbl:
                tables.append(tbl)

        return tables

    @staticmethod
    def _convert_lines_to_table(lines: list[str], page_number: Optional[int]) -> Optional[Table]:
        """Convert consecutive pipe-delimited table lines into a Table object."""
        if not lines or len(lines) < 2:
            return None

        parsed_rows: list[list[str]] = []
        for line in lines:
            cells = [c.strip() for c in line.strip("|").split("|")]
            # Filter markdown header separator row (e.g. |---|---|)
            if all(re.match(r"^:?-+:?$", c) for c in cells if c):
                continue
            parsed_rows.append(cells)

        if not parsed_rows:
            return None

        headers = parsed_rows[0]
        data_rows = parsed_rows[1:] if len(parsed_rows) > 1 else []
        col_count = len(headers)

        # Standardize data row widths
        normalized_data_rows: list[list[str]] = []
        for row in data_rows:
            if len(row) < col_count:
                row = row + [""] * (col_count - len(row))
            elif len(row) > col_count:
                row = row[:col_count]
            normalized_data_rows.append(row)

        table_id = f"tbl_{uuid.uuid4().hex[:8]}"
        cells: list[TableCell] = []

        # Headers
        for c_idx, h_val in enumerate(headers):
            cells.append(
                TableCell(row_idx=0, col_idx=c_idx, value=h_val, raw_value=h_val, is_header=True)
            )

        # Data cells
        for r_idx, row in enumerate(normalized_data_rows):
            for c_idx, val in enumerate(row):
                cells.append(
                    TableCell(row_idx=r_idx + 1, col_idx=c_idx, value=val, raw_value=val, is_header=False)
                )

        dataframe_dict = {
            headers[c]: [row[c] for row in normalized_data_rows]
            for c in range(col_count)
        }

        return Table(
            table_id=table_id,
            caption=f"Table ({col_count} columns x {len(normalized_data_rows)} rows)",
            headers=headers,
            rows=normalized_data_rows,
            row_count=len(normalized_data_rows),
            col_count=col_count,
            page_number=page_number,
            dataframe_dict=dataframe_dict,
            raw_text="\n".join(lines),
            normalized_text="\n".join(lines),
            cells=cells,
            confidence=0.98,
        )

    @classmethod
    def extract_equipment(
        cls,
        text: str,
        profile: RefineryProfile,
        page_number: Optional[int] = 1,
        section_title: Optional[str] = None,
    ) -> list[EquipmentEntity]:
        """Extract equipment tags, loop numbers, and operational limits."""
        entities: list[EquipmentEntity] = []
        seen_tags: set[str] = set()

        # Find tags matching profile patterns
        for pattern_str in profile.equipment_tag_patterns:
            for match in re.finditer(pattern_str, text):
                tag = match.group().strip().rstrip(".,;:)")
                tag_upper = tag.upper()
                if tag_upper in seen_tags or len(tag) < 2:
                    continue
                seen_tags.add(tag_upper)

                start_idx = match.start()
                # Grab context window around the match (up to 120 chars)
                ctx_start = max(0, start_idx - 60)
                ctx_end = min(len(text), start_idx + len(tag) + 60)
                context_window = text[ctx_start:ctx_end]

                eq_type = profile.resolve_equipment_type(tag, context_window)

                # Look for operating pressure in context
                p_match = cls.PRESSURE_PATTERN.search(context_window)
                op_pressure = f"{p_match.group(1)} {p_match.group(2)}" if p_match else None

                # Look for operating temperature in context
                t_match = cls.TEMPERATURE_PATTERN.search(context_window)
                op_temp = f"{t_match.group(1)} {t_match.group(2)}" if t_match else None

                units = {}
                if op_pressure and p_match:
                    units["pressure"] = p_match.group(2)
                if op_temp and t_match:
                    units["temperature"] = t_match.group(2)

                # Loop and pipe checks
                loop_match = cls.LOOP_PATTERN.search(context_window)
                loop_no = loop_match.group(1) if loop_match else None

                pipe_match = cls.PIPE_PATTERN.search(context_window)
                pipe_no = pipe_match.group(1) if pipe_match else None

                entity_id = f"eq_{uuid.uuid4().hex[:8]}"
                entities.append(
                    EquipmentEntity(
                        entity_id=entity_id,
                        tag=tag_upper,
                        name=f"{eq_type.value.replace('_', ' ').title()} {tag_upper}",
                        equipment_type=eq_type,
                        operating_pressure=op_pressure,
                        operating_temperature=op_temp,
                        units=units,
                        loop_number=loop_no,
                        pipe_number=pipe_no,
                        location_section=section_title,
                        page_number=page_number,
                        raw_text=tag,
                        normalized_text=tag_upper,
                        char_offset=start_idx,
                        citation_id=f"cit_{entity_id}",
                        confidence=0.92,
                    )
                )

        return entities

    @classmethod
    def build_entity_graph(
        cls,
        text: str,
        entities: list[EquipmentEntity],
    ) -> EntityGraph:
        """Construct directed relationship graph between equipment entities based on text proximity."""
        graph = EntityGraph()
        for e in entities:
            graph.add_node(e)

        if len(entities) < 2:
            return graph

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        tag_set = {e.tag for e in entities}

        for sentence in sentences:
            # Find all tags mentioned in sentence and sort by appearance order
            sent_upper = sentence.upper()
            tag_positions = []
            for t in tag_set:
                pos = sent_upper.find(t)
                if pos != -1:
                    tag_positions.append((pos, t))
            tag_positions.sort(key=lambda x: x[0])
            found_tags = [t for _, t in tag_positions]

            if len(found_tags) >= 2:
                # Check for relationship verbs
                matched_relation = RelationType.CONNECTED_TO
                for pattern, rel_type in cls.RELATION_MAP:
                    if pattern.search(sentence):
                        matched_relation = rel_type
                        break

                # Create pairwise edges for tags in the same sentence
                for i in range(len(found_tags) - 1):
                    src = found_tags[i]
                    tgt = found_tags[i + 1]
                    if src != tgt:

                        graph.add_edge(
                            EntityRelationship(
                                source_tag=src,
                                target_tag=tgt,
                                relation_type=matched_relation,
                                context_sentence=sentence.strip()[:200],
                                confidence=0.85,
                            )
                        )

        return graph

    @classmethod
    def extract_safety_warnings(
        cls,
        text: str,
        profile: RefineryProfile,
        page_number: Optional[int] = 1,
        section_title: Optional[str] = None,
    ) -> list[SafetyWarning]:
        """Extract DANGER, WARNING, CAUTION notices and standard references."""
        warnings: list[SafetyWarning] = []
        lines = text.split("\n")

        for line in lines:
            line_upper = line.upper().strip()
            for kw, sev_str in profile.safety_keywords.items():
                if kw in line_upper and len(line.strip()) > 5:
                    # Determine severity
                    sev = SafetyWarningSeverity(sev_str)
                    # Extract any standards cited in this line
                    standards_found = [
                        std for std in profile.standard_names if std.upper() in line_upper
                    ]

                    warn_id = f"warn_{uuid.uuid4().hex[:8]}"
                    warnings.append(
                        SafetyWarning(
                            warning_id=warn_id,
                            severity=sev,
                            text=line.strip(),
                            raw_text=line,
                            normalized_text=cls.normalize_text(line),
                            standards=standards_found,
                            page_number=page_number,
                            section=section_title,
                            citation_id=f"cit_{warn_id}",
                            confidence=0.95,
                        )
                    )
                    break

        return warnings

    @classmethod
    def extract_cross_references(cls, text: str, page_number: Optional[int] = 1) -> list[CrossReference]:
        """Extract figure, table, section, and standard pointers."""
        refs: list[CrossReference] = []
        for match in cls.CROSS_REF_PATTERN.finditer(text):
            val = match.group(1).strip()
            val_upper = val.upper()

            if "FIG" in val_upper:
                ref_type = "FIGURE"
            elif "TABLE" in val_upper:
                ref_type = "TABLE"
            elif "SECTION" in val_upper:
                ref_type = "SECTION"
            else:
                ref_type = "STANDARD"

            refs.append(
                CrossReference(
                    ref_type=ref_type,
                    target=val,
                    raw_text=val,
                    page_number=page_number,
                    citation_id=f"cit_ref_{uuid.uuid4().hex[:6]}",
                    confidence=0.90,
                )
            )
        return refs

    @classmethod
    def extract_operational_metadata(
        cls,
        text: str,
        profile: RefineryProfile,
    ) -> dict[str, list[str]]:
        """Extract dates, revisions, engineer names, and referenced standards."""
        dates = list(set(cls.DATE_PATTERN.findall(text)))
        revisions = list(set(cls.REVISION_PATTERN.findall(text)))
        engineers = [e.strip() for e in set(cls.ENGINEER_PATTERN.findall(text)) if len(e.strip()) > 2]
        standards = [std for std in profile.standard_names if re.search(r"\b" + re.escape(std) + r"\b", text, re.I)]

        return {
            "dates": sorted(dates),
            "revisions": sorted(revisions),
            "engineers": sorted(engineers),
            "standards": sorted(list(set(standards))),
        }
