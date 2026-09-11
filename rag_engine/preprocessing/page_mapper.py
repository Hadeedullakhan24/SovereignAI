"""Page number preservation and citation coordinate mapping engine."""

from __future__ import annotations

from typing import Optional
from rag_engine.schemas.parsed_document import (
    CitationCoordinates,
    PageMapEntry,
    ParsedDocument,
    Section,
    Table,
)


class PageMapper:
    """Stage 8: Preserves page numbers and generates citation coordinate maps.

    Ensures downstream Chunking and Retrieval engines never lose source citation traceability.
    """

    @classmethod
    def build_page_map(
        cls,
        sections: list[Section],
        tables: list[Table],
        total_pages: Optional[int] = None,
    ) -> dict[int, PageMapEntry]:
        """Construct dictionary mapping page_number to PageMapEntry with character offsets."""
        page_map: dict[int, PageMapEntry] = {}

        # 1. Track cumulative character offsets across sections
        current_offset = 0
        section_offsets: dict[str, tuple[int, int, int]] = {}  # sec_id -> (page, start, end)

        for sec in sections:
            sec_len = len(sec.content)
            page_num = sec.page_number or 1
            sec_start = current_offset
            sec_end = current_offset + sec_len
            section_offsets[sec.section_id] = (page_num, sec_start, sec_end)
            # Add section separator length (\n\n)
            current_offset = sec_end + 2

        # 2. Populate page map entries for all discovered pages
        all_pages: set[int] = {1}
        if total_pages:
            all_pages.update(range(1, total_pages + 1))

        for sec in sections:
            if sec.page_number:
                all_pages.add(sec.page_number)
        for tbl in tables:
            if tbl.page_number:
                all_pages.add(tbl.page_number)

        for page in sorted(all_pages):
            page_map[page] = PageMapEntry(
                page_number=page,
                char_start=0,
                char_end=0,
                section_ids=[],
                table_ids=[],
            )

        # 3. Associate sections to page map
        for sec in sections:
            page_num = sec.page_number or 1
            if page_num not in page_map:
                page_map[page_num] = PageMapEntry(
                    page_number=page_num,
                    char_start=0,
                    char_end=0,
                    section_ids=[],
                    table_ids=[],
                )
            page_entry = page_map[page_num]
            if sec.section_id not in page_entry.section_ids:
                page_entry.section_ids.append(sec.section_id)

            _, s_start, s_end = section_offsets.get(sec.section_id, (page_num, 0, 0))
            if page_entry.char_start == 0 or s_start < page_entry.char_start:
                page_entry.char_start = s_start
            if s_end > page_entry.char_end:
                page_entry.char_end = s_end

        # 4. Associate tables to page map
        for tbl in tables:
            tbl_page = tbl.page_number or 1
            if tbl_page not in page_map:
                page_map[tbl_page] = PageMapEntry(
                    page_number=tbl_page,
                    char_start=0,
                    char_end=0,
                    section_ids=[],
                    table_ids=[],
                )
            if tbl.table_id not in page_map[tbl_page].table_ids:
                page_map[tbl_page].table_ids.append(tbl.table_id)

        return page_map

    @classmethod
    def update_citation_coordinates(cls, sections: list[Section]) -> list[Section]:
        """Ensure all sections have valid, updated CitationCoordinates."""
        updated: list[Section] = []
        current_offset = 0

        for idx, sec in enumerate(sections):
            sec_len = len(sec.content)
            start_off = current_offset
            end_off = current_offset + sec_len

            coords = sec.citation_coords
            if coords is None:
                coords = CitationCoordinates(
                    page_number=sec.page_number or 1,
                    section_title=sec.title,
                    paragraph_index=0,
                    char_offset_start=start_off,
                    char_offset_end=end_off,
                )
            else:
                coords = CitationCoordinates(
                    page_number=coords.page_number or sec.page_number or 1,
                    section_title=coords.section_title or sec.title,
                    paragraph_index=coords.paragraph_index or 0,
                    char_offset_start=start_off,
                    char_offset_end=end_off,
                    citation_id=coords.citation_id,
                )

            updated.append(sec.model_copy(update={"citation_coords": coords}))
            current_offset = end_off + 2

        return updated
