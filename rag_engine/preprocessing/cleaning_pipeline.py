"""Production 12-stage sequential Cleaning and Normalization Pipeline."""

from __future__ import annotations

import time
from typing import Any, Optional

from rag_engine.preprocessing.base_cleaner import BaseCleaner
from rag_engine.preprocessing.cleaning_events import (
    CleaningEventBus,
    CleaningFailed,
    CleaningFinished,
    CleaningStarted,
    FooterRemoved,
    HeaderRemoved,
    NormalizationApplied,
    ProtectedTokenDetected,
    get_cleaning_event_bus,
)
from rag_engine.preprocessing.cleaning_metrics import (
    CleaningMetricsCollector,
    get_cleaning_metrics_collector,
)
from rag_engine.preprocessing.engineering_token_protector import EngineeringTokenProtector
from rag_engine.preprocessing.exceptions import CorruptedDocumentError, PipelineStageError
from rag_engine.preprocessing.header_footer_detector import HeaderFooterDetector
from rag_engine.preprocessing.normalizers import (
    BulletNormalizer,
    EncodingNormalizer,
    ListNormalizer,
    UnicodeNormalizer,
)
from rag_engine.preprocessing.page_mapper import PageMapper
from rag_engine.preprocessing.processing_history import ProcessingHistoryRecorder
from rag_engine.preprocessing.table_cleaner import TableCleaner
from rag_engine.preprocessing.utils import compute_text_diff_stats
from rag_engine.preprocessing.whitespace_cleaner import WhitespaceCleaner
from rag_engine.schemas.document import DocumentLifecycleState
from rag_engine.schemas.parsed_document import (
    CleanParsedDocument,
    DocumentStatistics,
    ParsedDocument,
    Section,
)


class CleaningPipeline(BaseCleaner):
    """Production 12-stage cleaning pipeline that transforms ParsedDocument to CleanParsedDocument.

    The pipeline strictly executes stages 1 to 12 in order:
      Stage 1:  Unicode normalization (NFKC)
      Stage 2:  Encoding normalization
      Stage 3:  Whitespace normalization
      Stage 4:  Line ending normalization
      Stage 5:  Broken paragraph reconstruction
      Stage 6:  Hyphenated word reconstruction
      Stage 7:  Header/Footer detection & removal
      Stage 8:  Page number preservation & citation mapping
      Stage 9:  Table whitespace cleanup
      Stage 10: Bullet normalization
      Stage 11: List normalization
      Stage 12: Engineering token protection
    """

    def __init__(
        self,
        event_bus: Optional[CleaningEventBus] = None,
        metrics_collector: Optional[CleaningMetricsCollector] = None,
        protector: Optional[EngineeringTokenProtector] = None,
        strict_token_verification: bool = False,
    ) -> None:
        self.event_bus = event_bus or get_cleaning_event_bus()
        self.metrics = metrics_collector or get_cleaning_metrics_collector()
        self.protector = protector or EngineeringTokenProtector()
        self.header_footer_detector = HeaderFooterDetector()
        self.table_cleaner = TableCleaner()
        self.strict_token_verification = strict_token_verification

    @property
    def name(self) -> str:
        return "RefineryCleaningPipeline"

    def clean(self, document: ParsedDocument) -> CleanParsedDocument:
        """Execute the deterministic 12-stage cleaning pipeline on a ParsedDocument."""
        if document is None or not getattr(document, "document_id", None):
            raise CorruptedDocumentError("ParsedDocument is null or missing document_id")

        doc_id = document.document_id
        self.metrics.start_document(doc_id)
        self.event_bus.publish(
            CleaningStarted(
                document_id=doc_id,
                total_sections=len(document.sections),
                total_tables=len(document.tables),
            )
        )

        history: list[dict[str, Any]] = list(document.processing_history)
        cleaning_warnings: list[str] = []
        all_protected_tokens: set[str] = set()

        # Extract initial tokens to protect from raw full text
        initial_raw_text = document.get_full_text()
        initial_tokens = self.protector.extract_tokens(initial_raw_text)
        all_protected_tokens.update(initial_tokens)

        # Working state copies
        sections = [s.model_copy(deep=True) for s in document.sections]
        tables = [t.model_copy(deep=True) for t in document.tables]
        removed_headers: list[str] = []
        removed_footers: list[str] = []

        # Mask engineering tokens across sections before transformations to prevent corruption
        section_token_maps: list[dict[str, str]] = []
        all_token_map: dict[str, str] = {}
        for i, sec in enumerate(sections):
            masked_c, token_map = self.protector.mask(sec.content)
            sections[i] = sec.model_copy(update={"content": masked_c})
            section_token_maps.append(token_map)
            all_token_map.update(token_map)

        try:
            # -------------------------------------------------------------
            # STAGE 1: Unicode Normalization (NFKC)
            # -------------------------------------------------------------
            s1_before = sum(len(s.content) for s in sections)
            for i, sec in enumerate(sections):
                norm_c = UnicodeNormalizer.normalize(sec.content)
                sections[i] = sec.model_copy(update={"content": norm_c})
            s1_after = sum(len(s.content) for s in sections)
            chars_rem, chars_norm = compute_text_diff_stats(
                " ".join(s.content for s in document.sections),
                " ".join(s.content for s in sections),
            )
            self.metrics.record_characters_removed(doc_id, chars_rem)
            self.metrics.record_characters_normalized(doc_id, chars_norm)
            self.event_bus.publish(
                NormalizationApplied(
                    document_id=doc_id,
                    stage_name="Stage 1: Unicode Normalization (NFKC)",
                    characters_modified=chars_norm,
                )
            )
            history.append(
                ProcessingHistoryRecorder.create_record(
                    stage_name="Stage 1: Unicode Normalization",
                    stage_order=1,
                    status="SUCCESS",
                    characters_before=s1_before,
                    characters_after=s1_after,
                    details={"standard": "NFKC"},
                )
            )

            # -------------------------------------------------------------
            # STAGE 2: Encoding Normalization
            # -------------------------------------------------------------
            s2_before = sum(len(s.content) for s in sections)
            for i, sec in enumerate(sections):
                norm_c = EncodingNormalizer.normalize(sec.content)
                sections[i] = sec.model_copy(update={"content": norm_c})
            s2_after = sum(len(s.content) for s in sections)
            self.metrics.record_characters_removed(doc_id, max(0, s2_before - s2_after))
            self.event_bus.publish(
                NormalizationApplied(
                    document_id=doc_id,
                    stage_name="Stage 2: Encoding Normalization",
                    characters_modified=max(0, s2_before - s2_after),
                )
            )
            history.append(
                ProcessingHistoryRecorder.create_record(
                    stage_name="Stage 2: Encoding Normalization",
                    stage_order=2,
                    status="SUCCESS",
                    characters_before=s2_before,
                    characters_after=s2_after,
                    details={"control_chars_cleaned": True, "zero_width_stripped": True},
                )
            )

            # -------------------------------------------------------------
            # STAGE 3: Whitespace Normalization (Horizontal & Line Ends)
            # -------------------------------------------------------------
            s3_before = sum(len(s.content) for s in sections)
            for i, sec in enumerate(sections):
                norm_c = WhitespaceCleaner.clean_whitespace(sec.content)
                sections[i] = sec.model_copy(update={"content": norm_c})
            s3_after = sum(len(s.content) for s in sections)
            ws_reduction = max(0, s3_before - s3_after)
            self.metrics.record_whitespace_reduction(doc_id, ws_reduction)
            history.append(
                ProcessingHistoryRecorder.create_record(
                    stage_name="Stage 3: Whitespace Normalization",
                    stage_order=3,
                    status="SUCCESS",
                    characters_before=s3_before,
                    characters_after=s3_after,
                    details={"whitespace_reduced": ws_reduction},
                )
            )

            # -------------------------------------------------------------
            # STAGE 4: Line Ending Normalization
            # -------------------------------------------------------------
            s4_before = sum(len(s.content) for s in sections)
            for i, sec in enumerate(sections):
                norm_c = WhitespaceCleaner.normalize_line_endings(sec.content)
                sections[i] = sec.model_copy(update={"content": norm_c})
            s4_after = sum(len(s.content) for s in sections)
            history.append(
                ProcessingHistoryRecorder.create_record(
                    stage_name="Stage 4: Line Ending Normalization",
                    stage_order=4,
                    status="SUCCESS",
                    characters_before=s4_before,
                    characters_after=s4_after,
                    details={"newline_standard": "LF", "max_consecutive": 2},
                )
            )

            # -------------------------------------------------------------
            # STAGE 5: Broken Paragraph Reconstruction
            # -------------------------------------------------------------
            s5_before = sum(len(s.content) for s in sections)
            for i, sec in enumerate(sections):
                norm_c = WhitespaceCleaner.reconstruct_broken_paragraphs(sec.content)
                sections[i] = sec.model_copy(update={"content": norm_c})
            s5_after = sum(len(s.content) for s in sections)
            history.append(
                ProcessingHistoryRecorder.create_record(
                    stage_name="Stage 5: Broken Paragraph Reconstruction",
                    stage_order=5,
                    status="SUCCESS",
                    characters_before=s5_before,
                    characters_after=s5_after,
                    details={"soft_wraps_reconstructed": True},
                )
            )

            # -------------------------------------------------------------
            # STAGE 6: Hyphenated Word Reconstruction
            # -------------------------------------------------------------
            s6_before = sum(len(s.content) for s in sections)
            for i, sec in enumerate(sections):
                norm_c = WhitespaceCleaner.reconstruct_hyphenated_words(sec.content)
                sections[i] = sec.model_copy(update={"content": norm_c})
            s6_after = sum(len(s.content) for s in sections)
            history.append(
                ProcessingHistoryRecorder.create_record(
                    stage_name="Stage 6: Hyphenated Word Reconstruction",
                    stage_order=6,
                    status="SUCCESS",
                    characters_before=s6_before,
                    characters_after=s6_after,
                    details={"hyphenated_words_repaired": True},
                )
            )

            # -------------------------------------------------------------
            # STAGE 7: Header/Footer Detection & Safe Removal
            # -------------------------------------------------------------
            s7_before = sum(len(s.content) for s in sections)
            sections, removed_headers, removed_footers = (
                self.header_footer_detector.detect_and_remove(sections)
            )
            s7_after = sum(len(s.content) for s in sections)
            if removed_headers:
                self.metrics.record_header_removed(doc_id, len(removed_headers))
                for hdr in removed_headers:
                    self.event_bus.publish(
                        HeaderRemoved(document_id=doc_id, header_text=hdr)
                    )
            if removed_footers:
                self.metrics.record_footer_removed(doc_id, len(removed_footers))
                for ftr in removed_footers:
                    self.event_bus.publish(
                        FooterRemoved(document_id=doc_id, footer_text=ftr)
                    )
            history.append(
                ProcessingHistoryRecorder.create_record(
                    stage_name="Stage 7: Header/Footer Detection",
                    stage_order=7,
                    status="SUCCESS",
                    characters_before=s7_before,
                    characters_after=s7_after,
                    details={
                        "headers_removed_count": len(removed_headers),
                        "footers_removed_count": len(removed_footers),
                    },
                )
            )

            # -------------------------------------------------------------
            # STAGE 8: Page Number Preservation & Citation Mapping
            # -------------------------------------------------------------
            sections = PageMapper.update_citation_coordinates(sections)
            page_map = PageMapper.build_page_map(
                sections=sections,
                tables=tables,
                total_pages=document.statistics.total_pages,
            )
            history.append(
                ProcessingHistoryRecorder.create_record(
                    stage_name="Stage 8: Page Number Preservation",
                    stage_order=8,
                    status="SUCCESS",
                    characters_before=s7_after,
                    characters_after=s7_after,
                    details={"mapped_pages": len(page_map)},
                )
            )

            # -------------------------------------------------------------
            # STAGE 9: Table Whitespace Cleanup
            # -------------------------------------------------------------
            t_before = sum(len(t.normalized_text) for t in tables)
            tables = self.table_cleaner.clean_tables(tables)
            t_after = sum(len(t.normalized_text) for t in tables)
            history.append(
                ProcessingHistoryRecorder.create_record(
                    stage_name="Stage 9: Table Whitespace Cleanup",
                    stage_order=9,
                    status="SUCCESS",
                    characters_before=t_before,
                    characters_after=t_after,
                    details={"cleaned_tables_count": len(tables)},
                )
            )

            # -------------------------------------------------------------
            # STAGE 10: Bullet Normalization
            # -------------------------------------------------------------
            s10_before = sum(len(s.content) for s in sections)
            for i, sec in enumerate(sections):
                norm_c = BulletNormalizer.normalize(sec.content)
                sections[i] = sec.model_copy(update={"content": norm_c})
            s10_after = sum(len(s.content) for s in sections)
            history.append(
                ProcessingHistoryRecorder.create_record(
                    stage_name="Stage 10: Bullet Normalization",
                    stage_order=10,
                    status="SUCCESS",
                    characters_before=s10_before,
                    characters_after=s10_after,
                    details={"bullet_standard": "Markdown '- '"},
                )
            )

            # -------------------------------------------------------------
            # STAGE 11: List Normalization
            # -------------------------------------------------------------
            s11_before = sum(len(s.content) for s in sections)
            for i, sec in enumerate(sections):
                norm_c = ListNormalizer.normalize(sec.content)
                sections[i] = sec.model_copy(update={"content": norm_c})
            s11_after = sum(len(s.content) for s in sections)
            history.append(
                ProcessingHistoryRecorder.create_record(
                    stage_name="Stage 11: List Normalization",
                    stage_order=11,
                    status="SUCCESS",
                    characters_before=s11_before,
                    characters_after=s11_after,
                    details={"list_standard": "Markdown '1. '"},
                )
            )

            # -------------------------------------------------------------
            # STAGE 12: Engineering Token Protection & Verification
            # -------------------------------------------------------------
            # Unmask sections to restore all original engineering tokens
            for i, sec in enumerate(sections):
                if i < len(section_token_maps) and section_token_maps[i]:
                    unmasked_c = self.protector.unmask(sec.content, section_token_maps[i])
                    sections[i] = sec.model_copy(update={"content": unmasked_c})

            # Unmask removed headers and footers if any placeholders were present
            if all_token_map:
                removed_headers = [self.protector.unmask(h, all_token_map) for h in removed_headers]
                removed_footers = [self.protector.unmask(f, all_token_map) for f in removed_footers]

            final_full_text = "\n\n".join(s.content for s in sections if s.content.strip())
            for tbl in tables:
                final_full_text += f"\n{tbl.normalized_text}"

            # Extract any newly formatted tokens as well
            current_tokens = self.protector.extract_tokens(final_full_text)
            all_protected_tokens.update(current_tokens)

            # Verify that original tokens were not corrupted
            missing_tokens = self.protector.verify_tokens_preserved(
                original_text=initial_raw_text,
                cleaned_text=final_full_text,
                strict=self.strict_token_verification,
            )

            if missing_tokens:
                warn_msg = f"{len(missing_tokens)} tokens could not be verified in final text: {missing_tokens[:3]}"
                cleaning_warnings.append(warn_msg)
                self.metrics.record_warning(doc_id, warn_msg)

            # Record protected tokens
            sorted_tokens = sorted(all_protected_tokens)
            self.metrics.record_protected_tokens(doc_id, sorted_tokens)
            for tok in sorted_tokens:
                self.event_bus.publish(
                    ProtectedTokenDetected(document_id=doc_id, token=tok)
                )

            history.append(
                ProcessingHistoryRecorder.create_record(
                    stage_name="Stage 12: Engineering Token Protection",
                    stage_order=12,
                    status="SUCCESS",
                    characters_before=len(final_full_text),
                    characters_after=len(final_full_text),
                    details={
                        "total_protected_tokens": len(sorted_tokens),
                        "missing_tokens_count": len(missing_tokens),
                    },
                )
            )

            # -------------------------------------------------------------
            # Finalize Document Statistics & CleanParsedDocument
            # -------------------------------------------------------------
            # Update paragraph lists on each section
            for i, sec in enumerate(sections):
                paras = [p.strip() for p in sec.content.split("\n\n") if p.strip()]
                sections[i] = sec.model_copy(
                    update={
                        "paragraphs": paras,
                        "normalized_text": sec.content,
                    }
                )

            total_chars = len(final_full_text)
            total_words = len(final_full_text.split())
            updated_stats = document.statistics.model_copy(
                update={
                    "total_characters": total_chars,
                    "total_words": total_words,
                    "total_sections": len(sections),
                    "total_tables": len(tables),
                }
            )

            # Finish metrics collector
            clean_stats = self.metrics.finish_document(doc_id)

            self.event_bus.publish(
                CleaningFinished(
                    document_id=doc_id,
                    execution_time_ms=clean_stats.execution_time_ms,
                    characters_removed=clean_stats.characters_removed,
                    characters_normalized=clean_stats.characters_normalized,
                    protected_tokens_count=clean_stats.protected_tokens_count,
                )
            )

            return CleanParsedDocument(
                document_id=document.document_id,
                raw_document_id=document.raw_document_id,
                title=document.title,
                category=document.category,
                sections=sections,
                tables=tables,
                equipment=document.equipment,
                entity_graph=document.entity_graph,
                warnings=document.warnings,
                cross_references=document.cross_references,
                drawing_metadata=document.drawing_metadata,
                metadata=document.metadata,
                statistics=updated_stats,
                processing_history=history,
                validation_report=document.validation_report,
                lifecycle_state=DocumentLifecycleState.CLEANED,
                cleaning_status="CLEANED",
                cleaning_statistics=clean_stats,
                normalization_version="1.0.0",
                page_map=page_map,
                removed_headers=removed_headers,
                removed_footers=removed_footers,
                protected_tokens=sorted_tokens,
                cleaning_warnings=cleaning_warnings,
            )

        except Exception as exc:
            self.metrics.record_failure(doc_id)
            self.event_bus.publish(
                CleaningFailed(
                    document_id=doc_id,
                    stage_name="CleaningPipeline",
                    error_message=str(exc),
                )
            )
            raise PipelineStageError(
                f"Cleaning pipeline failed on document {doc_id}: {exc}",
                stage_name="CleaningPipeline",
            ) from exc
