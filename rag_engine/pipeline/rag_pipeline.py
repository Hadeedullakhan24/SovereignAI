"""End-to-End Master RAG Pipeline.

Orchestrates the complete inference workflow for Member 1:
    User Question
    ↓
    Embedding
    ↓
    Dense Retrieval (Qdrant)
    ↓
    Sparse Retrieval (BM25)
    ↓
    RRF Fusion
    ↓
    Cross-Encoder Reranking
    ↓
    Context Expansion
    ↓
    Context Packing
    ↓
    Prompt Builder
    ↓
    Prompt Validation
    ↓
    Local LLM (In-Process Hugging Face)
    ↓
    Answer Validation (Phantom Pruning)
    ↓
    Citation Injection
    ↓
    Confidence Estimation
    ↓
    Answer Formatting
    ↓
    Final RAGResponse with Execution Trace
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence

from rag_engine.generation.generation_config import GenerationConfig
from rag_engine.generation.generation_pipeline import (
    GenerationPipeline,
    GenerationResponse,
)
from rag_engine.generation.prompt.prompt_templates import (
    PromptArchetype,
    detect_task_type,
)
from rag_engine.generation.prompt.task_intent import (
    ArtifactFormat,
    TaskClassifier,
    TaskIntent,
    TaskOperation,
)
from rag_engine.generation.response_formatter import ResponseFormatter
from rag_engine.retrieval.base_retriever import (
    CitationBundle,
    RetrievalResult,
    ScoredRetrievalChunk,
)
from rag_engine.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_engine.grounding.evidence import EvidencePackage, EvidenceSelector, StructuredReport
from rag_engine.schemas.chunk import Chunk

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_FALLBACK = "The uploaded documents do not contain sufficient information to answer this question."


@dataclass(frozen=True)
class GeneratedArtifact:
    """Metadata container for physical file artifacts produced by ToolExecutor."""

    artifact_type: str
    filename: str
    file_path: str
    file_size_bytes: int
    status: str = "success"
    download_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact metadata to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class RAGExecutionTrace:
    """Granular execution trace capturing stage-by-stage timings, token metrics, and confidence."""

    question: str
    embedding_model: str
    embedding_time_ms: float
    dense_retrieval_time_ms: float
    bm25_time_ms: float
    fusion_time_ms: float
    reranking_time_ms: float
    context_packing_time_ms: float
    prompt_tokens: int
    generation_time_ms: float
    retrieved_chunks: int
    confidence: float
    total_latency_ms: float
    model_used: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize trace to JSON."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class RAGResponse:
    """Master response container combining retrieval evidence, generated grounded answer, and execution trace."""

    query: str
    answer: str
    session_id: str
    citations: list[CitationBundle]
    retrieval_result: RetrievalResult
    generation_response: GenerationResponse
    confidence_score: float
    is_grounded: bool
    total_latency_ms: float
    model_used: str = ""
    artifact: Optional[GeneratedArtifact] = None
    execution_trace: Optional[RAGExecutionTrace] = None
    evidence_package: Optional[EvidencePackage] = None
    structured_report: Optional[StructuredReport] = None

    @property
    def sources(self) -> list[str]:
        """List unique source document names."""
        seen = set()
        srcs = []
        for c in self.citations:
            name = c.document_name or c.document_id
            if name and name not in seen:
                seen.add(name)
                srcs.append(name)
        return srcs

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for API and JSON serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "session_id": self.session_id,
            "citations": [asdict(c) if hasattr(c, "__dataclass_fields__") else c for c in self.citations],
            "confidence_score": self.confidence_score,
            "is_grounded": self.is_grounded,
            "total_latency_ms": self.total_latency_ms,
            "model_used": self.model_used,
            "sources": self.sources,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "execution_trace": self.execution_trace.to_dict() if self.execution_trace else None,
        }

    def format_clean_cli_output(self) -> str:
        """Render clean, concise CLI response without debug headers."""
        clean_ans = ResponseFormatter.deduplicate_lines_and_blocks(
            ResponseFormatter.strip_provenance(self.answer)
        )
        parts = [clean_ans]
        if self.artifact:
            parts.append(
                f"\n[Generated Artifact]\n"
                f"  - Type: {self.artifact.artifact_type.upper()}\n"
                f"  - File: {self.artifact.filename}\n"
                f"  - Size: {self.artifact.file_size_bytes} bytes\n"
                f"  - Path: {self.artifact.file_path}"
            )
        concise_sources = ResponseFormatter.format_concise_sources(
            self.citations,
            query=self.query,
            answer=clean_ans,
        )
        if concise_sources:
            parts.append(f"\n{concise_sources}")
        return "\n".join(parts)

    def format_cli_output(self, detailed: bool = True) -> str:
        """Render terminal output. If detailed=True, adheres to 3-section format; otherwise returns clean output."""
        if not detailed:
            return self.format_clean_cli_output()

        detected = detect_task_type(self.query)
        header_map = {
            PromptArchetype.EMAIL: "GENERATED EMAIL",
            PromptArchetype.APPROVAL_NOTE: "GENERATED APPROVAL NOTE",
            PromptArchetype.REPORT: "GENERATED REPORT",
            PromptArchetype.SUMMARY: "GENERATED SUMMARY",
            PromptArchetype.COMPARISON: "GENERATED COMPARISON",
            PromptArchetype.EXTRACTION: "GENERATED EXTRACTION",
            PromptArchetype.SAFETY_COMPLIANCE: "GENERATED SAFETY REQUIREMENTS",
            PromptArchetype.SOP_RETRIEVAL: "GENERATED PROCEDURE",
            PromptArchetype.PROCEDURE: "GENERATED PROCEDURE",
            PromptArchetype.SPECIFICATION: "GENERATED SPECIFICATION",
            PromptArchetype.EQUIPMENT_LOOKUP: "GENERATED SPECIFICATION",
            PromptArchetype.MAINTENANCE: "GENERATED MAINTENANCE SUMMARY" if "summary" in self.query.lower() else "GENERATED ANSWER",
            PromptArchetype.TROUBLESHOOTING: "GENERATED TROUBLESHOOTING",
            PromptArchetype.ANALYSIS: "GENERATED ANALYSIS",
            PromptArchetype.GENERAL_QA: "GENERATED ANSWER",
        }

        clean_ans = ResponseFormatter.deduplicate_lines_and_blocks(
            ResponseFormatter.strip_provenance(self.answer)
        )
        if "insufficient information" in clean_ans.lower() or "insufficient evidence" in clean_ans.lower() or not self.is_grounded:
            section_title = "GENERATED RESULT"
        else:
            section_title = header_map.get(detected, "GENERATED ANSWER")

        lines: list[str] = []
        lines.append("=" * 80)
        lines.append(f"MRPL SOVEREIGN AGENTIC AI WORKBENCH - {section_title}")
        lines.append("=" * 80)
        lines.append(f"Question: {self.query}\n")

        # --------------------------------------------------
        # SECTION 1: USER RESULT FIRST
        # --------------------------------------------------
        lines.append("-" * 80)
        lines.append(f"{section_title}")
        lines.append("-" * 80)
        lines.append(clean_ans.strip())
        lines.append("")

        # --------------------------------------------------
        # ARTIFACT (IF GENERATED)
        # --------------------------------------------------
        if self.artifact:
            lines.append("-" * 80)
            lines.append("GENERATED ARTIFACT")
            lines.append("-" * 80)
            lines.append(f"  - Format       : {self.artifact.artifact_type.upper()}")
            lines.append(f"  - Filename     : {self.artifact.filename}")
            lines.append(f"  - Size         : {self.artifact.file_size_bytes} bytes")
            lines.append(f"  - Local Path   : {self.artifact.file_path}")
            if self.artifact.download_url:
                lines.append(f"  - Download URL : {self.artifact.download_url}")
            lines.append("")

        # --------------------------------------------------
        # SECTION 2: SOURCES / EVIDENCE SECOND
        # --------------------------------------------------
        lines.append("-" * 80)
        lines.append("SOURCES / EVIDENCE")
        lines.append("-" * 80)
        if self.citations:
            for idx, c in enumerate(self.citations, 1):
                anchor = getattr(c, "citation_id", f"[{idx}]")
                doc = getattr(c, "document_name", None) or getattr(c, "document_id", "Document")
                page = f"Page {c.page_number}" if getattr(c, "page_number", None) else None
                section = f"Section: {c.section_title}" if getattr(c, "section_title", None) else None
                tag = f"Tag: {c.equipment_tag}" if getattr(c, "equipment_tag", None) else None

                meta_parts = [p for p in [page, section, tag] if p]
                meta_str = f" ({' | '.join(meta_parts)})" if meta_parts else ""

                lines.append(f"  {anchor} {doc}{meta_str}")
                if getattr(c, "verbatim_quote", None):
                    quote = c.verbatim_quote.strip().replace("\n", " ")
                    if len(quote) > 160:
                        quote = quote[:157] + "..."
                    lines.append(f'      "{quote}"')
        else:
            lines.append("  (No citations referenced or required)")
        lines.append("")

        # --------------------------------------------------
        # SECTION 3: AI / RAG DETAILS LAST
        # --------------------------------------------------
        lines.append("-" * 80)
        lines.append("AI / RAG DETAILS")
        lines.append("-" * 80)
        lines.append(f"  - Confidence Score    : {self.confidence_score * 100:.1f}%")
        lines.append(f"  - Factual Grounding   : {'Grounded' if self.is_grounded else 'Warning: Low Grounding'}")
        if self.execution_trace:
            lines.append(f"  - Retrieved Chunks    : {self.execution_trace.retrieved_chunks}")
        elif getattr(self.retrieval_result, "candidates", None):
            lines.append(f"  - Retrieved Chunks    : {len(self.retrieval_result.candidates)}")
        lines.append(f"  - Model Used          : {self.model_used or 'local-llm'}")
        if self.execution_trace and self.execution_trace.generation_time_ms is not None:
            lines.append(f"  - Generation Time     : {self.execution_trace.generation_time_ms:.2f} ms")
        lines.append(f"  - Total RAG Latency   : {self.total_latency_ms:.2f} ms")
        lines.append("=" * 80)
        return "\n".join(lines)


def _generate_artifact_filename(intent: TaskIntent, default_ext: str = "pdf") -> str:
    """Derive a safe, clean artifact filename from user intent and timestamp."""
    ext = intent.artifact_format.value if intent.artifact_format else default_ext
    raw = intent.raw_query.lower()
    m = re.search(r"([\w-]+\." + re.escape(ext) + r")\b", raw, re.IGNORECASE)
    if m:
        candidate = Path(m.group(1)).name
        if candidate and candidate not in {".", ".."}:
            return candidate

    topic = intent.subject_topic or "report"
    clean_topic = re.sub(r"[^\w\s-]", "", topic).strip().replace(" ", "_").lower()
    clean_topic = clean_topic[:32] if clean_topic else "report"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{clean_topic}_{timestamp}.{ext}"


def _parse_grounded_text_to_sections(
    raw_text: str,
    default_title: str = "Engineering & Compliance Report",
) -> tuple[str, Optional[str], list[dict[str, Any]], Optional[str]]:
    """Parse grounded LLM Markdown answer into structured components for document generators.

    Returns (title, summary, sections, conclusion).
    """
    clean_text = ResponseFormatter.deduplicate_lines_and_blocks(
        ResponseFormatter.strip_provenance(raw_text)
    ).strip()
    lines = clean_text.splitlines()

    title = default_title
    summary: Optional[str] = None
    conclusion: Optional[str] = None
    sections: list[dict[str, Any]] = []

    current_section: dict[str, Any] = {
        "heading": "Key Requirements & Details",
        "paragraphs": [],
        "bullets": [],
        "table": None,
        "callout": None,
    }

    in_table = False
    table_rows: list[list[str]] = []

    def _flush_table() -> None:
        nonlocal in_table, table_rows
        if in_table and len(table_rows) >= 2:
            headers = table_rows[0]
            data_rows = [r for r in table_rows[1:] if not all(re.match(r"^:?-+:?$", c.strip()) for c in r if c.strip())]
            if data_rows:
                current_section["table"] = {"headers": headers, "rows": data_rows}
        in_table = False
        table_rows = []

    def _flush_current_section() -> None:
        _flush_table()
        if (
            current_section["paragraphs"]
            or current_section["bullets"]
            or current_section["table"]
            or current_section["callout"]
        ):
            h_lower = current_section["heading"].lower()
            nonlocal summary, conclusion
            if "executive summary" in h_lower or "overview" in h_lower:
                summary = " ".join(current_section["paragraphs"])
            elif "conclusion" in h_lower or "closing remarks" in h_lower or "statutory remarks" in h_lower:
                conclusion = " ".join(current_section["paragraphs"])
            else:
                sections.append(dict(current_section))

    first_line_checked = False
    for line in lines:
        s_line = line.strip()
        if not s_line:
            continue

        # Check for title in the first 3 lines
        if not first_line_checked:
            if s_line.startswith("# ") or s_line.lower().startswith("title:"):
                title = re.sub(r"^(#\s*|title:\s*)", "", s_line, flags=re.IGNORECASE).strip()
                title = title.strip("*_`#")
                first_line_checked = True
                continue
            first_line_checked = True

        # Check for markdown table line
        if s_line.startswith("|") and s_line.endswith("|"):
            in_table = True
            cells = [c.strip() for c in s_line[1:-1].split("|")]
            table_rows.append(cells)
            continue
        elif in_table:
            _flush_table()

        # Check for section headings
        heading_match = re.match(r"^(?:#{2,4}\s+|\*\*(?:Section\s*\d*:\s*)?([^*]+)\*\*|([A-Z0-9\s-]{4,}):$)", s_line)
        if heading_match and not s_line.startswith(("-", "*", "•", "1.", "2.", "3.")):
            _flush_current_section()
            h_text = s_line.lstrip("#").strip().strip("*_:")
            current_section = {
                "heading": h_text,
                "paragraphs": [],
                "bullets": [],
                "table": None,
                "callout": None,
            }
            continue

        # Check for callouts / critical notes
        if re.search(r"\b(CRITICAL|MANDATORY|WARNING|CAUTION|SAFETY NOTE):", s_line, re.IGNORECASE):
            callout_text = re.sub(r"^.*?((?:CRITICAL|MANDATORY|WARNING|CAUTION|SAFETY NOTE):.*)$", r"\1", s_line, flags=re.IGNORECASE)
            current_section["callout"] = callout_text.strip("*_")
            continue

        # Check for bullet points
        bullet_match = re.match(r"^(?:[-*•]|\d+\.)\s+(.+)$", s_line)
        if bullet_match:
            b_text = bullet_match.group(1).strip()
            current_section["bullets"].append(b_text)
            continue

        # Standard paragraph line
        current_section["paragraphs"].append(s_line)

    _flush_current_section()

    if not sections and current_section["paragraphs"]:
        sections.append(current_section)

    return title, summary, sections, conclusion


class RAGPipeline:
    """Enterprise Master RAG Pipeline combining Retrieval and In-Process Generation."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline | None = None,
        generation_pipeline: GenerationPipeline | None = None,
        config: GenerationConfig | None = None,
    ) -> None:
        self.config = config or GenerationConfig()
        self._retrieval = retrieval_pipeline
        self._generation = generation_pipeline

    @property
    def retrieval(self) -> RetrievalPipeline:
        """Lazily initialize or return configured RetrievalPipeline."""
        if self._retrieval is None:
            self._retrieval = RetrievalPipeline()
        return self._retrieval

    @property
    def generation(self) -> GenerationPipeline:
        """Lazily initialize or return configured GenerationPipeline."""
        if self._generation is None:
            self._generation = GenerationPipeline(config=self.config)
        return self._generation

    def get_generation_pipeline(self, model_name: Optional[str] = None) -> GenerationPipeline:
        """Lazily initialize or return configured GenerationPipeline for a given model."""
        if not model_name or model_name.lower() == "auto":
            return self.generation
        if not hasattr(self, "_generation_pipelines"):
            self._generation_pipelines: Dict[str, GenerationPipeline] = {}
        key = model_name.strip().lower()
        if key not in self._generation_pipelines:
            cfg = GenerationConfig(default_model_name=model_name)
            self._generation_pipelines[key] = GenerationPipeline(config=cfg)
        return self._generation_pipelines[key]

    def retrieve_context(
        self,
        question: str,
        top_k: int = 5,
        is_artifact_request: bool = False,
    ) -> Dict[str, Any]:
        """Model-independent pure retrieval contract for Member 1 -> Member 2 integration.

        Performs:
            Dense (Qdrant) + Sparse (BM25) -> RRF Fusion -> Cross-Encoder Reranking ->
            Evidence Gating -> Context Packing -> Citations -> Retrieval Trace.

        Returns structured dictionary containing:
            query, context, citations, confidence, retrieval_trace, candidates, structured_report.
        """
        start_total = time.perf_counter()
        intent = TaskClassifier.classify(question)
        is_artifact = is_artifact_request or intent.is_artifact_request

        retrieval_start = time.perf_counter()
        retrieval_result = self.retrieval.retrieve(query=question, top_k=top_k)
        retrieval_total_ms = (time.perf_counter() - retrieval_start) * 1000.0

        rm = getattr(retrieval_result, "metrics", None)
        dense_time = getattr(rm, "dense_latency_ms", 0.0) if rm else 0.0
        sparse_time = getattr(rm, "sparse_latency_ms", 0.0) if rm else 0.0
        fusion_time = getattr(rm, "fusion_latency_ms", 0.0) if rm else 0.0
        rerank_time = getattr(rm, "rerank_latency_ms", 0.0) if rm else 0.0
        packing_time = getattr(rm, "context_packing_latency_ms", 0.0) if rm else 0.0

        candidates = getattr(retrieval_result, "candidates", [])
        evidence_package = EvidenceSelector().select(
            question,
            candidates,
            retrieval_result.citations,
            is_artifact_request=is_artifact,
        )
        selected = evidence_package.selected
        report_title = f"{intent.subject_topic.title()} Report" if intent.subject_topic else "Grounded Evidence Report"
        structured_report = evidence_package.report(report_title)
        filtered_result = retrieval_result.model_copy(update={
            "scored_chunks": [item.chunk for item in selected],
            "citations": list(evidence_package.citations),
            "packed_context": evidence_package.context(),
        })

        is_report_valid, validation_errors = structured_report.validate()
        insufficient_for_artifact = (
            is_artifact
            and (
                not is_report_valid
                or not structured_report.scope_established
                or structured_report.missing_information
                or not selected
            )
        )

        has_sufficient = bool(selected) and self._has_sufficient_evidence(question, [item.chunk for item in selected]) and not insufficient_for_artifact
        total_ms = (time.perf_counter() - start_total) * 1000.0

        citations_list = list(evidence_package.citations)
        confidence = 0.95 if has_sufficient else 0.20

        trace = {
            "question": question,
            "embedding_model": "local-bge",
            "dense_time_ms": round(dense_time, 2),
            "sparse_time_ms": round(sparse_time, 2),
            "fusion_time_ms": round(fusion_time, 2),
            "rerank_time_ms": round(rerank_time, 2),
            "context_packing_time_ms": round(packing_time, 2),
            "retrieval_total_ms": round(total_ms, 2),
            "retrieved_chunks": len(selected),
            "confidence": confidence,
        }

        return {
            "query": question,
            "context": evidence_package.context(),
            "citations": citations_list,
            "confidence": confidence,
            "retrieval_trace": trace,
            "candidates": [item.chunk for item in selected],
            "structured_report": structured_report,
            "has_sufficient_evidence": has_sufficient,
            "filtered_result": filtered_result,
            "evidence_package": evidence_package,
        }

    def _has_sufficient_evidence(
        self,
        question: str,
        candidates: Sequence[ScoredRetrievalChunk],
        min_relevance_score: float = 0.22,
    ) -> bool:
        """Determine whether retrieved evidence actually supports answering the question."""
        if not candidates or len(candidates) == 0:
            return False

        import re
        from rag_engine.retrieval.retrieval_utils import QUERY_STOPWORDS, tokenize_refinery_text
        meta_words = frozenset(
            "create generate report document available evidence actual prepare export produce build "
            "make draft write download artifact file format using use containing contain only based "
            "provide according please".split()
        )
        q_tokens = [
            t for t in tokenize_refinery_text(question)
            if t not in QUERY_STOPWORDS and t not in meta_words and len(t) > 2
        ]
        top_score = max((c.score for c in candidates), default=0.0)

        if not q_tokens:
            # If query has no substantive subject tokens, require high retrieval confidence
            return top_score >= 0.65

        context_parts = []
        for c in candidates[:3]:
            meta = c.chunk.metadata
            context_parts.append(c.chunk.content.lower())
            if meta:
                context_parts.append((getattr(meta, "document_name", "") or "").lower())
                context_parts.append((getattr(meta, "section_title", "") or "").lower())
                context_parts.append(" ".join(getattr(meta, "equipment_entities", []) or []).lower())
                context_parts.append(" ".join(getattr(meta, "safety_entities", []) or []).lower())
        all_context = " ".join(context_parts)
        norm_context = re.sub(r"[\s\-_]+", "", all_context)

        matching_tokens = 0
        for t in q_tokens:
            norm_t = re.sub(r"[\s\-_]+", "", t)
            if t in all_context or (norm_t and norm_t in norm_context):
                matching_tokens += 1

        overlap = matching_tokens / len(q_tokens) if q_tokens else 0.0
        if overlap < 0.25:
            logger.info(
                "Query content overlap %.2f below threshold 0.25; reporting insufficient evidence.",
                overlap,
            )
            return False

        # If retrieval score is below threshold on models using 0-1 scale, require higher content overlap
        # (Allows RRF / sparse BM25 scores which naturally range between 0.01 and 0.20 when overlap is strong).
        if top_score < min_relevance_score and overlap < 0.50:
            logger.info(
                "Top retrieval score %.4f below threshold %.4f and overlap %.2f < 0.50; reporting insufficient evidence.",
                top_score,
                min_relevance_score,
                overlap,
            )
            return False

        return True

    def _inspect_requested_sources(self, intent: TaskIntent, source_paths: Sequence[str]) -> tuple[list[ScoredRetrievalChunk], list[CitationBundle], list[str]]:
        """Turn explicitly supplied document/visual sources into gated evidence."""
        if not intent.requires_source_evidence and not intent.requires_vision_analysis:
            return [], [], []
        referenced_paths = list(source_paths) or [name for name in intent.requested_source_names if name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".svg", ".dwg"))]
        if not referenced_paths:
            return [], [], (["No referenced visual source path was provided."] if intent.requires_vision_analysis else [])
        from agent.tool_executor import VisionInspectorTool, DEFAULT_PROJECT_ROOT, DEFAULT_SANDBOX_DIR
        inspector = VisionInspectorTool(DEFAULT_SANDBOX_DIR, DEFAULT_PROJECT_ROOT)
        candidates: list[ScoredRetrievalChunk] = []
        citations: list[CitationBundle] = []
        errors: list[str] = []
        for path in referenced_paths:
            try:
                is_visual = str(path).lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".svg", ".dwg"))
                result = inspector.inspect(path, question=intent.raw_query, use_vlm=is_visual)
                extracted = "\n".join(p for p in [result.get("text", ""), result.get("summary", ""), (result.get("vlm") or {}).get("answer", "")] if p).strip()
                if result.get("status") == "error" or not extracted:
                    errors.append(f"No usable visual evidence was extracted from {path}.")
                    continue
                name = result.get("filename", str(path).replace("\\", "/").split("/")[-1])
                chunk = Chunk.create(document_id=f"visual:{name}", content=extracted, chunk_index=0, document_name=name, source_path=result.get("path", str(path)), section_title="Vision and OCR extraction", equipment_entities=list(intent.explicit_entities), image_reference=result.get("path", str(path)))
                candidate = ScoredRetrievalChunk(chunk=chunk, score=1.0, rank=len(candidates), explainability="explicit requested source processed by VisionInspectorTool")
                citation = CitationBundle(citation_id=f"[V{len(citations) + 1}]", document_id=chunk.metadata.document_id, document_name=name, source_path=chunk.metadata.source_path, page_number=None, section_title=chunk.metadata.section_title, chunk_id=chunk.chunk_id, verbatim_quote=extracted, score=1.0, equipment_tags=list(intent.explicit_entities), sha256=chunk.metadata.sha256)
                candidates.append(candidate)
                citations.append(citation)
            except Exception as exc:
                errors.append(f"Unable to analyze referenced visual source {path}: {exc}")
        return candidates, citations, errors

    @staticmethod
    def _user_provided_context(intent: TaskIntent) -> tuple[list[ScoredRetrievalChunk], str]:
        """Create non-citable context for free-form instructions from the user.

        This is deliberately separate from retrieved citations: it authorizes
        the model to retain the user's own statements, but can never make them
        appear to be a finding in a PDF, image, or report.
        """
        if not intent.raw_query or not intent.user_requirements:
            return [], ""
        requirements = intent.user_requirements
        content = "USER_PROVIDED_INFORMATION:\n" + "\n".join(f"- {item}" for item in requirements)
        content += "\nSOURCE_GROUNDED_INFORMATION: no document claim is implied by the user-provided information above."
        chunk = Chunk.create(
            document_id="__user_provided__", content=content, chunk_index=0,
            document_name="USER_PROVIDED_INFORMATION", section_title="User requirements",
        )
        return [ScoredRetrievalChunk(chunk=chunk, score=1.0, rank=0, explainability="explicit user-provided requirements")], content

    @staticmethod
    def _preserve_user_email_points(email: str, intent: TaskIntent) -> str:
        """Preserve non-factual user instructions without mislabeling them as evidence."""
        if intent.output_format.value != "email" or not intent.explicit_email_instructions:
            return email
        result = email
        missing = [p for p in intent.explicit_email_instructions if p.casefold() not in result.casefold()]
        if not missing:
            return result
        addition = "\n".join(f"Additionally, {point.rstrip('.')}." for point in missing)
        closing = re.search(r"\n\s*(?:Best regards|Regards|Sincerely),", result, re.IGNORECASE)
        if closing:
            return result[:closing.start()].rstrip() + "\n\n" + addition + "\n" + result[closing.start():].lstrip()
        return result.rstrip() + "\n\n" + addition

    def _create_artifact(
        self,
        intent: TaskIntent,
        generation_response: GenerationResponse,
        report: StructuredReport,
    ) -> Optional[GeneratedArtifact]:
        """Execute ToolExecutor to generate physical artifact from grounded RAG response."""
        # 1. Pre-execution report validation
        is_valid, validation_errors = report.validate()
        if not is_valid:
            logger.warning("StructuredReport failed pre-rendering validation: %s", validation_errors)
            return None

        try:
            from agent.tool_executor import ToolExecutor
            executor = ToolExecutor(rag_pipeline=self)

            tool_name = "pdf_generator"
            if intent.artifact_format:
                format_map = {
                    ArtifactFormat.PDF: "pdf_generator",
                    ArtifactFormat.DOCX: "document_generator",
                    ArtifactFormat.XLSX: "xlsx_generator",
                    ArtifactFormat.PPTX: "pptx_generator",
                }
                tool_name = format_map.get(intent.artifact_format, "pdf_generator")

            filename = _generate_artifact_filename(
                intent,
                default_ext=intent.artifact_format.value if intent.artifact_format else "pdf",
            )

            # Artifacts must never parse or independently reinterpret model
            # prose.  Every renderer receives this one canonical evidence report.
            payload = report.renderer_payload()

            metadata = {
                "Grounded": "Yes" if generation_response.grounding_report.is_grounded else "Caution",
                "Confidence": f"{generation_response.confidence.composite_score * 100:.1f}%",
                "Model": self.generation.model.model_name,
            }

            tool_result = executor.execute(
                tool_name,
                task=intent.raw_query,
                title=payload["title"],
                summary=payload["summary"],
                sections=payload["sections"],
                conclusion=payload["conclusion"],
                citations=payload["citations"],
                structured_report=report,
                filename=filename,
                metadata=metadata,
                use_rag_context=False,
            )

            if tool_result.status == "success" and tool_result.output:
                out = tool_result.output
                return GeneratedArtifact(
                    artifact_type=intent.artifact_format.value if intent.artifact_format else "pdf",
                    filename=out.get("filename", filename),
                    file_path=out.get("path", ""),
                    file_size_bytes=out.get("file_size_bytes", 0),
                    status="success",
                    metadata=out,
                )
            else:
                logger.warning("Artifact generation tool returned non-success: %s", tool_result.error)
                return None
        except Exception as exc:
            logger.error("Failed to generate artifact: %s", exc, exc_info=True)
            return None

    def answer(
        self,
        question: str,
        session_id: str = "default_session",
        archetype: PromptArchetype | str = PromptArchetype.GENERAL_QA,
        top_k: int = 5,
        model_name: Optional[str] = None,
        source_paths: Optional[Sequence[str]] = None,
        decision: Optional[Any] = None,
    ) -> RAGResponse:
        start_total = time.perf_counter()

        # Resolve authoritative routing decision if not supplied
        if decision is None:
            from agent.router import get_router
            decision = get_router().route(question)

        from agent.router import Capability, RoutingDecision
        intent = TaskClassifier.classify(question)
        routed_model = model_name
        if not routed_model or routed_model.lower() == "auto":
            if self.config.default_model_name and self.config.default_model_name.lower() != "auto":
                routed_model = self.config.default_model_name
            else:
                try:
                    from agent.router import get_router
                    decision = get_router().route(question)
                    if decision.model_record:
                        routed_model = decision.model_record.hf_repo_id
                except Exception:
                    routed_model = None
        gen_pipeline = self.get_generation_pipeline(routed_model) if routed_model else self.generation

        # 0. Image Generation Route: bypass document text retrieval completely
        if (
            decision.capability == Capability.IMAGE_GENERATION
            or intent.operation == TaskOperation.GENERATE_IMAGE
            or intent.artifact_format == ArtifactFormat.PNG
        ):
            from agent.tool_executor import get_tool_executor
            executor = get_tool_executor(rag_pipeline=self)

            if decision.capability != Capability.IMAGE_GENERATION:
                from agent.router import get_router
                router = get_router()
                model_rec = router.model_registry.ready_for_role("image_generation")
                decision = RoutingDecision(
                    capability=Capability.IMAGE_GENERATION,
                    archetype=PromptArchetype.GENERAL_QA,
                    model_record=model_rec[0] if model_rec else None,
                    capability_available=bool(model_rec),
                    tool_name="image_generator",
                    use_rag_context=False,
                    reason=f"Authoritative image generation route for: {question!r}",
                )

            tool_result = executor.execute(decision, task=question)
            total_ms = (time.perf_counter() - start_total) * 1000.0

            artifact = None
            if tool_result.status == "success" and tool_result.output:
                out = tool_result.output
                artifact = GeneratedArtifact(
                    artifact_type="png",
                    filename=out.get("filename", "generated_image.png"),
                    file_path=out.get("path", out.get("image_path", "")),
                    file_size_bytes=out.get("file_size_bytes", 0),
                    status="success",
                    metadata=out,
                )
                answer_text = (
                    f"Successfully generated image artifact '{artifact.filename}' ({out.get('width', 512)}x{out.get('height', 512)}) "
                    f"using local {out.get('model_name', 'stable-diffusion-v1-5')} on {out.get('device', 'cuda')}.\n"
                    f"Output Path: {artifact.file_path}\n"
                    f"Generation Time: {out.get('generation_time_seconds', 0):.2f}s | Peak VRAM: {out.get('peak_vram_mb', 0):.1f} MB"
                )
            else:
                answer_text = f"Image generation failed: {tool_result.error}"

            from rag_engine.generation.generation_metrics import GenerationMetrics
            from rag_engine.generation.guardrails.citation_validator import CitationValidationReport
            from rag_engine.generation.guardrails.confidence_scorer import ConfidenceScorer
            from rag_engine.generation.guardrails.hallucination_guard import GroundingVerificationReport

            gen_resp = GenerationResponse(
                session_id=session_id,
                query=question,
                answer=answer_text,
                raw_answer=answer_text,
                prompt_payload=None,
                citations=[],
                citation_report=CitationValidationReport(
                    cleaned_text=answer_text,
                    total_citations_found=0,
                    valid_citations=[],
                    phantom_citations=[],
                    citation_precision=1.0,
                    is_valid=True,
                ),
                grounding_report=GroundingVerificationReport(
                    grounding_score=1.0,
                    verified_entities=[],
                    unverified_entities=[],
                    is_grounded=True,
                ),
                confidence=ConfidenceScorer().calculate(
                    retrieval_confidence=1.0,
                    citation_precision=1.0,
                    grounding_score=1.0,
                ),
                metrics=GenerationMetrics(
                    session_id=session_id,
                    query=question,
                    model_name="stable-diffusion-v1-5",
                    prompt_tokens=0,
                    generated_tokens=0,
                ),
            )

            trace = RAGExecutionTrace(
                question=question,
                embedding_model="none",
                embedding_time_ms=0.0,
                dense_retrieval_time_ms=0.0,
                bm25_time_ms=0.0,
                fusion_time_ms=0.0,
                reranking_time_ms=0.0,
                context_packing_time_ms=0.0,
                prompt_tokens=0,
                generation_time_ms=total_ms,
                retrieved_chunks=0,
                confidence=1.0,
                total_latency_ms=round(total_ms, 2),
                model_used="stable-diffusion-v1-5",
            )

            return RAGResponse(
                query=question,
                answer=answer_text,
                session_id=session_id,
                citations=[],
                retrieval_result=RetrievalResult(query=question, candidates=[], citations=[]),
                generation_response=gen_resp,
                confidence_score=1.0,
                is_grounded=True,
                total_latency_ms=round(total_ms, 2),
                model_used="stable-diffusion-v1-5",
                artifact=artifact,
                execution_trace=trace,
            )

        # 1. Retrieval Phase (Milestone 8)
        retrieval_start = time.perf_counter()
        retrieval_result = self.retrieval.retrieve(
            query=question,
            top_k=max(top_k, 10) if intent.output_format.value == "email" and intent.requires_source_evidence else top_k,
        )
        retrieval_total_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # Extract retrieval stage metrics if available
        rm = getattr(retrieval_result, "metrics", None)
        dense_time = getattr(rm, "dense_latency_ms", 0.0) if rm else 0.0
        sparse_time = getattr(rm, "sparse_latency_ms", 0.0) if rm else 0.0
        fusion_time = getattr(rm, "fusion_latency_ms", 0.0) if rm else 0.0
        rerank_time = getattr(rm, "rerank_latency_ms", 0.0) if rm else 0.0
        packing_time = getattr(rm, "context_packing_latency_ms", 0.0) if rm else 0.0

        # 2. Evidence gate. Retrieval rank is not evidence: classify and scope
        # candidates before prompt construction or artifact creation.
        candidates = list(getattr(retrieval_result, "candidates", []))
        retrieval_citations = list(retrieval_result.citations)
        visual_candidates, visual_citations, visual_errors = self._inspect_requested_sources(intent, source_paths or ())
        candidates.extend(visual_candidates)
        retrieval_citations.extend(visual_citations)
        evidence_package = EvidenceSelector().select(
            question,
            candidates,
            retrieval_citations,
            is_artifact_request=intent.is_artifact_request,
            exact_entity_only=(intent.output_format.value == "email" and bool(intent.explicit_entities)),
            requested_source_names=intent.requested_source_names,
        )
        selected = evidence_package.selected
        report_title = f"{intent.subject_topic.title()} Report" if intent.subject_topic else "Grounded Evidence Report"
        structured_report = evidence_package.report(report_title)
        user_candidates, user_context = self._user_provided_context(intent)
        # A format-only request (for example, an email based entirely on the
        # user's own details) must not acquire unrelated facts from ambient
        # retrieval. Document evidence remains mandatory when requested.
        user_only = bool(user_candidates) and not intent.requires_source_evidence and not intent.is_artifact_request
        generation_items = [] if user_only else selected
        filtered_result = retrieval_result.model_copy(update={
            "scored_chunks": [item.chunk for item in generation_items] + user_candidates,
            "citations": list(evidence_package.citations) if not user_only else [],
            "packed_context": "\n\n".join(part for part in [evidence_package.context() if not user_only else "", user_context] if part),
        })

        is_report_valid, validation_errors = structured_report.validate()
        insufficient_for_artifact = (
            intent.is_artifact_request
            and (
                not is_report_valid
                or not structured_report.scope_established
                or structured_report.missing_information
                or not selected
            )
        )
        insufficient_for_email_source = (
            intent.output_format.value == "email"
            and ((intent.requires_vision_analysis and bool(visual_errors)) or (intent.requires_source_evidence and not selected))
        )

        has_user_only_task_context = user_only
        lacks_usable_evidence = (
            not has_user_only_task_context
            and (not selected or not self._has_sufficient_evidence(question, [item.chunk for item in selected]))
        )
        if (lacks_usable_evidence or insufficient_for_artifact or insufficient_for_email_source):
            total_ms = (time.perf_counter() - start_total) * 1000.0
            # Construct synthetic GenerationResponse for fallback
            prompt_payload = gen_pipeline.prompt_builder.build_prompt(
                query=question,
                candidates=[],
                citations=[],
                archetype=archetype,
                model_name=gen_pipeline.model.model_name,
            )
            from rag_engine.generation.generation_metrics import GenerationMetrics
            from rag_engine.generation.guardrails.citation_validator import CitationValidationReport
            from rag_engine.generation.guardrails.confidence_scorer import ConfidenceBreakdown
            from rag_engine.generation.guardrails.hallucination_guard import GroundingVerificationReport

            fallback_gen = GenerationResponse(
                session_id=session_id,
                query=question,
                answer=INSUFFICIENT_EVIDENCE_FALLBACK,
                raw_answer=INSUFFICIENT_EVIDENCE_FALLBACK,
                prompt_payload=prompt_payload,
                citations=[],
                citation_report=CitationValidationReport(
                    cleaned_text=INSUFFICIENT_EVIDENCE_FALLBACK,
                    total_citations_found=0,
                    valid_citations=[],
                    phantom_citations=[],
                    citation_precision=1.0,
                    is_valid=True,
                ),
                grounding_report=GroundingVerificationReport(
                    grounding_score=1.0,
                    verified_entities=[],
                    unverified_entities=[],
                    is_grounded=True,
                ),
                confidence=gen_pipeline.confidence_scorer.calculate(
                    retrieval_confidence=0.0,
                    citation_precision=0.0,
                    grounding_score=1.0,
                ),
                metrics=GenerationMetrics(
                    session_id=session_id,
                    query=question,
                    model_name=gen_pipeline.model.model_name,
                    prompt_tokens=0,
                    generated_tokens=len(INSUFFICIENT_EVIDENCE_FALLBACK.split()),
                ),
            )
            fallback_conf = fallback_gen.confidence.composite_score
            trace = RAGExecutionTrace(
                question=question,
                embedding_model="local-bge",
                embedding_time_ms=0.0,
                dense_retrieval_time_ms=dense_time,
                bm25_time_ms=sparse_time,
                fusion_time_ms=fusion_time,
                reranking_time_ms=rerank_time,
                context_packing_time_ms=packing_time,
                prompt_tokens=prompt_payload.estimated_tokens,
                generation_time_ms=0.0,
                retrieved_chunks=0,
                confidence=fallback_conf,
                total_latency_ms=round(total_ms, 2),
                model_used=gen_pipeline.model.model_name,
            )
            return RAGResponse(
                query=question,
                answer=INSUFFICIENT_EVIDENCE_FALLBACK,
                session_id=session_id,
                citations=[],
                retrieval_result=retrieval_result,
                generation_response=fallback_gen,
                confidence_score=fallback_conf,
                is_grounded=True,
                total_latency_ms=round(total_ms, 2),
                model_used=gen_pipeline.model.model_name,
                artifact=None,
                execution_trace=trace,
                evidence_package=evidence_package,
                structured_report=structured_report,
            )

        # 3. Generation Phase (Milestone 9)
        effective_archetype = (
            intent.archetype
            if archetype in (PromptArchetype.GENERAL_QA, "general_qa")
            else archetype
        )
        generation_start = time.perf_counter()
        generation_response = gen_pipeline.generate(
            query=question,
            retrieval_result=filtered_result,
            session_id=session_id,
            archetype=effective_archetype,
        )
        # The model receives these directives, but retain them deterministically
        # as a final invariant. They are user-provided instructions, never
        # represented as document findings.
        if intent.output_format.value == "email":
            preserved = self._preserve_user_email_points(generation_response.answer, intent)
            if preserved != generation_response.answer:
                generation_response = replace(generation_response, answer=preserved, raw_answer=preserved)
        generation_ms = (time.perf_counter() - generation_start) * 1000.0

        # Physical artifact creation when requested
        artifact: Optional[GeneratedArtifact] = None
        if intent.is_artifact_request and generation_response.grounding_report.is_grounded:
            artifact = self._create_artifact(intent, generation_response, structured_report)

        total_ms = (time.perf_counter() - start_total) * 1000.0

        # 4. Build Execution Trace
        trace = RAGExecutionTrace(
            question=question,
            embedding_model="local-bge",
            embedding_time_ms=round(dense_time * 0.2, 2),
            dense_retrieval_time_ms=round(dense_time, 2),
            bm25_time_ms=round(sparse_time, 2),
            fusion_time_ms=round(fusion_time, 2),
            reranking_time_ms=round(rerank_time, 2),
            context_packing_time_ms=round(packing_time, 2),
            prompt_tokens=generation_response.metrics.prompt_tokens,
            generation_time_ms=round(generation_ms, 2),
            retrieved_chunks=len(selected),
            confidence=generation_response.confidence.composite_score,
            total_latency_ms=round(total_ms, 2),
            model_used=gen_pipeline.model.model_name,
        )

        return RAGResponse(
            query=question,
            answer=generation_response.answer,
            session_id=session_id,
            citations=generation_response.citations,
            retrieval_result=filtered_result,
            generation_response=generation_response,
            confidence_score=generation_response.confidence.composite_score,
            is_grounded=generation_response.grounding_report.is_grounded,
            total_latency_ms=round(total_ms, 2),
            model_used=gen_pipeline.model.model_name,
            artifact=artifact,
            execution_trace=trace,
            evidence_package=evidence_package,
            structured_report=structured_report,
        )

    def query(
        self,
        query: str,
        session_id: str = "default_session",
        archetype: PromptArchetype | str = PromptArchetype.GENERAL_QA,
        top_k: int = 5,
        model_name: Optional[str] = None,
    ) -> RAGResponse:
        """Alias for answer() ensuring backward compatibility with existing tests and scripts."""
        return self.answer(
            question=query,
            session_id=session_id,
            archetype=archetype,
            top_k=top_k,
            model_name=model_name,
        )

    def stream_query(
        self,
        query: str,
        session_id: str = "default_session",
        archetype: PromptArchetype | str = PromptArchetype.GENERAL_QA,
        top_k: int = 5,
        model_name: Optional[str] = None,
    ) -> tuple[RetrievalResult, Iterator[str]]:
        """Stream RAG response tokens after performing retrieval."""
        # Requirement preservation and grounding validation are final-answer
        # invariants.  Use the canonical answer path when free-form user
        # requirements exist rather than bypassing it with raw token streaming.
        intent = TaskClassifier.classify(query)
        if intent.user_requirements:
            response = self.answer(
                question=query, session_id=session_id, archetype=archetype, top_k=top_k,
            )
            return response.retrieval_result, iter((response.answer,))
        effective_archetype = (
            detect_task_type(query)
            if archetype in (PromptArchetype.GENERAL_QA, "general_qa")
            else archetype
        )
        retrieval_result = self.retrieval.retrieve(query=query, top_k=top_k)
        routed_model = model_name
        if not routed_model or routed_model.lower() == "auto":
            if self.config.default_model_name and self.config.default_model_name.lower() != "auto":
                routed_model = self.config.default_model_name
            else:
                try:
                    from agent.router import get_router
                    decision = get_router().route(query)
                    if decision.model_record:
                        routed_model = decision.model_record.hf_repo_id
                except Exception:
                    routed_model = None
        gen_pipeline = self.get_generation_pipeline(routed_model) if routed_model else self.generation
        stream = gen_pipeline.stream_generate(
            query=query,
            retrieval_result=retrieval_result,
            session_id=session_id,
            archetype=effective_archetype,
        )
        return retrieval_result, stream

    def build_retrieved_prompt(
        self,
        query: str,
        top_k: int = 5,
        archetype: PromptArchetype | str = PromptArchetype.GENERAL_QA,
        session_id: str = "default_session",
        model_name: Optional[str] = None,
    ) -> Any:
        """Execute complete retrieval-to-prompt chain and return verified RetrievedPrompt."""
        retrieval_result = self.retrieval.retrieve(query=query, top_k=top_k)
        gen_pipeline = self.get_generation_pipeline(model_name) if model_name else self.generation
        history_text = gen_pipeline.memory.get_history_text(session_id, max_turns=3)
        return gen_pipeline.prompt_builder.build_prompt(
            query=query,
            candidates=retrieval_result.candidates,
            citations=retrieval_result.citations,
            archetype=archetype,
            conversation_history=history_text,
            model_name=gen_pipeline.model.model_name,
        )
