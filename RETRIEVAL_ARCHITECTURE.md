# Master Retrieval & Citation Architecture Specification
## Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
### Milestone 8: Hybrid Retrieval, Cross-Encoder Reranking & Citations

---

## 1. Executive Summary & Retrieval Challenges in Refineries

In general domain RAG systems, simple dense vector search is often sufficient. However, in a petroleum refinery operating under strict OISD and PNGRB safety regulations:
1. **Equipment Tag Precision:** Dense embeddings often cluster all centrifugal pumps closely in vector space. A dense-only query for *"Operating pressure of Pump P-203"* might easily retrieve passages for *"Pump P-204"* or *"Pump P-101"* because their semantic embeddings are nearly indistinguishable.
2. **Lexical Grounding:** Engineering documents rely on exact alphanumeric strings: equipment tags (`P-203`, `MOV-101`), standards (`OISD-105`), and line numbers (`LINE-101-CS`).
3. **Safety Criticality:** Hallucinated operational parameters can cause catastrophic equipment failure or safety violations.

**The Solution:** An air-gapped **Two-Stage Hybrid Retrieval Pipeline**:
- **Stage 1 (High-Recall Candidate Retrieval):** Parallel Dense Vector Search (ChromaDB) + Sparse Lexical Search (BM25) combined via **Reciprocal Rank Fusion (RRF)**.
- **Stage 2 (High-Precision Neural Reranking):** Local **Cross-Encoder Reranker** (`BAAI/bge-reranker-base`) evaluating full cross-attention between query and candidate chunks.
- **Stage 3 (Deterministic Citation Construction):** Converts top reranked chunks into auditable citations with exact page numbers and verbatim quotes.

---

## 2. Multi-Stage Retrieval Architecture

```mermaid
flowchart TD
    UserQuery[Engineer / Agent Query String] --> PreProc[Query Preprocessor & Entity Extractor]
    
    subgraph Stage1_CandidateRetrieval [Stage 1: High-Recall Hybrid Search]
        PreProc -->|Dense Vector Embedding| VSearch[Dense Vector Search ChromaDB Top-25]
        PreProc -->|Extracted Keywords & Tags| SSearch[Sparse BM25 Search Inverted Index Top-25]
        
        VSearch --> DenseList[Dense Ranked Candidates]
        SSearch --> SparseList[Sparse Ranked Candidates]
        
        DenseList & SparseList --> RRF[Reciprocal Rank Fusion RRF Algorithm]
        RRF --> FusedList[Top-15 Fused Candidate Chunks]
    end

    subgraph Stage2_NeuralReranking [Stage 2: High-Precision Reranking]
        FusedList --> CE[Local Cross-Encoder BGE-Reranker-Base]
        UserQuery --> CE
        CE -->|Cross-Attention Scoring| Reranked[Top-5 Scored Chunks with Calibrated Probabilities]
    end

    subgraph Stage3_CitationBuilding [Stage 3: Context & Citation Synthesis]
        Reranked --> CitBuilder[Citation & Source Attribution Builder]
        CitBuilder --> InlineCit[Inline Citation Anchors [1], [2]]
        CitBuilder --> GroundedQuotes[Verbatim Quotes & Exact Page Numbers]
        CitBuilder --> FinalDoc[RetrievedDocument Container]
    end
```

---

## 3. Detailed Component Specifications

### 3.1 Retriever Interface (`base_retriever.py`)
```python
class BaseRetriever(ABC):
    """Abstract Base Class for retrieval strategies."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> RetrievedDocument:
        """
        Retrieve and rank relevant chunks for a natural language query.

        Args:
            query: User or agent question.
            top_k: Number of final chunks to return.
            filters: Optional metadata filters (e.g. {'plant_unit': 'CDU-1'}).

        Returns:
            RetrievedDocument with scored chunks and verifiable citations.
        """
        ...
```

### 3.2 Stage 1: Reciprocal Rank Fusion (RRF)
RRF merges ranked lists from disparate retrieval mechanisms without requiring score normalization:

$$RRF\_Score(d \in D) = \sum_{m \in \{dense, sparse\}} \frac{1}{k + rank_m(d)}$$

Where:
- $k = 60$ (Standard constant preventing top-ranked outliers from dominating).
- $rank_m(d)$ is the 1-based rank position of chunk $d$ in system $m$.
- If a chunk appears in both the dense and sparse candidate sets, its score is compounded, giving it top priority.

### 3.3 Stage 2: Cross-Encoder Neural Reranking (`reranker.py`)
Bi-encoders (used for initial embedding) encode query and passage separately, missing cross-token interactions. The **Cross-Encoder** (`BAAI/bge-reranker-base`) feeds `(query, passage)` jointly into a multi-layer transformer with full bidirectional cross-attention:

```python
class LocalCrossEncoderReranker:
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.model = CrossEncoder(str(model_path), device=device, local_files_only=True)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int = 5) -> list[ScoredChunk]:
        pairs = [[query, c.chunk.content] for c in candidates]
        scores = self.model.predict(pairs)
        
        # Attach rerank score and sort descending
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
            
        candidates.sort(key=lambda x: x.rerank_score or -999.0, reverse=True)
        return candidates[:top_k]
```

### 3.4 Stage 3: Citation & Provenance Builder (`citation_builder.py`)
Converts retrieved chunks into exact citation anchors:
```python
class CitationBuilder:
    def build_citations(self, scored_chunks: list[ScoredChunk]) -> list[Citation]:
        citations = []
        for idx, item in enumerate(scored_chunks, start=1):
            chunk = item.chunk
            meta = chunk.metadata
            
            cit = Citation(
                citation_id=f"[{idx}]",
                source=Source(
                    document_id=meta.document_id,
                    document_name=meta.document_name,
                    source_path=meta.source_path,
                    page_number=meta.page_number,
                    section_title=meta.section_title,
                    sha256=meta.sha256,
                ),
                verbatim_quote=chunk.content[:250] + ("..." if len(chunk.content) > 250 else ""),
                relevance_score=item.rerank_score or item.score,
            )
            citations.append(cit)
        return citations
```

### 3.5 Context Builder for LLM Prompt Synthesis
Formats the prompt payload consumed by Member 3 (Agent / LLM):

```text
======================================================================
REFINERY OPERATIONAL CONTEXT (VERIFIED GROUND TRUTH)
======================================================================

[1] Source: Emerson_Control_Valve_Handbook.pdf | Page: 42 | Section: 3.1 Crude Feed Circuit
Equipment Tags: P-203, HX-01, MOV-101 | Standard: OISD-105
Quote: "Centrifugal feed pump P-203 discharges crude feed directly to shell-and-tube heat exchanger HX-01 at an operating pressure of 14.5 bar and an operating temperature of 135 °C..."

[2] Source: CDU1_Standard_Operating_Procedure.pdf | Page: 15 | Section: 2.4 Isolation Protocol
Equipment Tags: MOV-101, LINE-101-CS | Standard: OISD-105
Quote: "Bypass valve MOV-101 must remain locked closed during normal continuous operations. DANGER: Ensure compliance with OISD-105 isolation procedures prior to servicing line LINE-101-CS..."

======================================================================
STRICT INSTRUCTIONS FOR THE MODEL:
1. Answer the user query using ONLY the verified context facts above.
2. For EVERY technical statement, cite the exact source using bracketed numbers (e.g. [1], [2]).
3. State the exact page number and equipment tag when discussing operational limits.
4. If the provided context does not contain the answer, reply: "The on-premise knowledge base does not contain verified data for this query."
======================================================================
User Query: What is the operating pressure limit of Pump P-203 in CDU-1?
Answer:
```
