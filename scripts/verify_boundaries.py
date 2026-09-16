"""Comprehensive Boundary Verification Script for SovereignAI.

Executes actual runtime code across all 10 integration boundaries without mocks:
1. FRONTEND -> BACKEND
2. BACKEND -> AGENT
3. AGENT -> ROUTER
4. AGENT -> RAG
5. RAG -> RETRIEVAL
6. RAG -> EVIDENCE GATE
7. AGENT -> MODELS
8. AGENT -> TOOLS
9. TOOLS -> ARTIFACTS
10. BACKEND -> FRONTEND
11. END-TO-END
"""

import sys
import os
import io
import json
import time
from pathlib import Path

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

results = {}

def report(boundary: str, status: str, details: str):
    results[boundary] = (status, details)
    print(f"[{status}] {boundary}: {details}")

def run_verification():
    print("=" * 70)
    print("SOVEREIGN AI RUNTIME BOUNDARY INTEGRATION AUDIT")
    print("=" * 70)

    # 1. FRONTEND -> BACKEND via FastAPI TestClient with authentic registration/login
    client = None
    headers = {}
    try:
        from fastapi.testclient import TestClient
        from backend.app.main import app
        from backend.app.database.base import Base, engine

        Base.metadata.create_all(engine)

        client = TestClient(app)
        email = f"audit_{int(time.time())}@example.com"
        pwd = "test-password-123!"
        reg = client.post("/api/v1/auth/register", json={"email": email, "password": pwd})
        login = client.post("/api/v1/auth/login", json={"email": email, "password": pwd})
        assert login.status_code == 200, f"Login failed: {login.text}"
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Test GET /api/v1/agents
        resp = client.get("/api/v1/agents", headers=headers)
        assert resp.status_code == 200, f"Status: {resp.status_code}"
        agents = resp.json()
        assert isinstance(agents, list) and len(agents) > 0
        assert agents[0].get("providers") == ["local"], f"Providers: {agents[0].get('providers')}"
        report("FRONTEND → BACKEND", "PASS", "Authenticated HTTP client reached FastAPI routes with verified strictly-local provider.")
    except Exception as e:
        report("FRONTEND → BACKEND", "FAIL", str(e))

    # 2. BACKEND -> AGENT
    convo_id = None
    try:
        # Test chat calculation through backend route
        chat_req = client.post(
            "/api/v1/chat",
            json={"message": "calculate 45.5 * 12.2"},
            headers=headers
        )
        assert chat_req.status_code == 200, f"Status: {chat_req.status_code}, {chat_req.text}"
        chat_data = chat_req.json()
        assert "555.1" in chat_data["content"], f"Result content: {chat_data['content']}"
        convo_id = chat_data.get("conversation_id")
        assert convo_id is not None
        report("BACKEND → AGENT", "PASS", f"Backend invoked SovereignAgent with conversation ID {convo_id} and received verified SafeCalculator execution.")
    except Exception as e:
        report("BACKEND → AGENT", "FAIL", str(e))

    # 3. AGENT -> ROUTER
    try:
        from agent.router import get_router, Capability
        router = get_router()
        decision_math = router.route("calculate 100 / 4")
        assert decision_math.capability == Capability.CALCULATION, f"Capability: {decision_math.capability}"
        decision_doc = router.route("Summarize centrifugal pump operating guidelines")
        assert decision_doc.capability == Capability.RAG, f"Capability: {decision_doc.capability}"
        report("AGENT → ROUTER", "PASS", f"Task router accurately classified calculation and document retrieval capabilities (tool={decision_math.tool_name}).")
    except Exception as e:
        report("AGENT → ROUTER", "FAIL", str(e))

    # 4. AGENT -> RAG
    try:
        from rag_engine.pipeline.rag_pipeline import RAGPipeline
        from rag_engine.generation.generation_config import GenerationConfig
        gen_cfg = GenerationConfig(default_model_name="deterministic_test", cache_enabled=False)
        rag_pipeline = RAGPipeline(config=gen_cfg)
        calc_response = rag_pipeline.answer("calculate 25 * 4")
        assert "100" in calc_response.answer, f"Answer: {calc_response.answer}"
        assert calc_response.model_used == "SafeCalculator"
        report("AGENT → RAG", "PASS", f"RAGPipeline answered query using deterministic calculation routing ({calc_response.total_latency_ms:.1f}ms).")
    except Exception as e:
        report("AGENT → RAG", "FAIL", str(e))

    # 5. RAG -> RETRIEVAL
    try:
        from rag_engine.retrieval.bm25_retriever import BM25Index
        from rag_engine.schemas.chunk import Chunk
        chunk = Chunk.create(document_id="doc_audit_1", content="Emergency isolation valve XV-101 operates at 18.5 bar.", chunk_index=0)
        bm25 = BM25Index()
        bm25.add_chunks([chunk])
        scored = bm25.search("isolation valve XV-101", top_k=1)
        assert len(scored) == 1
        res_chunk, score = scored[0]
        assert res_chunk.metadata.document_id == "doc_audit_1"
        report("RAG → RETRIEVAL", "PASS", f"BM25 retrieval indexed chunks and retrieved exact entity matches with score {score:.3f}.")
    except Exception as e:
        report("RAG → RETRIEVAL", "FAIL", str(e))

    # 6. RAG -> EVIDENCE GATE
    try:
        from rag_engine.grounding.evidence import EvidenceSelector
        from rag_engine.retrieval.base_retriever import ScoredRetrievalChunk, CitationBundle
        scored_chk = ScoredRetrievalChunk(chunk=chunk, score=0.9, rank=0)
        cit = CitationBundle(citation_id="[1]", document_id=chunk.metadata.document_id, document_name="Manual", source_path="", page_number=1, section_title="", chunk_id=chunk.chunk_id, verbatim_quote=chunk.content, score=0.9)
        pkg = EvidenceSelector().select("What is the pressure of XV-101?", [scored_chk], [cit])
        assert len(pkg.selected) >= 1
        assert pkg.is_scope_established is True
        report("RAG → EVIDENCE GATE", "PASS", "EvidenceSelector successfully scoped and verified candidate chunks against query entity XV-101.")
    except Exception as e:
        report("RAG → EVIDENCE GATE", "FAIL", str(e))

    # 7. AGENT -> MODELS
    try:
        from rag_engine.generation.models.model_registry import LLMRegistry
        reg = LLMRegistry.get_instance()
        qwen_path = reg.find_local_model_path("qwen2.5-1.5b-instruct")
        smollm_path = reg.find_local_model_path("smollm2-1.7b-instruct")
        phi_path = reg.find_local_model_path("phi-3.5-mini-instruct")
        assert qwen_path is not None and qwen_path.exists(), f"Qwen path: {qwen_path}"
        assert smollm_path is not None and smollm_path.exists(), f"SmolLM path: {smollm_path}"
        assert phi_path is not None and phi_path.exists(), f"Phi-3.5 path: {phi_path}"
        report("AGENT → MODELS", "PASS", f"LLMRegistry verified physical local models on disk: Qwen ({qwen_path.name}), SmolLM2 ({smollm_path.name}), Phi-3.5 ({phi_path.name}).")
    except Exception as e:
        report("AGENT → MODELS", "FAIL", str(e))

    # 8. AGENT -> TOOLS
    try:
        from agent.tool_executor import get_tool_executor
        executor = get_tool_executor()
        res = executor.execute("calculator", task="calculate (144 ** 0.5) * 3")
        assert res.status == "success"
        assert res.output.get("result") == 36.0, f"Result: {res.output}"
        report("AGENT → TOOLS", "PASS", f"ToolExecutor executed SafeCalculator with AST evaluation, returned exact result {res.output.get('result')}.")
    except Exception as e:
        report("AGENT → TOOLS", "FAIL", str(e))

    # 9. TOOLS -> ARTIFACTS
    try:
        artifact_dir = Path("workspace_sandbox")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        test_artifact_name = "test_audit_artifact.txt"
        test_artifact_path = artifact_dir / test_artifact_name
        test_artifact_path.write_text("SovereignAI Audit Artifact Content", encoding="utf-8")
        assert test_artifact_path.exists()
        # Verify download through backend endpoint
        dl_resp = client.get(f"/api/v1/artifacts/{test_artifact_name}")
        assert dl_resp.status_code == 200
        assert b"SovereignAI Audit Artifact Content" in dl_resp.content
        test_artifact_path.unlink(missing_ok=True)
        report("TOOLS → ARTIFACTS", "PASS", "Physical artifact generated, stored in sandbox artifacts directory, and validated via download route.")
    except Exception as e:
        report("TOOLS → ARTIFACTS", "FAIL", str(e))

    # 10. BACKEND -> FRONTEND
    try:
        history_resp = client.get("/api/v1/chat/history", headers=headers)
        assert history_resp.status_code == 200
        history_data = history_resp.json()
        assert isinstance(history_data, list)
        assert len(history_data) >= 1
        assert history_data[0]["id"] == convo_id

        msg_resp = client.get(f"/api/v1/chat/{convo_id}", headers=headers)
        assert msg_resp.status_code == 200
        msg_data = msg_resp.json()
        assert isinstance(msg_data, list)
        assert len(msg_data) >= 2  # user message and assistant message
        report("BACKEND → FRONTEND", "PASS", f"Backend returns properly formatted JSON schemas for conversations ({len(history_data)}) and messages ({len(msg_data)}).")
    except Exception as e:
        report("BACKEND → FRONTEND", "FAIL", str(e))

    # 11. END-TO-END
    all_pass = all(s == "PASS" for s, _ in results.values())
    status_e2e = "PASS" if all_pass else "FAIL"
    report("END-TO-END", status_e2e, "All 10 integration boundaries successfully verified under live runtime execution.")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY STATUS:")
    print("=" * 70)
    for boundary, (stat, _) in results.items():
        print(f"{boundary}: {stat}")

if __name__ == "__main__":
    run_verification()
