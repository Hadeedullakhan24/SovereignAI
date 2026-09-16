"""End-to-end regression tests for SovereignAI Chat API, SovereignAgent routing, and artifact delivery."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

os.environ.setdefault("SOVEREIGNAI_DATABASE_URL", "sqlite:///./backend-chat-test.db")
os.environ.setdefault("SOVEREIGNAI_UPLOAD_DIR", "./backend-chat-test-uploads")

from fastapi.testclient import TestClient
from backend.app.database.base import Base, engine
from backend.app.main import app

client = TestClient(app)

def setup_module():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

def _auth_headers(email: str = "engineer@example.com") -> dict[str, str]:
    payload = {"email": email, "password": "correct-horse-battery-staple"}
    reg = client.post("/api/v1/auth/register", json=payload)
    if reg.status_code not in (201, 409):
        raise RuntimeError(f"Register failed: {reg.text}")
    login = client.post("/api/v1/auth/login", json=payload)
    assert login.status_code == 200, f"Login failed: {login.text}"
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_chat_normal_rag_question():
    """Test A & C: Normal question executes SovereignAgent, router selects RAG, returns grounded answer."""
    headers = _auth_headers("rag_user@example.com")
    query = "What safety precautions are mentioned in the provided documentation for hot work?"
    resp = client.post("/api/v1/chat", headers=headers, json={"message": query})
    assert resp.status_code == 200, f"Chat failed: {resp.text}"
    data = resp.json()

    # Must contain conversation and message IDs
    assert "conversation_id" in data
    assert "message_id" in data
    assert data["status"] in {"completed", "requires_verification", "failed"}
    assert isinstance(data["is_verified"], bool)
    assert isinstance(data["citations"], list)
    assert isinstance(data["tool_results"], list)

    # Must NOT return placeholder string
    assert data["content"] != "Request accepted by the configured agent provider."
    assert len(data["content"]) > 10

    # Normal Q&A must not generate an artifact
    assert data.get("artifact") is None

    # Offline airgap header must be present
    assert resp.headers.get("X-Offline-Airgap-Verified") == "true"

    # Verify message is persisted in conversation history
    hist = client.get(f"/api/v1/chat/{data['conversation_id']}", headers=headers)
    assert hist.status_code == 200
    messages = hist.json()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == data["content"]

def test_chat_calculation_routing():
    """Test mathematical calculation through chat endpoint returns deterministic AST result."""
    headers = _auth_headers("calc_user@example.com")
    query = "Calculate 10.5 * 600 / (1380 * 0.85 - 0.6 * 10.5)"
    resp = client.post("/api/v1/chat", headers=headers, json={"message": query})
    assert resp.status_code == 200
    data = resp.json()

    assert data["content"] != "Request accepted by the configured agent provider."
    assert "5.3998" in data["content"] or "5.4" in data["content"]
    assert data["status"] == "completed"
    assert data["is_verified"] is True
    assert data["insufficient_evidence"] is False
    assert data["error"] is None
    assert isinstance(data["tool_results"], list) and data["tool_results"]
    assert data.get("artifact") is None

def test_chat_negative_test_no_accidental_artifact():
    """Test 13: 'What safety precautions are mentioned in OISD_Standard_105.pdf?' must remain RAG Q&A, artifact=None."""
    headers = _auth_headers("neg_user@example.com")
    query = "What safety precautions are mentioned in OISD_Standard_105.pdf?"
    resp = client.post("/api/v1/chat", headers=headers, json={"message": query})
    assert resp.status_code == 200
    data = resp.json()

    assert data["content"] != "Request accepted by the configured agent provider."
    assert data.get("artifact") is None

def test_chat_streaming_lifecycle():
    """Test 7 & 11B: Streaming events return real result payload, never replacing answer with accepted status."""
    headers = _auth_headers("stream_user@example.com")
    query = "Calculate 25 * 4 + 50"
    resp = client.post("/api/v1/chat", headers=headers, json={"message": query, "stream": True})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    body = resp.text
    assert "event: message" in body
    assert "150" in body
    assert "event: done" in body
    assert "Request accepted by the configured agent provider." not in body

def test_chat_artifact_generation_and_download_endpoint():
    """Test 6 & 12: PDF deliverable generation, metadata return, and download through backend endpoint."""
    headers = _auth_headers("artifact_user@example.com")
    query = (
        "Prepare a professional PDF report summarizing the documented safety requirements for hot work. "
        "Use only the provided documentation. Include source references and a short conclusion. "
        "Generate the report as an actual PDF artifact."
    )
    resp = client.post("/api/v1/chat", headers=headers, json={"message": query})
    assert resp.status_code == 200
    data = resp.json()

    assert data["content"] != "Request accepted by the configured agent provider."
    artifact = data.get("artifact")
    assert artifact is not None, "Expected artifact metadata in response"
    assert artifact["artifact_type"] == "PDF"
    assert artifact["filename"].endswith(".pdf")
    assert artifact["status"] in ("verified", "generated")

    # Verify download via both endpoints
    dl1 = client.get(f"/api/v1/artifacts/{artifact['filename']}")
    assert dl1.status_code == 200
    assert dl1.headers.get("content-type") == "application/pdf"
    assert len(dl1.content) > 0

    dl2 = client.get(f"/artifacts/{artifact['filename']}")
    assert dl2.status_code == 200
    assert dl2.headers.get("content-type") == "application/pdf"

def test_artifact_path_traversal_protection():
    """Verify that path traversal attempts on artifact download are blocked."""
    resp1 = client.get("/api/v1/artifacts/../../etc/passwd")
    assert resp1.status_code in (400, 404)

    resp2 = client.get("/artifacts/..%2F..%2Fetc%2Fpasswd")
    assert resp2.status_code in (400, 404)

def test_chat_multi_turn_conversation_continuity_and_isolation():
    """Verify follow-up questions preserve conversation continuity and distinct sessions remain isolated."""
    headers = _auth_headers("multiturn_user@example.com")

    # Turn 1 in Conversation 1
    resp1 = client.post("/api/v1/chat", headers=headers, json={"message": "Calculate 10 + 20"})
    assert resp1.status_code == 200
    data1 = resp1.json()
    convo_id_1 = data1["conversation_id"]
    assert "30" in data1["content"]

    # Turn 2 in Conversation 1 (follow-up)
    resp2 = client.post("/api/v1/chat", headers=headers, json={"message": "Calculate 30 + 40", "conversation_id": convo_id_1})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["conversation_id"] == convo_id_1
    assert "70" in data2["content"]

    # Conversation 2 (isolated)
    resp3 = client.post("/api/v1/chat", headers=headers, json={"message": "Calculate 50 + 60"})
    assert resp3.status_code == 200
    data3 = resp3.json()
    convo_id_2 = data3["conversation_id"]
    assert convo_id_2 != convo_id_1
    assert "110" in data3["content"]

    # Verify message histories are isolated
    hist1 = client.get(f"/api/v1/chat/{convo_id_1}", headers=headers)
    assert hist1.status_code == 200
    msgs1 = hist1.json()
    assert len(msgs1) == 4
    assert msgs1[0]["content"] == "Calculate 10 + 20"
    assert "30" in msgs1[1]["content"]
    assert msgs1[2]["content"] == "Calculate 30 + 40"
    assert "70" in msgs1[3]["content"]

    hist2 = client.get(f"/api/v1/chat/{convo_id_2}", headers=headers)
    assert hist2.status_code == 200
    msgs2 = hist2.json()
    assert len(msgs2) == 2
    assert msgs2[0]["content"] == "Calculate 50 + 60"
    assert "110" in msgs2[1]["content"]

def test_agents_provider_is_strictly_local():
    """Verify /agents endpoint advertises only local air-gapped providers and never cloud APIs."""
    headers = _auth_headers("agent_check@example.com")
    resp = client.get("/api/v1/agents", headers=headers)
    assert resp.status_code == 200
    agents = resp.json()
    assert len(agents) >= 1
    for a in agents:
        assert "openai" not in a.get("providers", [])
        assert a.get("providers") == ["local"]

@pytest.mark.parametrize("query,extension", [
    ("Generate a PDF report on the inspection of V-2201.", ".pdf"),
    ("Prepare a Word report about the maintenance of C-118.", ".docx"),
    ("Create an Excel report containing the inspection details of V-2201.", ".xlsx"),
    ("Create a PowerPoint presentation about the maintenance activities for C-118.", ".pptx"),
])
def test_chat_artifact_contract_is_grounded_and_downloadable(query: str, extension: str):
    """Validate format routing and delivery without asserting any document facts."""
    headers = _auth_headers(f"artifact-{extension[1:]}@example.com")
    response = client.post("/api/v1/chat", headers=headers, json={"message": query})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] in {"completed", "requires_verification", "failed"}
    if payload["status"] == "completed":
        artifact = payload["artifact"]
        assert artifact and artifact["filename"].lower().endswith(extension)
        downloaded = client.get(f"/api/v1/artifacts/{artifact['filename']}")
        assert downloaded.status_code == 200
        assert downloaded.content
    else:
        # Grounding/tool failures remain explicit; they must not become a fake file.
        assert payload["artifact"] is None
        assert payload["error"] is not None

def test_chat_email_has_no_internal_provenance_section():
    headers = _auth_headers("email-user@example.com")
    response = client.post("/api/v1/chat", headers=headers, json={
        "message": "Draft an email to the maintenance team summarizing the inspection findings."
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact"] is None
    if payload["status"] == "completed":
        lowered = payload["content"].lower()
        assert "### sources" not in lowered
        assert "### provenance" not in lowered

if __name__ == "__main__":
    setup_module()
    test_chat_normal_rag_question()
    print("test_chat_normal_rag_question PASSED")
    test_chat_calculation_routing()
    print("test_chat_calculation_routing PASSED")
    test_chat_negative_test_no_accidental_artifact()
    print("test_chat_negative_test_no_accidental_artifact PASSED")
    test_chat_streaming_lifecycle()
    print("test_chat_streaming_lifecycle PASSED")
    test_chat_artifact_generation_and_download_endpoint()
    print("test_chat_artifact_generation_and_download_endpoint PASSED")
    test_artifact_path_traversal_protection()
    print("test_artifact_path_traversal_protection PASSED")
    test_chat_multi_turn_conversation_continuity_and_isolation()
    print("test_chat_multi_turn_conversation_continuity_and_isolation PASSED")
    test_agents_provider_is_strictly_local()
    print("test_agents_provider_is_strictly_local PASSED")
    print("ALL 8 REGRESSION TESTS PASSED!")
