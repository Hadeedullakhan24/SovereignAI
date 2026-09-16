"""Regression coverage for authentication, token revocation, and ownership."""
from __future__ import annotations
import os, sys, types
from pathlib import Path
os.environ.setdefault("SOVEREIGNAI_DATABASE_URL", "sqlite:///./backend-api-test.db")
os.environ.setdefault("SOVEREIGNAI_UPLOAD_DIR", "./backend-api-test-uploads")
from fastapi.testclient import TestClient
from backend.app.database.base import Base, engine
from backend.app.main import app

def setup_function():
    Base.metadata.drop_all(engine); Base.metadata.create_all(engine)
def _account(client: TestClient, email: str):
    body={"email":email,"password":"correct-horse-battery-staple"}
    assert client.post("/api/v1/auth/register",json=body).status_code==201
    response=client.post("/api/v1/auth/login",json=body); assert response.status_code==200
    return response.json()
def test_protected_routes_and_logout_revokes_access_token():
    client=TestClient(app); tokens=_account(client,"one@example.com"); headers={"Authorization":"Bearer "+tokens["access_token"]}
    assert client.get("/api/v1/tasks").status_code==401
    assert client.get("/api/v1/tasks",headers=headers).status_code==200
    assert client.post("/api/v1/auth/logout",headers=headers,json={"refresh_token":tokens["refresh_token"]}).status_code==204
    assert client.get("/api/v1/tasks",headers=headers).status_code==401
    assert client.post("/api/v1/auth/refresh",json={"refresh_token":tokens["refresh_token"]}).status_code==401
def test_task_and_conversation_ownership(monkeypatch):
    fake=types.SimpleNamespace(delay=lambda task_id: None)
    monkeypatch.setitem(sys.modules,"backend.app.workers.tasks",types.SimpleNamespace(execute_task=fake))
    client=TestClient(app); a=_account(client,"a@example.com"); b=_account(client,"b@example.com")
    ha={"Authorization":"Bearer "+a["access_token"]}; hb={"Authorization":"Bearer "+b["access_token"]}
    task=client.post("/api/v1/tasks",headers=ha,json={"type":"test"}).json()
    assert client.get("/api/v1/tasks/"+task["id"],headers=hb).status_code==404
    convo=client.post("/api/v1/chat",headers=ha,json={"message":"hello"}).json()
    assert client.get("/api/v1/chat/"+convo["conversation_id"],headers=hb).status_code==404
def test_task_approval_gate_and_rbac(monkeypatch):
    fake=types.SimpleNamespace(delay=lambda task_id: None)
    monkeypatch.setitem(sys.modules,"backend.app.workers.tasks",types.SimpleNamespace(execute_task=fake))
    client=TestClient(app); user=_account(client,"engineer@example.com")
    headers={"Authorization":"Bearer "+user["access_token"]}
    # Create task
    task=client.post("/api/v1/tasks",headers=headers,json={"type":"statutory_inspection"}).json()
    # Transition to WAITING_FOR_APPROVAL
    from backend.app.database.base import SessionLocal
    from backend.app.models import Task, TaskState
    with SessionLocal() as db:
        t=db.get(Task,task["id"]); t.state=TaskState.WAITING_FOR_APPROVAL; db.commit()
    # Approve task
    resp=client.post(f"/api/v1/tasks/{task['id']}/approve",headers=headers,params={"approved":True,"reason":"Approved by Inspector"})
    assert resp.status_code==200
    assert resp.json()["state"]==TaskState.QUEUED.value

