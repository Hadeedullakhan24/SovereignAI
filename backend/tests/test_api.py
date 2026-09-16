import os
from uuid import uuid4
os.environ["DATABASE_URL"]="sqlite:///./storage/test_api.db"
os.environ["JWT_SECRET_KEY"]="test-secret-key-that-is-at-least-32-characters"
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database.session import init_db
init_db()
client=TestClient(app)
def register(email):
 return client.post("/api/v1/auth/register",json={"email":email,"password":"correct-horse-battery-staple","name":"Test User"})
def auth(email):
 token=client.post("/api/v1/auth/login",json={"email":email,"password":"correct-horse-battery-staple"}).json()["data"]["access_token"]; return {"Authorization":f"Bearer {token}"}
def test_auth_and_protected_endpoint():
 email=f"test-{uuid4()}@example.com"
 assert register(email).status_code==201
 assert register(email).status_code==409
 assert client.get("/api/v1/chats").status_code==401
 assert client.get("/api/v1/auth/me",headers=auth(email)).status_code==200
def test_chat_file_and_task_flow():
 email=f"flow-{uuid4()}@example.com"; assert register(email).status_code==201
 headers=auth(email); chat=client.post("/api/v1/chats",json={"title":"Test"},headers=headers).json()["data"]
 message=client.post(f"/api/v1/chats/{chat['id']}/messages",json={"content":"Hello"},headers=headers); assert message.status_code==200
 upload=client.post("/api/v1/files/upload",headers=headers,files={"file":("note.txt",b"hello document","text/plain")}); assert upload.status_code==201
 file_id=upload.json()["data"]["id"]; assert client.get(f"/api/v1/documents/{file_id}/content",headers=headers).status_code==200
 task=client.post("/api/v1/tasks",headers=headers,json={"task_type":"document_analysis","input":{"instruction":"summarize"}}); assert task.status_code==201; assert task.json()["data"]["status"]=="completed"
