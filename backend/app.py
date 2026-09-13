"""Authenticated local API over the real SovereignAI agent; no cloud services."""
from __future__ import annotations
import hashlib, hmac, json, os, secrets, shutil, sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from agent.agent import SovereignAgent
from rag_engine.config.runtime_paths import runtime_file

app = FastAPI(title="SovereignAI Local API", version="1.0")
DB_PATH, UPLOAD_DIR, ARTIFACT_DIR = runtime_file("backend", "sovereignai.db"), runtime_file("backend", "uploads"), runtime_file("backend", "artifacts")
def _now() -> str: return datetime.now(timezone.utc).isoformat()
def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True); conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; conn.execute("PRAGMA journal_mode=WAL"); return conn
def _init() -> None:
    with _db() as c: c.executescript("""CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY,password_hash TEXT NOT NULL,salt TEXT NOT NULL,created_at TEXT NOT NULL);CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY,username TEXT NOT NULL,expires_at TEXT NOT NULL);CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,username TEXT NOT NULL,status TEXT NOT NULL,response_json TEXT NOT NULL,created_at TEXT NOT NULL);CREATE TABLE IF NOT EXISTS audit_events(id INTEGER PRIMARY KEY,username TEXT,action TEXT NOT NULL,detail TEXT NOT NULL,created_at TEXT NOT NULL);""")
def _audit(user: str|None, action: str, detail: str) -> None:
    with _db() as c: c.execute("INSERT INTO audit_events(username,action,detail,created_at) VALUES(?,?,?,?)",(user,action,detail[:2000],_now()))
def _password(password: str, salt: str) -> str: return hashlib.scrypt(password.encode(),salt=salt.encode(),n=2**14,r=8,p=1).hex()
def _safe_name(name: str) -> str:
    value=Path(name).name
    if not value or value != name or value in {".",".."}: raise HTTPException(400,"Invalid filename")
    return value
class Credentials(BaseModel):
    username: str=Field(pattern=r"^[A-Za-z0-9_.-]{3,64}$")
    password: str=Field(min_length=12,max_length=256)
class ChatRequest(BaseModel):
    message: str=Field(min_length=1,max_length=20_000)
    report_filename: str|None=None
@app.on_event("startup")
def startup() -> None: _init(); UPLOAD_DIR.mkdir(parents=True,exist_ok=True); ARTIFACT_DIR.mkdir(parents=True,exist_ok=True)
def current_user(request: Request) -> str:
    _init()
    header=request.headers.get("Authorization","")
    if not header.startswith("Bearer "): raise HTTPException(401,"Bearer authentication is required")
    with _db() as c: row=c.execute("SELECT username,expires_at FROM sessions WHERE token_hash=?",(hashlib.sha256(header[7:].encode()).hexdigest(),)).fetchone()
    if not row or datetime.fromisoformat(row["expires_at"])<=datetime.now(timezone.utc): raise HTTPException(401,"Invalid or expired session")
    return str(row["username"])
@app.get("/health")
def health() -> dict[str,Any]: _init(); return {"status":"ok","offline":True,"runtime_dir":str(runtime_file())}
@app.post("/auth/register")
def register(data: Credentials, request: Request) -> dict[str,str]:
    _init()
    expected=os.environ.get("SOVEREIGNAI_BOOTSTRAP_TOKEN","")
    if not expected or not hmac.compare_digest(request.headers.get("X-Bootstrap-Token",""),expected): raise HTTPException(403,"Bootstrap registration is disabled or unauthorized")
    salt=secrets.token_hex(16)
    try:
        with _db() as c: c.execute("INSERT INTO users VALUES(?,?,?,?)",(data.username,_password(data.password,salt),salt,_now()))
    except sqlite3.IntegrityError: raise HTTPException(409,"Username already exists")
    _audit(data.username,"register","local account created"); return {"status":"created"}
@app.post("/auth/login")
def login(data: Credentials) -> dict[str,str]:
    _init()
    with _db() as c: row=c.execute("SELECT password_hash,salt FROM users WHERE username=?",(data.username,)).fetchone()
    if not row or not hmac.compare_digest(row["password_hash"],_password(data.password,row["salt"])): raise HTTPException(401,"Invalid credentials")
    token=secrets.token_urlsafe(32); expires=(datetime.now(timezone.utc)+timedelta(hours=8)).isoformat()
    with _db() as c: c.execute("INSERT INTO sessions VALUES(?,?,?)",(hashlib.sha256(token.encode()).hexdigest(),data.username,expires))
    _audit(data.username,"login","session issued"); return {"access_token":token,"token_type":"bearer","expires_at":expires}
@app.post("/uploads")
async def upload(request: Request, user: str=Depends(current_user)) -> dict[str,Any]:
    UPLOAD_DIR.mkdir(parents=True,exist_ok=True)
    """Store raw upload bytes; filename is supplied in X-Filename.

    Raw uploads avoid a hidden multipart runtime dependency in air-gapped
    deployments and preserve the same authenticated file contract.
    """
    target=UPLOAD_DIR/f"{secrets.token_hex(8)}_{_safe_name(request.headers.get('X-Filename','upload.bin'))}"
    body=await request.body()
    if not body: raise HTTPException(400,"Upload body is empty")
    target.write_bytes(body)
    _audit(user,"upload",target.name); return {"filename":target.name,"size_bytes":target.stat().st_size}
@app.post("/chat")
def chat(payload: ChatRequest, user: str=Depends(current_user)) -> dict[str,Any]:
    report=None
    if payload.report_filename:
        report=_safe_name(payload.report_filename); source=UPLOAD_DIR/report
        if not source.is_file(): raise HTTPException(404,"Uploaded report not found")
        sandbox=Path(__file__).resolve().parents[1]/"workspace_sandbox"; sandbox.mkdir(exist_ok=True); shutil.copy2(source,sandbox/report)
    response=SovereignAgent().handle(payload.message,report_filename=report); task_id=secrets.token_urlsafe(12); result=response.to_dict()
    with _db() as c: c.execute("INSERT INTO tasks VALUES(?,?,?,?,?)",(task_id,user,response.status,json.dumps(result),_now()))
    _audit(user,"chat",f"task={task_id},status={response.status}"); return {"task_id":task_id,"result":result}
@app.get("/tasks/{task_id}")
def task(task_id: str,user: str=Depends(current_user)) -> dict[str,Any]:
    with _db() as c: row=c.execute("SELECT status,response_json FROM tasks WHERE id=? AND username=?",(task_id,user)).fetchone()
    if not row: raise HTTPException(404,"Task not found")
    return {"task_id":task_id,"status":row["status"],"result":json.loads(row["response_json"])}
@app.get("/audit")
def audit(user: str=Depends(current_user)) -> list[dict[str,Any]]:
    with _db() as c: rows=c.execute("SELECT action,detail,created_at FROM audit_events WHERE username=? ORDER BY id DESC LIMIT 100",(user,)).fetchall()
    return [dict(r) for r in rows]
@app.get("/artifacts/{filename}")
def artifact(filename: str,user: str=Depends(current_user)) -> FileResponse:
    target=ARTIFACT_DIR/_safe_name(filename)
    if not target.is_file(): raise HTTPException(404,"Artifact not found")
    _audit(user,"download",target.name); return FileResponse(target,filename=target.name)
