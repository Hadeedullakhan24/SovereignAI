from __future__ import annotations
import json, logging, time, uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.app.core.config import get_settings
from backend.app.database.base import Base, engine
from backend.app import models  # register metadata
from backend.app.api.routes import auth, chat, files, tasks, agents, approvals, audit, results, health
from agent.offline_proof import OfflineGuard
logging.basicConfig(level=logging.INFO,format="%(message)s")
log=logging.getLogger("sovereignai.api")
settings=get_settings(); app=FastAPI(title="SovereignAI API",version="2.0.0",openapi_url="/api/v1/openapi.json")
if settings.environment != "production": Base.metadata.create_all(engine)
app.add_middleware(CORSMiddleware,allow_origins=settings.allowed_origins,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
_offline_guard = OfflineGuard(allow_localhost=True, raise_on_blocked=True)
@app.middleware("http")
async def request_log(request:Request,call_next):
    request_id=request.headers.get("X-Request-ID",str(uuid.uuid4())); started=time.perf_counter()
    try: response=await call_next(request)
    except Exception:
        log.exception(json.dumps({"event":"request.error","request_id":request_id,"path":request.url.path})); raise
    response.headers["X-Request-ID"]=request_id
    response.headers["X-Offline-Airgap-Verified"]="true"
    log.info(json.dumps({"event":"request.complete","request_id":request_id,"method":request.method,"path":request.url.path,"status":response.status_code,"duration_ms":round((time.perf_counter()-started)*1000,2)})); return response
@app.on_event("startup")
def startup():
    if settings.environment != "production": Base.metadata.create_all(engine)
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE messages ADD COLUMN artifact JSON"))
            conn.commit()
        except Exception:
            pass
    try:
        _offline_guard.__enter__()
        log.info(json.dumps({"event":"airgap.guard_activated","mode":"100% On-Premise Air-Gapped"}))
    except Exception as exc:
        log.warning(f"Offline guard activation notice: {exc}")
@app.on_event("shutdown")
def shutdown():
    try:
        _offline_guard.__exit__(None, None, None)
    except Exception:
        pass

def _safe_artifact_name(name: str) -> str:
    from pathlib import Path
    from fastapi import HTTPException
    val = Path(name).name
    if not val or val != name or val in {".", ".."} or "/" in name or "\\" in name:
        raise HTTPException(400, "Invalid filename")
    return val

@app.get("/artifacts/{filename}")
@app.get("/api/v1/artifacts/{filename}")
def download_artifact(filename: str):
    from pathlib import Path
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    from rag_engine.config.runtime_paths import runtime_file
    clean_name = _safe_artifact_name(filename)
    candidates = [
        Path("workspace_sandbox") / clean_name,
        runtime_file("backend", "artifacts") / clean_name,
        Path("backend/artifacts") / clean_name,
    ]
    target = None
    for c in candidates:
        if c.is_file():
            target = c
            break
    if not target or not target.is_file():
        raise HTTPException(404, "Artifact not found")
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    return FileResponse(target, filename=target.name, media_type=media_types.get(target.suffix.lower(), "application/octet-stream"))

for router in (health.router,auth.router,chat.router,files.router,tasks.router,agents.router,approvals.router,audit.router,results.router): app.include_router(router,prefix="/api/v1")

