from __future__ import annotations
from datetime import timedelta
from sqlalchemy.orm import Session
from backend.app.core.config import get_settings
from backend.app.core.security import create_token, hash_password, now, token_fingerprint, verify_password
from backend.app.models import AuditEvent, RefreshSession, Task, TaskState, User
def audit(db:Session, actor:str|None, action:str, kind:str, resource_id:str|None=None, **detail): db.add(AuditEvent(actor_id=actor,action=action,resource_type=kind,resource_id=resource_id,detail=detail))
def issue_tokens(db:Session,user:User)->dict:
    settings=get_settings(); access,_=create_token(user.id,"access",timedelta(minutes=settings.access_token_minutes)); refresh,expiry=create_token(user.id,"refresh",timedelta(days=settings.refresh_token_days)); db.add(RefreshSession(user_id=user.id,token_hash=token_fingerprint(refresh),expires_at=expiry)); return {"access_token":access,"refresh_token":refresh,"expires_in":settings.access_token_minutes*60}
def run_task(db:Session, task:Task)->Task:
    if task.state==TaskState.CANCELLED: return task
    task.state=TaskState.RUNNING; task.attempts+=1; db.flush()
    try:
        # Actual broker workers replace this adapter; no provider is hardcoded.
        task.result={"accepted":True,"task_type":task.type,"payload":task.payload}; task.state=TaskState.COMPLETED
    except Exception as exc: task.error=str(exc); task.state=TaskState.FAILED
    return task
