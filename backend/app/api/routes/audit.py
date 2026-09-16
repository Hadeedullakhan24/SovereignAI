from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.app.api.dependencies import current_user, require_role
from backend.app.database.base import get_db
from backend.app.models import AuditEvent, Role, User
router=APIRouter(prefix="/audit",tags=["audit"])
@router.get("")
def events(db:Session=Depends(get_db),user:User=Depends(current_user)): return list(db.scalars(select(AuditEvent).where(AuditEvent.actor_id==user.id).order_by(AuditEvent.created_at.desc()).limit(200)))
@router.get("/admin")
def all_events(db:Session=Depends(get_db),user:User=Depends(require_role(Role.ADMIN))): return list(db.scalars(select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(500)))
