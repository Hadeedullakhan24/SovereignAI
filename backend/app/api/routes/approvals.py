from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.api.dependencies import current_user
from backend.app.database.base import get_db
from backend.app.models import Approval, Task, TaskState, User
from backend.app.services.domain import audit
router=APIRouter(prefix="/approvals",tags=["approvals"])
@router.post("/{task_id}")
def approve(task_id:str,approved:bool,db:Session=Depends(get_db),user:User=Depends(current_user)):
    task=db.get(Task,task_id)
    if not task or task.owner_id!=user.id: raise HTTPException(404,"Task not found")
    if task.state!=TaskState.WAITING_FOR_APPROVAL: raise HTTPException(409,"Task is not awaiting approval")
    task.state=TaskState.QUEUED if approved else TaskState.CANCELLED
    record=Approval(task_id=task.id,requested_by=task.owner_id,decided_by=user.id,decision="APPROVED" if approved else "REJECTED")
    db.add(record); audit(db,user.id,"approval.decide","approval",record.id,task_id=task.id,decision=record.decision); db.commit()
    return {"id":task.id,"state":task.state,"approval_id":record.id}
