from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.app.api.dependencies import current_user
from backend.app.database.base import get_db
from backend.app.models import Task, TaskState, User
from backend.app.schemas.api import TaskIn, TaskOut
from backend.app.services.domain import audit
router=APIRouter(prefix="/tasks",tags=["tasks"])
def owned(db,id,user):
    task=db.get(Task,id)
    if not task or task.owner_id!=user.id: raise HTTPException(404,"Task not found")
    return task
def out(t): return TaskOut(id=t.id,type=t.type,state=t.state.value,result=t.result,error=t.error,attempts=t.attempts)
@router.post("",response_model=TaskOut,status_code=202)
def create(data:TaskIn,db:Session=Depends(get_db),user:User=Depends(current_user)):
    task=Task(owner_id=user.id,type=data.type,payload=data.payload,state=TaskState.QUEUED); db.add(task); db.flush(); audit(db,user.id,"task.create","task",task.id); db.commit()
    try:
        from backend.app.workers.tasks import execute_task
        execute_task.delay(task.id)
    except Exception as exc:
        task.state=TaskState.FAILED; task.error="Task could not be queued"; audit(db,user.id,"task.enqueue_failed","task",task.id); db.commit(); raise HTTPException(503,"Task broker unavailable") from exc
    return out(task)
@router.get("",response_model=list[TaskOut])
def list_tasks(db:Session=Depends(get_db),user:User=Depends(current_user)): return [out(t) for t in db.scalars(select(Task).where(Task.owner_id==user.id).order_by(Task.created_at.desc()))]
@router.get("/{task_id}",response_model=TaskOut)
def get_task(task_id:str,db:Session=Depends(get_db),user:User=Depends(current_user)): return out(owned(db,task_id,user))
@router.post("/{task_id}/cancel",response_model=TaskOut)
def cancel(task_id:str,db:Session=Depends(get_db),user:User=Depends(current_user)):
    task=owned(db,task_id,user)
    if task.state in {TaskState.COMPLETED,TaskState.FAILED}: raise HTTPException(409,"Terminal task cannot be cancelled")
    task.state=TaskState.CANCELLED; audit(db,user.id,"task.cancel","task",task.id); db.commit(); return out(task)
@router.post("/{task_id}/retry",response_model=TaskOut)
def retry(task_id:str,db:Session=Depends(get_db),user:User=Depends(current_user)):
    task=owned(db,task_id,user)
    if task.state not in {TaskState.FAILED,TaskState.CANCELLED}: raise HTTPException(409,"Task is not retryable")
    task.state=TaskState.QUEUED; task.error=None; audit(db,user.id,"task.retry","task",task.id); db.commit()
    try:
        from backend.app.workers.tasks import execute_task
        execute_task.delay(task.id)
    except Exception as exc: task.state=TaskState.FAILED; task.error="Task could not be queued"; db.commit(); raise HTTPException(503,"Task broker unavailable") from exc
    return out(task)
@router.post("/{task_id}/approve",response_model=TaskOut)
def approve_task(task_id:str,approved:bool=True,reason:str|None=None,db:Session=Depends(get_db),user:User=Depends(current_user)):
    from backend.app.models import Approval
    task=owned(db,task_id,user)
    if task.state!=TaskState.WAITING_FOR_APPROVAL: raise HTTPException(409,"Task is not awaiting approval")
    task.state=TaskState.QUEUED if approved else TaskState.CANCELLED
    record=Approval(task_id=task.id,requested_by=task.owner_id,decided_by=user.id,decision="APPROVED" if approved else "REJECTED",reason=reason)
    db.add(record); audit(db,user.id,"task.approve" if approved else "task.reject","task",task.id,decision=record.decision,reason=reason); db.commit()
    if approved:
        try:
            from backend.app.workers.tasks import execute_task
            execute_task.delay(task.id)
        except Exception:
            pass
    return out(task)

