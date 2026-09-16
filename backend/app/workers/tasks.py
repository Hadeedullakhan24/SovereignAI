from __future__ import annotations
from celery import shared_task
from backend.app.database.base import SessionLocal
from backend.app.models import Task, TaskState
from backend.app.services.agent_executor import execute_agent_task
from backend.app.services.domain import audit
@shared_task(bind=True,autoretry_for=(ConnectionError,),retry_backoff=True,retry_jitter=True,max_retries=3,name="sovereignai.execute_task")
def execute_task(self,task_id:str)->dict:
    db=SessionLocal()
    try:
        task=db.get(Task,task_id)
        if not task or task.state==TaskState.CANCELLED: return {"task_id":task_id,"status":"cancelled_or_missing"}
        task.state=TaskState.RUNNING; task.attempts+=1; audit(db,task.owner_id,"task.started","task",task.id); db.commit()
        try:
            result=execute_agent_task(task.type,task.payload)
            task.state=TaskState.COMPLETED; task.result=result; audit(db,task.owner_id,"task.completed","task",task.id)
        except Exception as exc:
            task.state=TaskState.FAILED; task.error=str(exc)[:4000]; audit(db,task.owner_id,"task.failed","task",task.id)
            db.commit(); raise
        db.commit(); return {"task_id":task.id,"status":task.state.value}
    finally: db.close()
