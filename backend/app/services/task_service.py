import time
from datetime import datetime,timezone
from sqlalchemy.orm import Session
from backend.app.database.models import Task,User
from backend.app.services.agent_service import AgentService
from backend.app.services.audit_service import audit
async def execute_task(db:Session,user:User,task:Task,request_id:str|None)->Task:
 task.status="running"; task.started_at=datetime.now(timezone.utc); db.commit(); started=time.perf_counter()
 try:
  task.output_data=await AgentService().execute(task.task_type,task.input_data); task.status="completed"; audit(db,user_id=user.id,action="task_execute",agent_action=task.task_type,model_used=task.output_data["model_used"],task_id=task.id,status="success",request_id=request_id)
 except Exception as exc:
  task.status="failed"; task.error_message="Task execution failed"; audit(db,user_id=user.id,action="task_execute",task_id=task.id,status="failed",request_id=request_id,error_message=type(exc).__name__)
 task.completed_at=datetime.now(timezone.utc); task.execution_time_ms=int((time.perf_counter()-started)*1000); db.commit(); db.refresh(task); return task
