import time
from datetime import datetime,timezone
from sqlalchemy.orm import Session
from backend.app.database.models import Task,User
from backend.app.services.agent_service import AgentService
from backend.app.services.agent_service import ModelNotReadyError
from backend.app.services.audit_service import audit
async def execute_task(db:Session,user:User,task:Task,request_id:str|None)->Task:
 task.status="running"; task.started_at=datetime.now(timezone.utc); db.commit(); started=time.perf_counter()
 try:
  task.input_data={**task.input_data,"session_id":task.id}
  task.output_data=await AgentService().execute(task.task_type,task.input_data)
  artifact=(task.output_data or {}).get("artifact")
  if artifact and artifact.get("path"):
   task.output_file_path=artifact["path"]
  task.status="completed"; audit(db,user_id=user.id,action="task_execute",agent_action=task.task_type,model_used=task.output_data["model_used"],task_id=task.id,status="success",request_id=request_id,metadata_json={"execution_details":task.output_data.get("execution_details",{}),"citation_count":len(task.output_data.get("citations",[]))})
 except ModelNotReadyError as exc:
  task.status="failed"; task.error_message=str(exc); task.output_data={"error":{"code":"MODEL_NOT_READY","message":str(exc),"model":exc.model}}; audit(db,user_id=user.id,action="task_execute",task_id=task.id,status="failed",request_id=request_id,error_message="MODEL_NOT_READY")
 except Exception as exc:
  task.status="failed"; task.error_message="Task execution failed"; audit(db,user_id=user.id,action="task_execute",task_id=task.id,status="failed",request_id=request_id,error_message=type(exc).__name__)
 task.completed_at=datetime.now(timezone.utc); task.execution_time_ms=int((time.perf_counter()-started)*1000); db.commit(); db.refresh(task); return task
