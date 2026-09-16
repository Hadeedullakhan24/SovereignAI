from pathlib import Path
from fastapi import APIRouter,Depends,HTTPException,Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from backend.app.auth.dependencies import get_current_user
from backend.app.database.models import Task,User
from backend.app.database.session import get_db
from backend.app.services.audit_service import audit
router=APIRouter(prefix="/results",tags=["Results"])
def owned(db,id,user):
 t=db.get(Task,id)
 if not t: raise HTTPException(404,"Task not found")
 if t.user_id!=user.id: raise HTTPException(403,"You do not own this task")
 return t
@router.get("/{task_id}")
def result(task_id:str,user:User=Depends(get_current_user),db:Session=Depends(get_db)):
 t=owned(db,task_id,user); return {"success":True,"data":{"id":t.id,"status":t.status,"output":t.output_data,"error_message":t.error_message,"execution_time_ms":t.execution_time_ms,"has_download":bool(t.output_file_path)}}
@router.get("/{task_id}/download")
def download(task_id:str,request:Request,user:User=Depends(get_current_user),db:Session=Depends(get_db)):
 t=owned(db,task_id,user)
 if not t.output_file_path: raise HTTPException(404,"Task has no generated file")
 path=Path(t.output_file_path).resolve()
 if not path.is_file(): raise HTTPException(404,"Generated file is unavailable")
 audit(db,user_id=user.id,action="result_download",task_id=t.id,status="success",request_id=request.state.request_id); db.commit(); return FileResponse(path,filename=path.name)
