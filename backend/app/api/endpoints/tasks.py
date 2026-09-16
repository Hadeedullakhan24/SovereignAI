from fastapi import APIRouter,Depends,HTTPException,Request
from sqlalchemy.orm import Session
from backend.app.auth.dependencies import get_current_user
from backend.app.database.models import Task,User
from backend.app.database.session import get_db
from backend.app.schemas.api import TaskCreate
from backend.app.services.task_service import execute_task
router=APIRouter(prefix="/tasks",tags=["Tasks"])
def serialize(t): return {"id":t.id,"task_type":t.task_type,"status":t.status,"input":t.input_data,"output":t.output_data,"error_message":t.error_message,"execution_time_ms":t.execution_time_ms,"created_at":t.created_at,"completed_at":t.completed_at}
@router.post("",status_code=201)
async def create(payload:TaskCreate,request:Request,user:User=Depends(get_current_user),db:Session=Depends(get_db)):
 t=Task(user_id=user.id,task_type=payload.task_type,input_data=payload.input); db.add(t); db.commit(); await execute_task(db,user,t,request.state.request_id); return {"success":True,"data":serialize(t)}
