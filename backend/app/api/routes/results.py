from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.api.dependencies import current_user
from backend.app.database.base import get_db
from backend.app.models import Task, User
router=APIRouter(prefix="/results",tags=["results"])
@router.get("/{task_id}")
def result(task_id:str,db:Session=Depends(get_db),user:User=Depends(current_user)):
    task=db.get(Task,task_id)
    if not task or task.owner_id!=user.id: raise HTTPException(404,"Task not found")
    return {"task_id":task.id,"state":task.state,"result":task.result,"error":task.error}
