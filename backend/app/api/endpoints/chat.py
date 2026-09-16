from fastapi import APIRouter,Depends,HTTPException,Request
from sqlalchemy.orm import Session
from backend.app.auth.dependencies import get_current_user
from backend.app.database.models import Chat,ChatMessage,Task,User
from backend.app.database.session import get_db
from backend.app.schemas.api import ChatCreate,MessageCreate
from backend.app.services.audit_service import audit
from backend.app.services.task_service import execute_task
router=APIRouter(prefix="/chats",tags=["Chats"])
def data(value): return {"success":True,"data":value}
def owned(db,id,user):
 chat=db.get(Chat,id)
 if not chat: raise HTTPException(404,"Chat not found")
 if chat.user_id!=user.id: raise HTTPException(403,"You do not own this chat")
 return chat
def serialize(chat,full=False):
 output={"id":chat.id,"title":chat.title,"created_at":chat.created_at,"updated_at":chat.updated_at}
 if full: output["messages"]=[{"id":m.id,"role":m.role,"content":m.content,"model_used":m.model_used,"created_at":m.created_at} for m in chat.messages]
 return output
@router.post("",status_code=201)
def create(payload:ChatCreate,request:Request,user:User=Depends(get_current_user),db:Session=Depends(get_db)):
 chat=Chat(user_id=user.id,title=payload.title); db.add(chat); db.flush(); audit(db,user_id=user.id,action="chat_create",status="success",request_id=request.state.request_id); db.commit(); db.refresh(chat); return data(serialize(chat))
@router.get("")
def list_chats(user:User=Depends(get_current_user),db:Session=Depends(get_db)): return data([serialize(c) for c in db.query(Chat).filter(Chat.user_id==user.id).order_by(Chat.updated_at.desc()).all()])
@router.get("/{chat_id}")
def get(chat_id:str,user:User=Depends(get_current_user),db:Session=Depends(get_db)): return data(serialize(owned(db,chat_id,user),True))
@router.post("/{chat_id}/messages")
async def message(chat_id:str,payload:MessageCreate,request:Request,user:User=Depends(get_current_user),db:Session=Depends(get_db)):
 chat=owned(db,chat_id,user); incoming=ChatMessage(chat_id=chat.id,role="user",content=payload.content); task=Task(user_id=user.id,task_type="chat",input_data={"message":payload.content}); db.add_all([incoming,task]); db.commit(); await execute_task(db,user,task,request.state.request_id)
 if task.status != "completed":
  error=(task.output_data or {}).get("error",{})
  raise HTTPException(503,{"code":error.get("code","TASK_FAILED"),"message":error.get("message",task.error_message or "Task execution failed."),"task_id":task.id})
 reply=ChatMessage(chat_id=chat.id,role="assistant",content=task.output_data["result"],model_used=task.output_data.get("model_used")); db.add(reply); audit(db,user_id=user.id,action="chat_message",status="success",task_id=task.id,request_id=request.state.request_id); db.commit(); db.refresh(reply); return data({"message":{"id":reply.id,"role":reply.role,"content":reply.content,"model_used":reply.model_used,"citations":task.output_data.get("citations",[]),"execution_details":task.output_data.get("execution_details",{}),"artifact":task.output_data.get("artifact")},"task_id":task.id})
