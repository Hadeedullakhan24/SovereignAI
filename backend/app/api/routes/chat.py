import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.app.api.dependencies import current_user
from backend.app.database.base import get_db
from backend.app.models import Conversation, Message, StoredFile, User
from backend.app.schemas.api import ChatError, ChatIn, ChatOut
from backend.app.services.domain import audit
router=APIRouter(prefix="/chat",tags=["chat"])
def conversation(db,id,user):
    row=db.get(Conversation,id)
    if not row or row.user_id!=user.id: raise HTTPException(404,"Conversation not found")
    return row
def _response_error(response):
    if response.status == "failed":
        return ChatError(code="agent_failure", message=response.error or "The local agent could not complete the request.")
    if response.status == "requires_verification":
        return ChatError(code="insufficient_evidence", message=response.error or response.get_answer() or "The available evidence is insufficient to answer safely.")
    return None

@router.post("", response_model=ChatOut)
def chat(data:ChatIn,db:Session=Depends(get_db),user:User=Depends(current_user)):
    convo=conversation(db,data.conversation_id,user) if data.conversation_id else Conversation(user_id=user.id,title=data.message[:80],agent_name=data.agent,model_name=data.model)
    if not data.conversation_id: db.add(convo); db.flush()
    attachments=[]
    agent_kwargs={}
    for file_id in data.file_ids:
        item=db.get(StoredFile,file_id)
        if not item or item.owner_id!=user.id: raise HTTPException(404,"Attachment not found")
        attachments.append(file_id)
        try:
            from backend.app.storage.local import LocalStorage
            fpath=str(LocalStorage().path(item.storage_key))
            if item.mime_type and any(m in item.mime_type for m in ("image","png","jpeg","jpg")):
                agent_kwargs["vision_file_path"]=fpath
            else:
                agent_kwargs["report_filename"]=fpath
        except Exception:
            pass
    db.add(Message(conversation_id=convo.id,role="user",content=data.message,attachments=attachments)); db.commit()
    agent_kwargs["session_id"]=convo.id
    agent_kwargs["conversation_id"]=convo.id
    if data.agent: agent_kwargs["agent_name"]=data.agent
    if data.model: agent_kwargs["model_name"]=data.model
    from agent.agent import SovereignAgent
    agent=SovereignAgent()
    try:
        response=agent.handle(data.message,**agent_kwargs)
        answer=response.get_answer() or ("Task completed." if response.status == "completed" else "The request could not be completed safely.")
        citations=response.citations or []
        artifact=response.get_artifact()
    except Exception as exc:
        from agent.agent import AgentResponse
        response=AgentResponse(status="failed",is_verified=False,error=str(exc),answer="The local agent failed while processing this request.")
        answer=response.answer; citations=[]; artifact=None
    assistant=Message(conversation_id=convo.id,role="assistant",content=answer,citations=citations,artifact=artifact); db.add(assistant); audit(db,user.id,"chat.message","conversation",convo.id,agent=data.agent,model=data.model); db.commit()
    result=ChatOut(conversation_id=convo.id,message_id=assistant.id,content=answer,status=response.status,is_verified=response.is_verified,insufficient_evidence=response.status=="requires_verification",citations=citations,artifact=artifact,tool_results=response.reasoning_steps or [],error=_response_error(response)).model_dump()
    if data.stream:
        def events():
            yield f"event: status\ndata: {json.dumps({'status':'processing','conversation_id':convo.id})}\n\n"
            yield f"event: message\ndata: {json.dumps(result)}\n\n"
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(events(),media_type="text/event-stream",headers={"Cache-Control":"no-cache"})
    return result
@router.get("/history")
def history(db:Session=Depends(get_db),user:User=Depends(current_user)): return list(db.scalars(select(Conversation).where(Conversation.user_id==user.id).order_by(Conversation.created_at.desc())))
@router.get("/{conversation_id}")
def get_conversation(conversation_id:str,db:Session=Depends(get_db),user:User=Depends(current_user)):
    conversation(db,conversation_id,user); return list(db.scalars(select(Message).where(Message.conversation_id==conversation_id).order_by(Message.created_at)))
@router.delete("/{conversation_id}",status_code=204)
def delete_conversation(conversation_id:str,db:Session=Depends(get_db),user:User=Depends(current_user)):
    row=conversation(db,conversation_id,user); [db.delete(m) for m in db.scalars(select(Message).where(Message.conversation_id==row.id))]; db.delete(row); audit(db,user.id,"chat.delete","conversation",row.id); db.commit()
