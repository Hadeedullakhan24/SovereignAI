from pathlib import Path
from fastapi import APIRouter,Depends
from sqlalchemy.orm import Session
from backend.app.auth.dependencies import get_current_user
from backend.app.database.models import User
from backend.app.database.session import get_db
from backend.app.integrations.document_provider import LocalDocumentProvider
from backend.app.services.file_service import owned_file,safe_path
router=APIRouter(prefix="/documents",tags=["Documents"])
@router.get("/{file_id}")
def metadata(file_id:str,user:User=Depends(get_current_user),db:Session=Depends(get_db)):
 f=owned_file(db,file_id,user); return {"success":True,"data":{"id":f.id,"filename":f.original_filename,"mime_type":f.mime_type,"size":f.file_size,"status":f.status,"created_at":f.created_at}}
@router.get("/{file_id}/content")
async def content(file_id:str,user:User=Depends(get_current_user),db:Session=Depends(get_db)):
 f=owned_file(db,file_id,user); return {"success":True,"data":await LocalDocumentProvider().retrieve(f.id,Path(f.file_path))}
