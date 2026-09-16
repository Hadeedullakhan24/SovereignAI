from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.app.api.dependencies import current_user
from backend.app.database.base import get_db
from backend.app.models import StoredFile, User
from backend.app.services.domain import audit
from backend.app.storage.local import LocalStorage
router=APIRouter(prefix="/files",tags=["files"])
def owned(db,id,user):
    row=db.get(StoredFile,id)
    if not row or row.owner_id!=user.id: from fastapi import HTTPException; raise HTTPException(404,"File not found")
    return row
@router.post("/upload",status_code=201)
async def upload(file:UploadFile=File(...),db:Session=Depends(get_db),user:User=Depends(current_user)):
    key,name,size,detected=await LocalStorage().save(file); row=StoredFile(owner_id=user.id,original_name=name,storage_key=key,mime_type=detected,size_bytes=size); db.add(row); audit(db,user.id,"file.upload","file",row.id,name=name,size=size,mime_type=detected); db.commit(); return {"id":row.id,"name":name,"status":row.status,"mime_type":detected}
@router.get("")
def list_files(db:Session=Depends(get_db),user:User=Depends(current_user)): return list(db.scalars(select(StoredFile).where(StoredFile.owner_id==user.id).order_by(StoredFile.created_at.desc())))
@router.get("/{file_id}")
def get_file(file_id:str,db:Session=Depends(get_db),user:User=Depends(current_user)): return owned(db,file_id,user)
@router.get("/{file_id}/preview")
def preview(file_id:str,db:Session=Depends(get_db),user:User=Depends(current_user)):
    row=owned(db,file_id,user); return FileResponse(LocalStorage().path(row.storage_key),media_type=row.mime_type,filename=row.original_name)
@router.get("/{file_id}/download")
def download(file_id:str,db:Session=Depends(get_db),user:User=Depends(current_user)):
    row=owned(db,file_id,user); audit(db,user.id,"file.download","file",row.id); db.commit(); return FileResponse(LocalStorage().path(row.storage_key),media_type=row.mime_type,filename=row.original_name)
@router.delete("/{file_id}",status_code=204)
def delete(file_id:str,db:Session=Depends(get_db),user:User=Depends(current_user)):
    row=owned(db,file_id,user); LocalStorage().path(row.storage_key).unlink(missing_ok=True); db.delete(row); audit(db,user.id,"file.delete","file",file_id); db.commit()
