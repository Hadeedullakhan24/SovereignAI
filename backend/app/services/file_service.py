import secrets
from pathlib import Path
from fastapi import HTTPException,UploadFile
from sqlalchemy.orm import Session
from backend.app.configuration.settings import settings
from backend.app.database.models import File,User
from backend.app.services.audit_service import audit
def safe_path(directory:Path,filename:str)->Path:
 root=directory.resolve(); candidate=(root/filename).resolve()
 if root!=candidate.parent: raise HTTPException(400,"Unsafe file path")
 return candidate
async def save_upload(db:Session,user:User,upload:UploadFile,request_id:str|None)->File:
 if upload.content_type not in settings.allowed_mime_types: raise HTTPException(415,"Unsupported file type")
 original=Path(upload.filename or "upload").name; stored=f"{secrets.token_urlsafe(20)}{Path(original).suffix.lower()}"; settings.upload_dir.mkdir(parents=True,exist_ok=True); target=safe_path(settings.upload_dir,stored); total=0
 try:
  with target.open("xb") as output:
   while chunk:=await upload.read(1024*1024):
    total+=len(chunk)
    if total>settings.max_upload_size_mb*1024*1024: raise HTTPException(413,"File exceeds configured size limit")
    output.write(chunk)
 except Exception: target.unlink(missing_ok=True); raise
 record=File(user_id=user.id,original_filename=original,stored_filename=stored,file_path=str(target),mime_type=upload.content_type,file_size=total); db.add(record); db.flush(); audit(db,user_id=user.id,action="file_upload",status="success",request_id=request_id,file_id=record.id); db.commit(); db.refresh(record); return record
def owned_file(db:Session,file_id:str,user:User)->File:
 file=db.get(File,file_id)
 if not file: raise HTTPException(404,"File not found")
 if file.user_id!=user.id: raise HTTPException(403,"You do not own this file")
 return file
