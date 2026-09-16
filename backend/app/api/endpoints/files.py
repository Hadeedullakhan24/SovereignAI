from pathlib import Path
from fastapi import APIRouter,Depends,Request,UploadFile,HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from backend.app.auth.dependencies import get_current_user
from backend.app.database.models import User
from backend.app.database.session import get_db
from backend.app.services.file_service import owned_file,save_upload
router=APIRouter(prefix="/files",tags=["Files"])
@router.post("/upload",status_code=201)
async def upload(file:UploadFile,request:Request,user:User=Depends(get_current_user),db:Session=Depends(get_db)):
 item=await save_upload(db,user,file,request.state.request_id)
 return {"success":True,"data":{"id":item.id,"filename":item.original_filename,"size":item.file_size,"mime_type":item.mime_type,"status":item.status}}
@router.get("/{file_id}/download")
def download(file_id:str,user:User=Depends(get_current_user),db:Session=Depends(get_db)):
 item=owned_file(db,file_id,user); path=Path(item.file_path).resolve()
 if not path.is_file(): raise HTTPException(404,"Uploaded file is unavailable")
 return FileResponse(path,filename=item.original_filename,media_type=item.mime_type)
