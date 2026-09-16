from fastapi import APIRouter,Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from backend.app.database.session import get_db
router=APIRouter(tags=["System"])
@router.get("/health")
def health(db:Session=Depends(get_db)):
 db.execute(text("SELECT 1")); return {"success":True,"data":{"status":"healthy","database":"connected","version":"1.0.0"}}
