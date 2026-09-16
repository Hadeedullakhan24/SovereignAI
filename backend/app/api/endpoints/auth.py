from datetime import datetime,timezone
from fastapi import APIRouter,Depends,HTTPException,Request
from sqlalchemy.orm import Session
from backend.app.auth.dependencies import get_current_user
from backend.app.auth.jwt import create_access_token
from backend.app.auth.password import hash_password,verify_password
from backend.app.configuration.settings import settings
from backend.app.database.models import User
from backend.app.database.session import get_db
from backend.app.schemas.api import LoginRequest,RegisterRequest
from backend.app.services.audit_service import audit
router=APIRouter(prefix="/auth",tags=["Authentication"])
def envelope(data): return {"success":True,"data":data}
@router.post("/register",status_code=201)
def register(payload:RegisterRequest,request:Request,db:Session=Depends(get_db)):
 if db.query(User).filter(User.email==payload.email.lower()).first(): raise HTTPException(409,"Email already registered")
 user=User(email=payload.email.lower(),name=payload.name,password_hash=hash_password(payload.password)); db.add(user); db.flush(); audit(db,user_id=user.id,action="register",status="success",request_id=request.state.request_id); db.commit(); db.refresh(user); return envelope({"id":user.id,"email":user.email,"name":user.name,"role":user.role})
@router.post("/login")
def login(payload:LoginRequest,request:Request,db:Session=Depends(get_db)):
 user=db.query(User).filter(User.email==payload.email.lower()).first()
 if not user or not verify_password(payload.password,user.password_hash): raise HTTPException(401,"Invalid email or password")
 if not user.is_active: raise HTTPException(403,"User account is inactive")
 user.last_login_at=datetime.now(timezone.utc); audit(db,user_id=user.id,action="login",status="success",request_id=request.state.request_id); db.commit(); return envelope({"access_token":create_access_token(user.id),"token_type":"bearer","expires_in":settings.access_token_expire_minutes*60})
@router.get("/me")
def me(user:User=Depends(get_current_user)): return envelope({"id":user.id,"email":user.email,"name":user.name,"role":user.role,"is_active":user.is_active,"created_at":user.created_at})
