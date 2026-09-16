from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.app.api.dependencies import current_user
from backend.app.core.security import decode_token, hash_password, now, token_fingerprint, verify_password
from backend.app.database.base import get_db
from backend.app.models import RefreshSession, RevokedAccessToken, User
from backend.app.schemas.api import LoginIn, RefreshIn, RegisterIn, TokenOut
from backend.app.services.domain import audit, issue_tokens
router=APIRouter(prefix="/auth",tags=["auth"])
@router.post("/register",status_code=201)
def register(data:RegisterIn,db:Session=Depends(get_db)):
    if db.scalar(select(User).where(User.email==data.email.lower())): raise HTTPException(409,"Email already registered")
    user=User(email=data.email.lower(),password_hash=hash_password(data.password)); db.add(user); audit(db,user.id,"user.register","user",user.id); db.commit(); return {"id":user.id,"email":user.email}
@router.post("/login",response_model=TokenOut)
def login(data:LoginIn,db:Session=Depends(get_db)):
    user=db.scalar(select(User).where(User.email==data.email.lower()))
    if not user or not verify_password(data.password,user.password_hash) or not user.is_active: raise HTTPException(status.HTTP_401_UNAUTHORIZED,"Invalid credentials")
    result=issue_tokens(db,user); audit(db,user.id,"auth.login","user",user.id); db.commit(); return result
@router.post("/refresh",response_model=TokenOut)
def refresh(data:RefreshIn,db:Session=Depends(get_db)):
    try: payload=decode_token(data.refresh_token,"refresh")
    except ValueError: raise HTTPException(401,"Invalid refresh token")
    session=db.scalar(select(RefreshSession).where(RefreshSession.token_hash==token_fingerprint(data.refresh_token)))
    user=db.get(User,payload["sub"])
    if not session or session.revoked_at or (session.expires_at if session.expires_at.tzinfo else session.expires_at.replace(tzinfo=timezone.utc)) <= now() or not user: raise HTTPException(401, "Refresh session expired")
    session.revoked_at=now(); result=issue_tokens(db,user); audit(db,user.id,"auth.refresh","session",session.id); db.commit(); return result
@router.post("/logout",status_code=204)
def logout(data:RefreshIn,request:Request,db:Session=Depends(get_db),user:User=Depends(current_user)):
    session=db.scalar(select(RefreshSession).where(RefreshSession.token_hash==token_fingerprint(data.refresh_token),RefreshSession.user_id==user.id))
    if session: session.revoked_at=now()
    token=request.headers.get("Authorization","").removeprefix("Bearer ")
    try:
        access=decode_token(token,"access")
        db.add(RevokedAccessToken(jti=access["jti"],user_id=user.id,expires_at=datetime.fromtimestamp(access["exp"],timezone.utc)))
    except (ValueError, KeyError, TypeError):
        pass
    audit(db,user.id,"auth.logout","session",session.id if session else None); db.commit()
