from __future__ import annotations
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from backend.app.core.security import decode_token
from backend.app.database.base import get_db
from backend.app.models import RevokedAccessToken, Role, User
bearer=HTTPBearer(auto_error=False)
def current_user(credentials: HTTPAuthorizationCredentials|None=Depends(bearer), db:Session=Depends(get_db))->User:
    if not credentials: raise HTTPException(status.HTTP_401_UNAUTHORIZED,"Authentication required")
    try: payload=decode_token(credentials.credentials,"access"); subject=payload["sub"]
    except ValueError: raise HTTPException(status.HTTP_401_UNAUTHORIZED,"Invalid access token")
    if db.get(RevokedAccessToken,payload["jti"]): raise HTTPException(status.HTTP_401_UNAUTHORIZED,"Access token revoked")
    user=db.get(User,subject)
    if not user or not user.is_active: raise HTTPException(status.HTTP_401_UNAUTHORIZED,"Inactive account")
    return user
def require_role(*roles:Role):
    def checker(user:User=Depends(current_user))->User:
        if user.role not in roles: raise HTTPException(status.HTTP_403_FORBIDDEN,"Insufficient role")
        return user
    return checker
