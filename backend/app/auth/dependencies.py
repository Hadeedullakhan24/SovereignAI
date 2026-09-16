from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from backend.app.auth.jwt import decode_access_token
from backend.app.database.models import User
from backend.app.database.session import get_db
oauth2_scheme=OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")
def get_current_user(token: str=Depends(oauth2_scheme),db: Session=Depends(get_db))->User:
    try: claims=decode_access_token(token)
    except ValueError: raise HTTPException(status.HTTP_401_UNAUTHORIZED,"Invalid or expired authentication token",headers={"WWW-Authenticate":"Bearer"})
    user=db.get(User,claims["sub"])
    if not user or not user.is_active: raise HTTPException(status.HTTP_401_UNAUTHORIZED,"Inactive or unknown user")
    return user
