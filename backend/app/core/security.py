from __future__ import annotations
import hashlib, secrets
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
from .config import get_settings
# PBKDF2-SHA256 avoids the bcrypt 72-byte input limit while providing a
# deliberately slow, salted password hash with a maintained stdlib backend.
password_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
def hash_password(password: str) -> str: return password_context.hash(password)
def verify_password(password: str, hashed: str) -> bool: return password_context.verify(password, hashed)
def now() -> datetime: return datetime.now(timezone.utc)
def token_fingerprint(token: str) -> str: return hashlib.sha256(token.encode()).hexdigest()
def create_token(subject: str, token_type: str, expires: timedelta) -> tuple[str, datetime]:
    settings=get_settings(); expiry=now()+expires
    return jwt.encode({"sub":subject,"type":token_type,"jti":secrets.token_urlsafe(16),"exp":expiry},settings.jwt_secret,algorithm=settings.jwt_algorithm), expiry
def decode_token(token: str, expected_type: str) -> dict:
    settings=get_settings()
    try: payload=jwt.decode(token,settings.jwt_secret,algorithms=[settings.jwt_algorithm])
    except JWTError as exc: raise ValueError("Invalid or expired token") from exc
    if payload.get("type") != expected_type or not payload.get("sub"): raise ValueError("Invalid token type")
    return payload
