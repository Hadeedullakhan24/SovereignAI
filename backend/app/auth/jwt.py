import base64, hashlib, hmac, json, time, uuid
from backend.app.configuration.settings import settings
def _encode(value: dict) -> str: return base64.urlsafe_b64encode(json.dumps(value,separators=(",",":")).encode()).rstrip(b"=").decode()
def _decode(value: str) -> dict: return json.loads(base64.urlsafe_b64decode(value+"="*(-len(value)%4)))
def create_access_token(subject: str) -> str:
    if len(settings.jwt_secret_key) < 32: raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
    head=_encode({"alg":settings.jwt_algorithm,"typ":"JWT"}); payload=_encode({"sub":subject,"iat":int(time.time()),"exp":int(time.time())+settings.access_token_expire_minutes*60,"jti":str(uuid.uuid4())}); sig=hmac.new(settings.jwt_secret_key.encode(),f"{head}.{payload}".encode(),hashlib.sha256).digest(); return f"{head}.{payload}.{base64.urlsafe_b64encode(sig).rstrip(b'=').decode()}"
def decode_access_token(token: str) -> dict:
    try:
        if len(settings.jwt_secret_key) < 32: raise ValueError
        head,payload,sig=token.split("."); expected=hmac.new(settings.jwt_secret_key.encode(),f"{head}.{payload}".encode(),hashlib.sha256).digest()
        if not hmac.compare_digest(base64.urlsafe_b64decode(sig+"="*(-len(sig)%4)),expected): raise ValueError
        claims=_decode(payload)
        if claims.get("exp",0)<=time.time() or not claims.get("sub"): raise ValueError
        return claims
    except (ValueError,UnicodeDecodeError,json.JSONDecodeError): raise ValueError("Invalid or expired token")
