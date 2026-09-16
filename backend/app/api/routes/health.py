from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from backend.app.core.config import get_settings
from backend.app.database.base import engine
router=APIRouter(tags=["health"])
@router.get("/health")
def health(): return {"status":"ok","environment":get_settings().environment}
@router.get("/ready")
def ready():
    try:
        with engine.connect() as connection: connection.execute(text("SELECT 1"))
    except Exception as exc: raise HTTPException(503,"Database unavailable") from exc
    return {"status":"ready"}
