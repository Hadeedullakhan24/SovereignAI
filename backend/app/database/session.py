from __future__ import annotations

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from backend.app.configuration.settings import settings

if settings.database_url.startswith("sqlite:///"):
    # SQLite does not create parent directories; local development should work from a clean checkout.
    Path(settings.database_url.removeprefix("sqlite:///" )).parent.mkdir(parents=True, exist_ok=True)
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db() -> None:
    from backend.app.database import models  # ensure mappings are registered
    Base = __import__("backend.app.database.base", fromlist=["Base"]).Base
    Base.metadata.create_all(bind=engine)
