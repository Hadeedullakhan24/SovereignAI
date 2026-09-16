from __future__ import annotations
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from backend.app.core.config import get_settings
class Base(DeclarativeBase): pass
settings=get_settings()
if settings.database_url.startswith("sqlite:///"):
    # SQLite creates the database file, but not a missing parent directory.
    Path(settings.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True,exist_ok=True)
engine=create_engine(settings.database_url,pool_pre_ping=True,connect_args={"check_same_thread":False} if settings.database_url.startswith("sqlite") else {})
SessionLocal=sessionmaker(bind=engine,autoflush=False,autocommit=False)
def get_db():
    db=SessionLocal()
    try: yield db
    finally: db.close()
