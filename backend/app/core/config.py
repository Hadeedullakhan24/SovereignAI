from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SOVEREIGNAI_", extra="ignore")
    environment: str = "development"
    database_url: str = "sqlite:///./.sovereignai_runtime/api.db"
    redis_url: str | None = None
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    jwt_secret: str = Field(min_length=32, default="change-this-development-secret-before-production")
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = Field(default=15, ge=1, le=120)
    refresh_token_days: int = Field(default=14, ge=1, le=90)
    upload_dir: Path = Path(".sovereignai_runtime/uploads")
    max_upload_bytes: int = 50 * 1024 * 1024
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    @model_validator(mode="after")
    def validate_production(self) -> "Settings":
        if self.environment == "production" and (self.database_url.startswith("sqlite") or self.jwt_secret.startswith("change-this-") or not self.redis_url):
            raise ValueError("Production requires PostgreSQL and a unique JWT secret")
        return self
@lru_cache
def get_settings() -> Settings: return Settings()
