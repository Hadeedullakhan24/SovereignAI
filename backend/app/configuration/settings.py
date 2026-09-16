from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed configuration; never put secrets in source control."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_name: str = "SovereignAI Integration API"
    app_env: str = "development"
    debug: bool = False
    database_url: str = "sqlite:///./storage/sovereignai.db"
    # Empty by default so deployments cannot accidentally inherit a public secret.
    jwt_secret_key: str = Field(default="", min_length=0)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    max_upload_size_mb: int = 25
    allowed_file_types: str = "application/pdf,text/plain,text/markdown,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    upload_dir: Path = Path("storage/uploads")
    output_dir: Path = Path("storage/outputs")
    temp_dir: Path = Path("storage/temporary")
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    cleanup_age_hours: int = 24

    @property
    def allowed_mime_types(self) -> set[str]:
        return {item.strip() for item in self.allowed_file_types.split(",") if item.strip()}

    @property
    def origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
