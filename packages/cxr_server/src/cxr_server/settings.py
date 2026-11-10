"""Server configuration from environment variables (prefix CXR_). See .env.example."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CXR_", env_file=".env", extra="ignore")

    bundle_dir: Path = Path("/data/cxr/artifacts/bundles/current")
    ort_providers: str = Field(
        default="auto", description="'auto' (CUDA if available, else CPU), 'cpu', or 'cuda'"
    )
    max_upload_mb: float = 25.0
    min_image_side: int = 256
    max_image_side: int = 8192
    display_max_side: int = 1024
    cors_origins: list[str] = ["http://localhost:3000"]
    rate_limit: str = "30/minute"
    log_level: str = "INFO"
    log_json: bool = True
    warmup: bool = True

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
