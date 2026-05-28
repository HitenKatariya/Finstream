"""Environment-driven application settings."""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "FinStream Sentiment API"
    app_version: str = "1.0.0"
    model_name: str = Field(default="hitenvk22/FinStream-Sentiment")
    hf_token: str | None = Field(default=None)
    log_level: str = Field(default="INFO")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        """Accept JSON arrays or comma-separated origin strings."""

        if value is None:
            return ["*"]
        if isinstance(value, str):
            if not value.strip():
                return ["*"]
            if value.strip() == "*":
                return ["*"]
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
