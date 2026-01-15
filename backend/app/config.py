# backend/app/config.py
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: str = "dev"
    TZ: str = "America/Detroit"
    PORT: int = 8000

    # Used in email links. In dev you can leave it as localhost.
    # In production, set to your real domain (https://goal.yourdomain.com).
    PUBLIC_BASE_URL: str = "http://127.0.0.1:8000"

    DB_URL: str = "sqlite:///./data/app.db"

    # Scheduler controls
    DAILY_PLAN_HOUR: int = 7
    DAILY_PLAN_MINUTE: int = 30
    MIDDAY_NUDGE_HOUR: int = 13
    MIDDAY_NUDGE_MINUTE: int = 0

    # What the scheduler sends
    DAILY_PLAN_MODE: str = "single"   # single | split
    DAILY_PLAN_PROJECT: str = "onestream"  # haven | onestream | (ignored in split)

    WEBHOOK_URL: str | None = None

    SMTP_HOST: str | None = None
    SMTP_PORT: int = 587
    SMTP_USER: str | None = None
    SMTP_PASS: str | None = None
    EMAIL_FROM: str | None = None
    EMAIL_TO: str | None = None

    # --- LLM (OFF by default) ---
    LLM_ENABLED: bool = False

    OPENAI_BASE_URL: str = ""
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = ""


settings = Settings()
