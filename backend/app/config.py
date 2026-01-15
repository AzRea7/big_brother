# backend/app/config.py
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: str = "dev"
    TZ: str = "America/Detroit"
    PORT: int = 8000

    DB_URL: str = "sqlite:///./data/app.db"

    DAILY_PLAN_HOUR: int = 7
    DAILY_PLAN_MINUTE: int = 30
    MIDDAY_NUDGE_HOUR: int = 13
    MIDDAY_NUDGE_MINUTE: int = 0

    WEBHOOK_URL: str | None = None

    SMTP_HOST: str | None = None
    SMTP_PORT: int = 587
    SMTP_USER: str | None = None
    SMTP_PASS: str | None = None
    EMAIL_FROM: str | None = None
    EMAIL_TO: str | None = None

    # --- LLM (OFF by default) ---
    LLM_ENABLED: bool = False

    # If you enable it, you must set these appropriately.
    # For OpenAI:
    #   OPENAI_BASE_URL=https://api.openai.com/v1
    #   OPENAI_API_KEY=...
    #   OPENAI_MODEL=gpt-4o-mini (or whatever you want)
    #
    # For LM Studio (local OpenAI-compatible server):
    #   OPENAI_BASE_URL=http://127.0.0.1:1234/v1
    #   OPENAI_API_KEY=dummy
    #   OPENAI_MODEL=your-local-model-name
    OPENAI_BASE_URL: str = ""
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = ""


settings = Settings()
