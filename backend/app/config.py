# backend/app/config.py
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: str = "dev"
    TZ: str = "America/Detroit"
    PORT: int = 8000

    PUBLIC_BASE_URL: str = "http://127.0.0.1:8000"
    DB_URL: str = "sqlite:///./data/app.db"

    # Scheduler controls
    DAILY_PLAN_HOUR: int = 7
    DAILY_PLAN_MINUTE: int = 30
    MIDDAY_NUDGE_HOUR: int = 13
    MIDDAY_NUDGE_MINUTE: int = 0

    # Repo pipeline schedule
    REPO_SYNC_HOUR: int = 6
    REPO_SYNC_MINUTE: int = 0
    REPO_SYNC_PROJECT: str = "haven"

    # What the scheduler sends
    DAILY_PLAN_MODE: str = "single"  # single | split
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

    # --- Repo task rules ---
    HAVEN_REPO_ONLY: bool = True

    # Preferred: local path scan (no tokens)
    # If set, github_sync should read files from disk instead of GitHub API.
    REPO_LOCAL_PATH: str | None = None
    REPO_LOCAL_GIT_SHA: str | None = None

    # --- GitHub repo sync ---
    GITHUB_REPO: str = "AzRea7/OneHaven"
    GITHUB_BRANCH: str = "main"
    GITHUB_TOKEN: str | None = None

    GITHUB_MAX_FILE_BYTES: int = 150_000
    GITHUB_MAX_FILES_PER_SYNC: int = 5_000

    GITHUB_CONNECT_TIMEOUT_S: float = 10.0
    GITHUB_READ_TIMEOUT_S: float = 60.0

    


    # GitHub sync exclusions
    GITHUB_EXCLUDE_PREFIXES: list[str] = [
        ".git/",
        ".github/",
        ".venv/",
        "venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        ".cache/",
        "coverage/",
        ".idea/",
        ".vscode/",
        "node_modules/",
        "dist/",
        "build/",
    ]

    GITHUB_EXCLUDE_DIR_NAMES: list[str] = [
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".git",
        ".idea",
        ".vscode",
        "dist",
        "build",
    ]

    GITHUB_EXCLUDE_EXTENSIONS: list[str] = [
        "pyc",
        "pyo",
        "exe",
        "dll",
        "db",
        "pem",
        "key",
        "p12",
        "pfx",
        "zip",
        "gz",
        "tar",
        "7z",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "webp",
    ]
 
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
