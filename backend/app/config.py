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

    # --- GitHub repo sync (Level 1) ---
    # Repo in "owner/name" form:
    GITHUB_REPO: str = "AzRea7/OneHaven"
    GITHUB_BRANCH: str = "main"

    # Optional (recommended): classic token or fine-grained token with read-only repo access.
    # Public repos work without token but you will hit rate limits fast.
    GITHUB_TOKEN: str | None = None

    # Safety: don’t store huge/binary files in your DB.
    # Only store file content when:
    #  - under MAX_BYTES
    #  - looks like text
    GITHUB_MAX_FILE_BYTES: int = 150_000  # 150 KB
    GITHUB_MAX_FILES_PER_SYNC: int = 5_000

    # Timeouts for GitHub API calls
    GITHUB_CONNECT_TIMEOUT_S: float = 10.0
    GITHUB_READ_TIMEOUT_S: float = 60.0

    # GitHub sync exclusions
    GITHUB_EXCLUDE_PREFIXES: list[str] = [
        "onehaven/node_modules/",
        "onehaven/.venv/",
        "onehaven/venv/",
        "onehaven/dist/",
        "onehaven/build/",
        "onehaven/.pytest_cache/",
        "onehaven/__pycache__/",
        "node_modules/",
        ".venv/",
        "__pycache__/",
        ".git/",
        ".github/",
        "node_modules/",
        "dist/",
        "build/",
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
        ".venv/",
        "**/__pycache__/",
        "**/*.pyc",
        "node_modules/",
        ".git/",
        "dist/", 
        "build/",
        "backend/.venv/",
        "backend/__pycache__/",
        "backend/.pytest_cache/",
        "backend/.mypy_cache/",
        "backend/.ruff_cache/",
        "backend/.git/",
        ".git/",
        "backend/app/__pycache__/",
        "backend/app/services/__pycache__/",
        "backend/app/routes/__pycache__/",
        "backend/app/models/__pycache__/",
    ]

    GITHUB_EXCLUDE_DIR_NAMES: list[str] = [
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
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




settings = Settings()
