# backend/app/config.py
from __future__ import annotations

import os
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -----------------------
    # Core runtime
    # -----------------------
    TZ: str = "America/Detroit"
    PORT: int = 8000

    # Used in email links. In dev you can leave it as localhost.
    # In production, set to your real domain (https://goal.yourdomain.com).
    PUBLIC_BASE_URL: str = "http://127.0.0.1:8000"

    DB_URL: str = "sqlite:///./data/app.db"

    API_KEY: str = "change-me"
    # If true, /debug endpoints are disabled unless ENV=dev
    DISABLE_DEBUG_IN_PROD: bool = True
    ENV: str = "dev"  # dev | prod | ci

    ENABLE_METRICS: bool = True

    # -----------------------
    # Scheduler controls
    # -----------------------
    MORNING_PLAN_HOUR: int = 8
    MORNING_PLAN_MINUTE: int = 0

    MIDDAY_NUDGE_HOUR: int = 13
    MIDDAY_NUDGE_MINUTE: int = 0

    # Repo pipeline schedule
    REPO_SYNC_HOUR: int = 6
    REPO_SYNC_MINUTE: int = 0
    REPO_SYNC_PROJECT: str = "haven"

    # What the scheduler sends
    DAILY_PLAN_MODE: str = "single"  # single | split
    DAILY_PLAN_PROJECT: str = "onestream"  # haven | onestream | (ignored in split)

    LLM_ENABLED: bool = True

    OPENAI_BASE_URL: str = ""
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = ""
    HAVEN_REPO_ONLY: bool = True

    # Preferred: local path scan (no tokens)
    # If set, github_sync should read files from disk instead of GitHub API.
    REPO_LOCAL_PATH: str | None = None
    REPO_LOCAL_GIT_SHA: str | None = None

    # -----------------------
    # --- GitHub repo sync (Level 1) ---
    # -----------------------
    GITHUB_REPO: str = "AzRea7/OneHaven"
    GITHUB_BRANCH: str = "main"
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


    REPO_TASK_MAX_FILES: int = 18
    REPO_TASK_EXCERPT_CHARS: int = 800

    # Output
    REPO_TASK_COUNT: int = 3
    # -----------------------
    # GitHub sync exclusions (restored from your old config)
    # Cleaned duplicates but preserved intent.
    # -----------------------

    REPO_TASK_MAX_TOTAL_CHARS: int = 12_000
    LLM_READ_TIMEOUT_S: float = 660.0

    STATIC_SCAN_MAX_SECONDS: int = 120

    # -----------------------
    # Level 2 RAG (chunking + retrieval)
    # -----------------------
    REPO_CHUNK_LINES: int = 120
    REPO_CHUNK_OVERLAP: int = 25
    REPO_CHUNK_MAX_CHARS: int = 9_000

    REPO_RAG_TOP_K: int = 16

    # Optional comma-separated “query seed” phrases to bias retrieval.
    # Example: "auth,db session,fastapi router,github sync,repo pipeline"
    REPO_RAG_QUERY_SEEDS: str = ""


    GITHUB_EXCLUDE_PREFIXES: list[str] = [
            # OneHaven-specific
            "onehaven/node_modules/",
            "onehaven/.venv/",
            "onehaven/venv/",
            "onehaven/dist/",
            "onehaven/build/",
            "onehaven/.pytest_cache/",
            "onehaven/__pycache__/",

            # Generic
            "node_modules/",
            ".venv/",
            "venv/",
            "__pycache__/",
            ".git/",
            ".github/",
            "dist/",
            "build/",
            "coverage/",
            ".idea/",
            ".vscode/",

            # Backend-specific paths (if repo contains nested backend/)
            "backend/.venv/",
            "backend/__pycache__/",
            "backend/.pytest_cache/",
            "backend/.mypy_cache/",
            "backend/.ruff_cache/",
            "backend/.git/",
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
            ".git",
            ".idea",
            ".vscode",
            "dist",
            "build",
            "coverage",
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