# backend/app/config.py
from __future__ import annotations

import os
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _csv(v: str | None) -> list[str]:
    if not v:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


class Settings(BaseSettings):
    # -----------------------
    # Core runtime
    # -----------------------
    TZ: str = "America/Detroit"
    PORT: int = 8000

    # Used in email links. In dev you can leave it as localhost.
    # In production, set to your real domain (https://goal.yourdomain.com).
    PUBLIC_BASE_URL: str = "http://127.0.0.1:8000"

    ENV: str = "dev"

    # -----------------------
    # DB
    # -----------------------
    DB_URL: str = "sqlite:///./data/app.db"

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

    # -----------------------
    # API auth
    # -----------------------
    API_KEY: str = "change-me"

    # -----------------------
    # --- LLM (OFF by default) ---
    # Keep the old OpenAI fields exactly, because other modules may import them.
    # -----------------------
    LLM_ENABLED: bool = False

    OPENAI_BASE_URL: str = ""
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = ""

    # -----------------------
    # --- Repo task rules ---
    # -----------------------
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

    # -----------------------
    # ✅ NEW (but compatible): include allowlist to stop “999 files” regressions
    #
    # If you want the old behavior (no include filter), set:
    #   GITHUB_INCLUDE_PREFIXES=
    #   GITHUB_INCLUDE_FILES=
    #
    # For OneHaven to get ~137 files, keep defaults:
    #   GITHUB_INCLUDE_PREFIXES=onehaven/
    #   GITHUB_INCLUDE_FILES=README.md,.gitignore
    # -----------------------
    GITHUB_INCLUDE_PREFIXES: list[str] = Field(default_factory=lambda: ["onehaven/"])
    GITHUB_INCLUDE_FILES: list[str] = Field(default_factory=lambda: ["README.md", ".gitignore"])

    # -----------------------
    # GitHub sync exclusions (restored from your old config)
    # Cleaned duplicates but preserved intent.
    # -----------------------
    GITHUB_EXCLUDE_PREFIXES: list[str] = Field(
        default_factory=lambda: [
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
    )

    GITHUB_EXCLUDE_DIR_NAMES: list[str] = Field(
        default_factory=lambda: [
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
    )

    GITHUB_EXCLUDE_EXTENSIONS: list[str] = Field(
        default_factory=lambda: [
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
    )

    # -----------------------
    # Pydantic v2 config
    # -----------------------
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @classmethod
    def _apply_list_overrides(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Allow comma-separated env overrides for list fields.
        This is important for Docker: you can tune filters without rebuilding.
        """
        for key in [
            "GITHUB_INCLUDE_PREFIXES",
            "GITHUB_INCLUDE_FILES",
            "GITHUB_EXCLUDE_PREFIXES",
            "GITHUB_EXCLUDE_DIR_NAMES",
            "GITHUB_EXCLUDE_EXTENSIONS",
        ]:
            v = os.getenv(key)
            if v is not None:
                data[key] = _csv(v)
        return data


# Instantiate settings (and apply env overrides for list fields)
settings = Settings(**Settings._apply_list_overrides({}))
