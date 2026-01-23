# backend/app/config.py
from __future__ import annotations

from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # -----------------------
    # Core runtime
    # -----------------------
    TZ: str = "America/Detroit"
    PORT: int = 8000

    # Used in email links. In dev you can leave it as localhost.
    # In production, set to your real domain (https://goal.yourdomain.com).
    PUBLIC_BASE_URL: str = "http://127.0.0.1:8000"

    DB_URL: str = "sqlite:///./data/app.db"

    # Auth
    API_KEY: str = "9f4a6c8b1a2e4f7c8e9d0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b"
    ENV: str = "dev"  # dev | prod | ci
    DISABLE_DEBUG_IN_PROD: bool = True

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

    # -----------------------
    # LLM
    # -----------------------
    LLM_ENABLED: bool = True

    # OpenAI-compatible config
    # IMPORTANT: base URL should NOT include /v1
    # Example: https://api.openai.com
    OPENAI_BASE_URL: str = ""
    OPENAI_API_KEY: str = "REDACTED_OPENAI_KEY"
    OPENAI_MODEL: str = "openai"

    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_TIMEOUT_S: float = 60.0

    # Optional safety / behavior toggles
    HAVEN_REPO_ONLY: bool = True
    LLM_READ_TIMEOUT_S: float = 660.0

    # -----------------------
    # Preferred: local path scan (no tokens)
    # If set, github_sync should read files from disk instead of GitHub API.
    # -----------------------
    REPO_LOCAL_PATH: Optional[str] = None
    REPO_LOCAL_GIT_SHA: Optional[str] = None

    # -----------------------
    # --- GitHub repo sync (Level 1) ---
    # -----------------------
    GITHUB_REPO: str = "AzRea7/OneHaven"
    GITHUB_BRANCH: str = "main"
    GITHUB_TOKEN: Optional[str] = None

    # Safety: don’t store huge/binary files in your DB.
    GITHUB_MAX_FILE_BYTES: int = 150_000  # 150 KB
    GITHUB_MAX_FILES_PER_SYNC: int = 5_000

    # Timeouts for GitHub API calls
    GITHUB_CONNECT_TIMEOUT_S: float = 10.0
    GITHUB_READ_TIMEOUT_S: float = 60.0

    # -----------------------
    # Task generation shaping
    # -----------------------
    REPO_TASK_MAX_FILES: int = 18
    REPO_TASK_EXCERPT_CHARS: int = 800
    REPO_TASK_MAX_TOTAL_CHARS: int = 12_000
    REPO_TASK_COUNT: int = 3

    STATIC_SCAN_MAX_SECONDS: int = 120

    # -----------------------
    # Level 2 RAG (chunking + retrieval)
    # -----------------------
    REPO_CHUNK_LINES: int = 120
    REPO_CHUNK_OVERLAP: int = 25
    REPO_CHUNK_MAX_CHARS: int = 9_000

    REPO_RAG_TOP_K: int = 8
    # Used by repo_rag.py to cap incoming top_k
    RAG_MAX_TOP_K: int = 20

    FINDING_RAG_MAX_CHARS: int = 6_000

    # Optional comma-separated “query seed” phrases to bias retrieval.
    REPO_RAG_QUERY_SEEDS: str = ""

    # Embeddings
    EMBEDDINGS_ENABLED: bool = True
    EMBEDDINGS_PROVIDER: str = "off"  # off | openai
    EMBEDDINGS_MODEL: str = "text-embedding-3-small"

    # If true, chunk build will attempt to embed chunks too.
    EMBED_CHUNKS_ON_BUILD: bool = True

    # -----------------------
    # --- PR workflow (Level 3) ---
    # -----------------------
    ENABLE_PR_WORKFLOW: bool = True

    # Paths allowed to be modified by PR workflow (relative to repo root)
    PR_ALLOWLIST_DIRS: List[str] = Field(default_factory=lambda: ["backend/", "onehaven/backend/"])

    # Hard caps so you don’t nuke the repo by accident
    PR_MAX_FILES_CHANGED: int = 8
    PR_MAX_LINES_CHANGED: int = 400

    # Patch application/test settings
    PR_TEST_CMD: str = "pytest -q"
    PR_TIMEOUT_SECONDS: int = 900

    # -----------------------
    # GitHub sync exclusions
    # -----------------------
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


settings = Settings()
