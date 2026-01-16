# backend/app/migrations.py
from __future__ import annotations

from sqlalchemy import Engine, inspect, text


def _has_column(engine: Engine, table: str, col: str) -> bool:
    insp = inspect(engine)
    cols = [c["name"] for c in insp.get_columns(table)]
    return col in cols


def _add_column_sqlite(engine: Engine, table: str, col: str, col_ddl: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_ddl}"))


def _ensure_table(engine: Engine, ddl: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(ddl))


def ensure_schema(engine: Engine) -> None:
    insp = inspect(engine)
    tables = set(insp.get_table_names())

    # ---- Goal.project ----
    if "goals" in tables and not _has_column(engine, "goals", "project"):
        _add_column_sqlite(engine, "goals", "project", "VARCHAR(32)")

    # ---- Task fields ----
    if "tasks" in tables:
        for col, ddl in [
            ("project", "VARCHAR(32)"),
            ("tags", "TEXT"),
            ("link", "TEXT"),
            ("starter", "TEXT"),
            ("dod", "TEXT"),
            ("impact_score", "INTEGER"),
            ("confidence", "INTEGER"),
            ("energy", "VARCHAR(16)"),
            ("parent_task_id", "INTEGER"),
        ]:
            if not _has_column(engine, "tasks", col):
                _add_column_sqlite(engine, "tasks", col, ddl)

    # ---- Dependencies ----
    if "task_dependencies" not in tables:
        _ensure_table(
            engine,
            """
            CREATE TABLE IF NOT EXISTS task_dependencies (
              id INTEGER PRIMARY KEY,
              task_id INTEGER NOT NULL,
              depends_on_task_id INTEGER NOT NULL,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(task_id, depends_on_task_id)
            )
            """,
        )

    # ---- Plan runs ----
    if "plan_runs" not in tables:
        _ensure_table(
            engine,
            """
            CREATE TABLE IF NOT EXISTS plan_runs (
              id INTEGER PRIMARY KEY,
              project VARCHAR(32),
              generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              top_task_ids TEXT,
              content TEXT
            )
            """,
        )

    # ---- Repo snapshots ----
    if "repo_snapshots" not in tables:
        _ensure_table(
            engine,
            """
            CREATE TABLE IF NOT EXISTS repo_snapshots (
              id INTEGER PRIMARY KEY,
              repo VARCHAR(200) NOT NULL,
              branch VARCHAR(80) NOT NULL,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              commit_sha VARCHAR(80),
              file_count INTEGER DEFAULT 0,
              stored_content_files INTEGER DEFAULT 0,
              warnings_json TEXT
            )
            """,
        )

    if "repo_files" not in tables:
        _ensure_table(
            engine,
            """
            CREATE TABLE IF NOT EXISTS repo_files (
              id INTEGER PRIMARY KEY,
              snapshot_id INTEGER NOT NULL,
              path VARCHAR(600) NOT NULL,
              sha VARCHAR(80),
              size INTEGER,
              content TEXT,
              content_kind VARCHAR(16) DEFAULT 'skipped',
              skipped BOOLEAN DEFAULT 0,
              is_text BOOLEAN DEFAULT 1,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        _ensure_table(engine, "CREATE INDEX IF NOT EXISTS idx_repo_files_snapshot ON repo_files(snapshot_id)")
        _ensure_table(engine, "CREATE INDEX IF NOT EXISTS idx_repo_files_path ON repo_files(path)")
    else:
        # upgrade existing repo_files table
        for col, ddl in [
            ("skipped", "BOOLEAN DEFAULT 0"),
            ("is_text", "BOOLEAN DEFAULT 1"),
        ]:
            if not _has_column(engine, "repo_files", col):
                _add_column_sqlite(engine, "repo_files", col, ddl)
