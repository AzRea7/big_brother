from __future__ import annotations

from sqlalchemy import Engine, inspect, text


def _has_column(engine: Engine, table: str, col: str) -> bool:
    insp = inspect(engine)
    cols = [c["name"] for c in insp.get_columns(table)]
    return col in cols


def _add_column_sqlite(engine: Engine, table: str, col: str, col_ddl: str) -> None:
    # SQLite supports ADD COLUMN (with limits). Good enough for your use case.
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_ddl}"))


def _ensure_table(engine: Engine, ddl: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(ddl))


def ensure_schema(engine: Engine) -> None:
    """
    Lightweight migrations:
    - Adds new Task columns if missing
    - Creates task_dependencies + plan_runs tables if missing
    Works on SQLite without Alembic.
    """
    insp = inspect(engine)
    tables = set(insp.get_table_names())

    # ---- Task columns (Improvement 1/3/4) ----
    if "tasks" in tables:
        # project/tags/link/starter/dod
        if not _has_column(engine, "tasks", "project"):
            _add_column_sqlite(engine, "tasks", "project", "VARCHAR(32)")
        if not _has_column(engine, "tasks", "tags"):
            _add_column_sqlite(engine, "tasks", "tags", "TEXT")
        if not _has_column(engine, "tasks", "link"):
            _add_column_sqlite(engine, "tasks", "link", "TEXT")
        if not _has_column(engine, "tasks", "starter"):
            _add_column_sqlite(engine, "tasks", "starter", "TEXT")
        if not _has_column(engine, "tasks", "dod"):
            _add_column_sqlite(engine, "tasks", "dod", "TEXT")

        # precision scoring
        if not _has_column(engine, "tasks", "impact_score"):
            _add_column_sqlite(engine, "tasks", "impact_score", "INTEGER")
        if not _has_column(engine, "tasks", "confidence"):
            _add_column_sqlite(engine, "tasks", "confidence", "INTEGER")
        if not _has_column(engine, "tasks", "energy"):
            _add_column_sqlite(engine, "tasks", "energy", "VARCHAR(16)")

        # microtasks
        if not _has_column(engine, "tasks", "parent_task_id"):
            _add_column_sqlite(engine, "tasks", "parent_task_id", "INTEGER")

    # ---- Dependencies table (Improvement 2) ----
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

    # ---- Plan runs table (Improvement 5) ----
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
