from __future__ import annotations

from sqlalchemy import Engine, inspect, text


def _has_column(engine: Engine, table: str, col: str) -> bool:
    insp = inspect(engine)
    cols = [c["name"] for c in insp.get_columns(table)]
    return col in cols


def _add_column_sqlite(engine: Engine, table: str, col: str, col_ddl: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_ddl}"))


def _ensure(engine: Engine, ddl: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _is_sqlite(engine: Engine) -> bool:
    try:
        return engine.dialect.name == "sqlite"
    except Exception:
        return False


def _sqlite_has_fts5(engine: Engine) -> bool:
    """
    Detect if SQLite was compiled with FTS5 support.
    If not, we skip creating FTS tables/triggers.
    """
    if not _is_sqlite(engine):
        return False
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("PRAGMA compile_options;")).fetchall()
        opts = {str(r[0]).upper() for r in rows}
        return any("FTS5" in o for o in opts)
    except Exception:
        return False


def ensure_schema(engine: Engine) -> None:
    """
    SQLite-friendly schema upgrades.
    Runs every startup (init_db calls ensure_schema()).

    NOTE:
    - Keep this idempotent (CREATE IF NOT EXISTS / ALTER-if-missing).
    - Avoid executing SQL at import time.
    """
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
        _ensure(
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
        _ensure(
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
        _ensure(
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

    # ---- Repo files ----
    if "repo_files" not in tables:
        _ensure(
            engine,
            """
            CREATE TABLE IF NOT EXISTS repo_files (
              id INTEGER PRIMARY KEY,
              snapshot_id INTEGER NOT NULL,
              path VARCHAR(1024) NOT NULL,
              sha VARCHAR(120),
              size INTEGER,
              content TEXT,
              content_text TEXT,
              content_kind VARCHAR(30) DEFAULT 'skipped',
              skipped BOOLEAN DEFAULT 0,
              is_text BOOLEAN DEFAULT 1,
              skip_reason TEXT,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_files_snapshot ON repo_files(snapshot_id)")
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_files_path ON repo_files(path)")
    else:
        for col, ddl in [
            ("sha", "VARCHAR(120)"),
            ("size", "INTEGER"),
            ("content_kind", "VARCHAR(30) DEFAULT 'skipped'"),
            ("skipped", "BOOLEAN DEFAULT 0"),
            ("is_text", "BOOLEAN DEFAULT 1"),
            ("content_text", "TEXT"),
            ("skip_reason", "TEXT"),
            ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ]:
            if not _has_column(engine, "repo_files", col):
                _add_column_sqlite(engine, "repo_files", col, ddl)

        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_files_snapshot ON repo_files(snapshot_id)")
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_files_path ON repo_files(path)")

    # ---- Repo findings (LLM scan output) ----
    if "repo_findings" not in tables:
        _ensure(
            engine,
            """
            CREATE TABLE IF NOT EXISTS repo_findings (
              id INTEGER PRIMARY KEY,
              snapshot_id INTEGER NOT NULL,
              path VARCHAR(1024) NOT NULL,
              line INTEGER,
              category VARCHAR(48) NOT NULL,
              severity INTEGER DEFAULT 3,
              title VARCHAR(240) NOT NULL,
              evidence TEXT,
              recommendation TEXT,
              acceptance TEXT,
              fingerprint VARCHAR(64) NOT NULL,
              is_resolved BOOLEAN DEFAULT 0,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(snapshot_id, fingerprint)
            )
            """,
        )
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_findings_snapshot ON repo_findings(snapshot_id)")
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_findings_sev ON repo_findings(severity)")
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_findings_resolved ON repo_findings(is_resolved)")
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_findings_path ON repo_findings(path)")
    else:
        for col, ddl in [
            ("path", "VARCHAR(1024)"),
            ("line", "INTEGER"),
            ("category", "VARCHAR(48)"),
            ("severity", "INTEGER DEFAULT 3"),
            ("title", "VARCHAR(240)"),
            ("evidence", "TEXT"),
            ("recommendation", "TEXT"),
            ("acceptance", "TEXT"),
            ("fingerprint", "VARCHAR(64)"),
            ("is_resolved", "BOOLEAN DEFAULT 0"),
            ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ]:
            if not _has_column(engine, "repo_findings", col):
                _add_column_sqlite(engine, "repo_findings", col, ddl)

        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_findings_snapshot ON repo_findings(snapshot_id)")
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_findings_sev ON repo_findings(severity)")
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_findings_resolved ON repo_findings(is_resolved)")
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_findings_path ON repo_findings(path)")

    # -----------------------
    # Level 2 RAG: repo_chunks
    # -----------------------
    if "repo_chunks" not in tables:
        _ensure(
            engine,
            """
            CREATE TABLE IF NOT EXISTS repo_chunks (
              id INTEGER PRIMARY KEY,
              snapshot_id INTEGER NOT NULL,
              path VARCHAR(1024) NOT NULL,
              start_line INTEGER NOT NULL,
              end_line INTEGER NOT NULL,
              chunk_text TEXT NOT NULL,
              symbols_json TEXT,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(snapshot_id, path, start_line, end_line)
            )
            """,
        )
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_chunks_snapshot ON repo_chunks(snapshot_id)")
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_chunks_path ON repo_chunks(path)")
    else:
        for col, ddl in [
            ("path", "VARCHAR(1024)"),
            ("start_line", "INTEGER"),
            ("end_line", "INTEGER"),
            ("chunk_text", "TEXT"),
            ("symbols_json", "TEXT"),
            ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ]:
            if not _has_column(engine, "repo_chunks", col):
                _add_column_sqlite(engine, "repo_chunks", col, ddl)

        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_chunks_snapshot ON repo_chunks(snapshot_id)")
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_chunks_path ON repo_chunks(path)")

    # Optional: FTS5 for chunk retrieval
    if _sqlite_has_fts5(engine):
        _ensure(
            engine,
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS repo_chunks_fts
            USING fts5(chunk_text, path, symbols_json, content='repo_chunks', content_rowid='id');
            """,
        )

        _ensure(
            engine,
            """
            CREATE TRIGGER IF NOT EXISTS repo_chunks_ai AFTER INSERT ON repo_chunks BEGIN
              INSERT INTO repo_chunks_fts(rowid, chunk_text, path, symbols_json)
              VALUES (new.id, new.chunk_text, new.path, new.symbols_json);
            END;
            """,
        )
        _ensure(
            engine,
            """
            CREATE TRIGGER IF NOT EXISTS repo_chunks_ad AFTER DELETE ON repo_chunks BEGIN
              INSERT INTO repo_chunks_fts(repo_chunks_fts, rowid, chunk_text, path, symbols_json)
              VALUES('delete', old.id, old.chunk_text, old.path, old.symbols_json);
            END;
            """,
        )
        _ensure(
            engine,
            """
            CREATE TRIGGER IF NOT EXISTS repo_chunks_au AFTER UPDATE ON repo_chunks BEGIN
              INSERT INTO repo_chunks_fts(repo_chunks_fts, rowid, chunk_text, path, symbols_json)
              VALUES('delete', old.id, old.chunk_text, old.path, old.symbols_json);
              INSERT INTO repo_chunks_fts(rowid, chunk_text, path, symbols_json)
              VALUES (new.id, new.chunk_text, new.path, new.symbols_json);
            END;
            """,
        )

    # -----------------------
    # Level 2.5: repo_chunk_embeddings (cosine rerank)
    # -----------------------
    if "repo_chunk_embeddings" not in tables:
        _ensure(
            engine,
            """
            CREATE TABLE IF NOT EXISTS repo_chunk_embeddings (
              id INTEGER PRIMARY KEY,
              chunk_id INTEGER NOT NULL,
              model VARCHAR(120) NOT NULL,
              vector_json TEXT NOT NULL,
              embedding_norm REAL,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(chunk_id, model)
            )
            """,
        )
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_chunk_embeddings_chunk ON repo_chunk_embeddings(chunk_id)")
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_chunk_embeddings_model ON repo_chunk_embeddings(model)")
    else:
        for col, ddl in [
            ("chunk_id", "INTEGER"),
            ("model", "VARCHAR(120)"),
            ("vector_json", "TEXT"),
            ("embedding_norm", "REAL"),
            ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ]:
            if not _has_column(engine, "repo_chunk_embeddings", col):
                _add_column_sqlite(engine, "repo_chunk_embeddings", col, ddl)

        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_chunk_embeddings_chunk ON repo_chunk_embeddings(chunk_id)")
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_chunk_embeddings_model ON repo_chunk_embeddings(model)")

    # -----------------------
    # Level 3: Patch runs + Pull requests
    # -----------------------

    # ---- repo_patch_runs ----
    if "repo_patch_runs" not in tables:
        _ensure(
            engine,
            """
            CREATE TABLE IF NOT EXISTS repo_patch_runs (
              id INTEGER PRIMARY KEY,
              snapshot_id INTEGER NOT NULL,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

              patch_text TEXT NOT NULL,

              -- NEW: traceability + PR metadata
              finding_id INTEGER,
              objective TEXT,
              suggested_pr_title VARCHAR(300),
              suggested_pr_body TEXT,

              files_changed INTEGER,
              lines_changed INTEGER,
              file_paths_json TEXT,

              valid BOOLEAN DEFAULT 0,
              validation_error TEXT,

              applied BOOLEAN DEFAULT 0,
              apply_error TEXT,

              tests_ran BOOLEAN DEFAULT 0,
              tests_ok BOOLEAN DEFAULT 0,
              test_output TEXT,

              sandbox_path VARCHAR(800),
              diff_output TEXT
            )
            """,
        )
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_patch_runs_snapshot ON repo_patch_runs(snapshot_id)")
    else:
        for col, ddl in [
            ("snapshot_id", "INTEGER"),
            ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
            ("patch_text", "TEXT"),

            # NEW columns
            ("finding_id", "INTEGER"),
            ("objective", "TEXT"),
            ("suggested_pr_title", "VARCHAR(300)"),
            ("suggested_pr_body", "TEXT"),

            ("files_changed", "INTEGER"),
            ("lines_changed", "INTEGER"),
            ("file_paths_json", "TEXT"),

            ("valid", "BOOLEAN DEFAULT 0"),
            ("validation_error", "TEXT"),

            ("applied", "BOOLEAN DEFAULT 0"),
            ("apply_error", "TEXT"),

            ("tests_ran", "BOOLEAN DEFAULT 0"),
            ("tests_ok", "BOOLEAN DEFAULT 0"),
            ("test_output", "TEXT"),

            ("sandbox_path", "VARCHAR(800)"),
            ("diff_output", "TEXT"),
        ]:
            if not _has_column(engine, "repo_patch_runs", col):
                _add_column_sqlite(engine, "repo_patch_runs", col, ddl)

        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_patch_runs_snapshot ON repo_patch_runs(snapshot_id)")

    # ---- repo_pull_requests ----
    if "repo_pull_requests" not in tables:
        _ensure(
            engine,
            """
            CREATE TABLE IF NOT EXISTS repo_pull_requests (
              id INTEGER PRIMARY KEY,
              snapshot_id INTEGER NOT NULL,
              patch_run_id INTEGER,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

              repo VARCHAR(200) NOT NULL,
              branch VARCHAR(100) NOT NULL,

              pr_number INTEGER,
              pr_url VARCHAR(500),

              title VARCHAR(300) NOT NULL,
              body TEXT
            )
            """,
        )
        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_pull_requests_snapshot ON repo_pull_requests(snapshot_id)")
    else:
        for col, ddl in [
            ("snapshot_id", "INTEGER"),
            ("patch_run_id", "INTEGER"),
            ("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
            ("repo", "VARCHAR(200)"),
            ("branch", "VARCHAR(100)"),
            ("pr_number", "INTEGER"),
            ("pr_url", "VARCHAR(500)"),
            ("title", "VARCHAR(300)"),
            ("body", "TEXT"),
        ]:
            if not _has_column(engine, "repo_pull_requests", col):
                _add_column_sqlite(engine, "repo_pull_requests", col, ddl)

        _ensure(engine, "CREATE INDEX IF NOT EXISTS idx_repo_pull_requests_snapshot ON repo_pull_requests(snapshot_id)")
