# backend/app/services/repo_rag.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoChunk, RepoFile


# -----------------------
# Data model returned by retrieval
# -----------------------
@dataclass(frozen=True)
class ChunkHit:
    id: int
    path: str
    start_line: int
    end_line: int
    score: float | None
    chunk_text: str


# -----------------------
# Chunking helpers
# -----------------------
_SYMBOL_RE = re.compile(r"^\s*(class|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.MULTILINE)


def _extract_symbols(chunk_text: str) -> list[str]:
    out: list[str] = []
    for m in _SYMBOL_RE.finditer(chunk_text):
        kind = m.group(1)
        name = m.group(2)
        out.append(f"{kind}:{name}")
    return out[:50]


def _line_chunks(lines: list[str], chunk_lines: int, overlap: int) -> Iterable[tuple[int, int, str]]:
    """
    Yields (start_line, end_line, chunk_text).
    Lines are 1-indexed for line numbers.
    """
    if chunk_lines <= 0:
        chunk_lines = 120
    if overlap < 0:
        overlap = 0
    step = max(1, chunk_lines - overlap)

    n = len(lines)
    start = 0
    while start < n:
        end = min(n, start + chunk_lines)
        chunk = "".join(lines[start:end])

        # 1-indexed inclusive
        start_line = start + 1
        end_line = end

        yield start_line, end_line, chunk
        if end == n:
            break
        start += step


def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n…(truncated)…\n"


def chunk_snapshot(db: Session, snapshot_id: int, force: bool = False) -> dict:
    """
    Build repo_chunks rows for a snapshot using RepoFile.content_text.
    FTS5 stays in sync via triggers created in migrations.ensure_schema().
    """
    if force:
        db.query(RepoChunk).filter(RepoChunk.snapshot_id == snapshot_id).delete()
        db.commit()

    existing = db.query(RepoChunk).filter(RepoChunk.snapshot_id == snapshot_id).limit(1).first()
    if existing and not force:
        return {"snapshot_id": snapshot_id, "created": 0, "skipped": 0, "note": "chunks already exist"}

    files: list[RepoFile] = (
        db.query(RepoFile)
        .filter(RepoFile.snapshot_id == snapshot_id)
        .filter(RepoFile.content_text.isnot(None))
        .filter(RepoFile.skipped == False)  # noqa: E712
        .all()
    )

    created = 0
    skipped = 0

    for rf in files:
        if not rf.content_text:
            skipped += 1
            continue

        lines = rf.content_text.splitlines(keepends=True)
        for start_line, end_line, chunk_text in _line_chunks(lines, settings.REPO_CHUNK_LINES, settings.REPO_CHUNK_OVERLAP):
            chunk_text = _truncate(chunk_text, settings.REPO_CHUNK_MAX_CHARS)
            symbols = _extract_symbols(chunk_text)
            symbols_json = json.dumps(symbols) if symbols else None

            db.add(
                RepoChunk(
                    snapshot_id=snapshot_id,
                    path=rf.path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_text=chunk_text,
                    symbols_json=symbols_json,
                )
            )
            created += 1

    db.commit()
    return {"snapshot_id": snapshot_id, "created": created, "skipped": skipped}


# -----------------------
# Retrieval
# -----------------------
def _sqlite_fts_available(db: Session) -> bool:
    try:
        db.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='repo_chunks_fts'")).fetchone()
        return True
    except Exception:
        return False


def search_chunks(
    db: Session,
    snapshot_id: int,
    query: str,
    top_k: int = 12,
    prefer_path: str | None = None,
) -> List[ChunkHit]:
    """
    Retrieve top-k chunks for snapshot_id using SQLite FTS5 when available.
    If FTS table is absent (or non-sqlite), falls back to LIKE query.
    """
    query = (query or "").strip()
    if not query:
        return []

    top_k = max(1, min(50, top_k))

    # ---- FTS path (SQLite only, when available) ----
    if settings.DB_URL.startswith("sqlite") and _sqlite_fts_available(db):
        # Join repo_chunks_fts(rowid) -> repo_chunks(id) and filter snapshot_id
        # bm25() is available in FTS5; lower is "better". We'll keep it as score.
        sql = text(
            """
            SELECT
              c.id AS id,
              c.path AS path,
              c.start_line AS start_line,
              c.end_line AS end_line,
              bm25(repo_chunks_fts) AS score,
              c.chunk_text AS chunk_text
            FROM repo_chunks_fts
            JOIN repo_chunks c ON c.id = repo_chunks_fts.rowid
            WHERE c.snapshot_id = :snapshot_id
              AND repo_chunks_fts MATCH :q
            ORDER BY score ASC
            LIMIT :k
            """
        )
        rows = db.execute(sql, {"snapshot_id": snapshot_id, "q": query, "k": top_k}).fetchall()
        hits = [
            ChunkHit(
                id=int(r[0]),
                path=str(r[1]),
                start_line=int(r[2]),
                end_line=int(r[3]),
                score=float(r[4]) if r[4] is not None else None,
                chunk_text=str(r[5]),
            )
            for r in rows
        ]
    else:
        # ---- Fallback: naive LIKE ----
        like = f"%{query}%"
        rows = (
            db.query(RepoChunk)
            .filter(RepoChunk.snapshot_id == snapshot_id)
            .filter(RepoChunk.chunk_text.ilike(like))
            .limit(top_k)
            .all()
        )
        hits = [
            ChunkHit(
                id=c.id,
                path=c.path,
                start_line=c.start_line,
                end_line=c.end_line,
                score=None,
                chunk_text=c.chunk_text,
            )
            for c in rows
        ]

    # Optional: prefer chunks from a specific path (finding.path)
    if prefer_path:
        prefer_path = prefer_path.strip()
        hits.sort(key=lambda h: (0 if h.path == prefer_path else 1, h.score if h.score is not None else 999999.0))

    return hits
