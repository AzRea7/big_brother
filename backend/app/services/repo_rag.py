# backend/app/services/repo_rag.py
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoChunk, RepoFile


_SYMBOL_RE = re.compile(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:64]


def _extract_symbols(chunk_text: str) -> str | None:
    hits = []
    for m in _SYMBOL_RE.finditer(chunk_text):
        hits.append(f"{m.group(1)} {m.group(2)}")
    if not hits:
        return None
    # cap to avoid huge rows
    return "\n".join(hits[:50])


def _is_sqlite(db: Session) -> bool:
    try:
        return db.bind is not None and db.bind.dialect.name == "sqlite"
    except Exception:
        return False


def _fts_insert(db: Session, *, snapshot_id: int, path: str, start_line: int, end_line: int, chunk_text_val: str) -> None:
    """
    Insert into SQLite FTS table if present.
    Safe no-op if table doesn't exist / isn't SQLite.
    """
    if not _is_sqlite(db):
        return

    # If FTS table doesn't exist, this will raise; that's fine and visible.
    db.execute(
        text(
            """
            INSERT INTO repo_chunks_fts (chunk_text, path, snapshot_id, start_line, end_line)
            VALUES (:chunk_text, :path, :snapshot_id, :start_line, :end_line)
            """
        ),
        {
            "chunk_text": chunk_text_val,
            "path": path,
            "snapshot_id": snapshot_id,
            "start_line": start_line,
            "end_line": end_line,
        },
    )


def chunk_snapshot(db: Session, snapshot_id: int, *, force: bool = False) -> dict[str, Any]:
    """
    Build RepoChunk rows for a snapshot.
    If not forced, does nothing if chunks already exist.

    Returns:
      {snapshot_id, created, skipped, total_chunks}
    """
    existing = db.query(RepoChunk).filter(RepoChunk.snapshot_id == snapshot_id).count()
    if existing > 0 and not force:
        return {"snapshot_id": snapshot_id, "created": 0, "skipped": existing, "total_chunks": existing}

    if force:
        # delete existing chunks
        db.query(RepoChunk).filter(RepoChunk.snapshot_id == snapshot_id).delete(synchronize_session=False)
        if _is_sqlite(db):
            # clear FTS rows for that snapshot
            db.execute(text("DELETE FROM repo_chunks_fts WHERE snapshot_id = :sid"), {"sid": snapshot_id})
        db.commit()

    files = (
        db.query(RepoFile)
        .filter(RepoFile.snapshot_id == snapshot_id)
        .filter(RepoFile.content_text.isnot(None))
        .order_by(RepoFile.path.asc())
        .all()
    )

    created = 0

    for rf in files:
        content = rf.content_text or ""
        if not content.strip():
            continue

        lines = content.splitlines()
        n = len(lines)
        if n == 0:
            continue

        step = max(1, settings.REPO_CHUNK_LINES - settings.REPO_CHUNK_OVERLAP)

        start = 0
        while start < n:
            end = min(n, start + settings.REPO_CHUNK_LINES)

            span_lines = lines[start:end]
            chunk_text_val = "\n".join(span_lines)
            if len(chunk_text_val) > settings.REPO_CHUNK_MAX_CHARS:
                chunk_text_val = chunk_text_val[: settings.REPO_CHUNK_MAX_CHARS]

            # 1-indexed lines
            start_line = start + 1
            end_line = end

            fp = _sha(f"{snapshot_id}|{rf.path}|{start_line}|{end_line}|{chunk_text_val[:200]}")

            exists = (
                db.query(RepoChunk)
                .filter(
                    RepoChunk.snapshot_id == snapshot_id,
                    RepoChunk.path == rf.path,
                    RepoChunk.start_line == start_line,
                    RepoChunk.end_line == end_line,
                )
                .first()
            )
            if exists:
                start += step
                continue

            sym = _extract_symbols(chunk_text_val)

            db.add(
                RepoChunk(
                    snapshot_id=snapshot_id,
                    path=rf.path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_text=chunk_text_val,
                    symbols=sym,
                    fingerprint=fp,
                    created_at=datetime.utcnow(),
                )
            )

            # Insert into FTS (SQLite)
            _fts_insert(
                db,
                snapshot_id=snapshot_id,
                path=rf.path,
                start_line=start_line,
                end_line=end_line,
                chunk_text_val=chunk_text_val,
            )

            created += 1
            start += step

    db.commit()

    total = db.query(RepoChunk).filter(RepoChunk.snapshot_id == snapshot_id).count()
    return {"snapshot_id": snapshot_id, "created": created, "skipped": max(0, total - created), "total_chunks": total}


def _normalize_fts_query(q: str) -> str:
    """
    SQLite FTS query sanitizer:
    - strip weird characters
    - keep it simple: words joined by spaces
    """
    q = (q or "").strip()
    q = re.sub(r"[^A-Za-z0-9_\-./\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def search_chunks(db: Session, snapshot_id: int, query: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Retrieve top-k relevant chunks.
    Uses SQLite FTS if available; otherwise falls back to LIKE scanning.
    """
    k = top_k or settings.REPO_RAG_TOP_K
    q = (query or "").strip()
    if not q:
        return []

    seeds = [s.strip() for s in (settings.REPO_RAG_QUERY_SEEDS or "").split(",") if s.strip()]
    if seeds:
        q = q + " " + " ".join(seeds)

    if _is_sqlite(db):
        q2 = _normalize_fts_query(q)
        if q2:
            rows = db.execute(
                text(
                    """
                    SELECT
                      path,
                      snapshot_id,
                      start_line,
                      end_line,
                      snippet(repo_chunks_fts, 0, '[', ']', '…', 12) AS snippet
                    FROM repo_chunks_fts
                    WHERE repo_chunks_fts MATCH :q
                      AND snapshot_id = :sid
                    LIMIT :k
                    """
                ),
                {"q": q2, "sid": snapshot_id, "k": int(k)},
            ).mappings().all()

            # we return snippet for preview; caller can load full text via RepoChunk
            out: list[dict[str, Any]] = []
            for r in rows:
                out.append(
                    {
                        "path": r["path"],
                        "snapshot_id": int(r["snapshot_id"]),
                        "start_line": int(r["start_line"]),
                        "end_line": int(r["end_line"]),
                        "snippet": r["snippet"],
                    }
                )
            return out

    # Fallback: LIKE search across RepoChunk rows (slower but works everywhere)
    like = f"%{q}%"
    rows2 = (
        db.query(RepoChunk)
        .filter(RepoChunk.snapshot_id == snapshot_id)
        .filter(RepoChunk.chunk_text.ilike(like))  # type: ignore[attr-defined]
        .limit(int(k))
        .all()
    )

    return [
        {
            "path": r.path,
            "snapshot_id": r.snapshot_id,
            "start_line": r.start_line,
            "end_line": r.end_line,
            "snippet": (r.chunk_text[:400] + "…") if len(r.chunk_text) > 400 else r.chunk_text,
        }
        for r in rows2
    ]


def load_chunk_text(
    db: Session, snapshot_id: int, path: str, start_line: int, end_line: int
) -> str | None:
    row = (
        db.query(RepoChunk)
        .filter(
            RepoChunk.snapshot_id == snapshot_id,
            RepoChunk.path == path,
            RepoChunk.start_line == start_line,
            RepoChunk.end_line == end_line,
        )
        .first()
    )
    return row.chunk_text if row else None
