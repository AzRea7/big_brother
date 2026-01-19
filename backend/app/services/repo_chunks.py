# backend/app/services/repo_chunks.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text, select, func
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoFile, RepoChunk


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_text(t: str) -> str:
    return t.replace("\r\n", "\n").replace("\r", "\n")


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: int
    path: str
    start_line: int
    end_line: int
    snippet: str
    rank: float | None = None


def build_chunks_for_snapshot(db: Session, snapshot_id: int) -> dict[str, Any]:
    """
    Build/refresh RepoChunk rows from RepoFile content for a snapshot.
    Uses fingerprints to dedupe, so you can safely rebuild often.
    """
    files = db.scalars(select(RepoFile).where(RepoFile.snapshot_id == snapshot_id)).all()

    created = 0
    skipped = 0
    deleted = 0

    # Build a set of desired fingerprints so we can GC old chunks later
    desired_fps: set[str] = set()

    line_window = max(30, int(settings.repo_chunk_lines))
    overlap = max(0, min(line_window - 1, int(settings.repo_chunk_overlap)))
    max_chars = max(2000, int(settings.repo_chunk_max_chars))

    # Existing fingerprints (avoid re-insert)
    existing = db.scalars(select(RepoChunk.fingerprint).where(RepoChunk.snapshot_id == snapshot_id)).all()
    existing_fps = set(existing)

    for rf in files:
        if rf.content_kind != "text":
            continue
        raw = (getattr(rf, "content_text", None) or rf.content or "")
        if not raw.strip():
            continue

        txt = _normalize_text(raw)
        lines = txt.split("\n")

        # Sliding window over lines
        start = 1  # 1-indexed lines
        while start <= len(lines):
            end = min(len(lines), start + line_window - 1)

            chunk_lines = lines[start - 1 : end]
            chunk_text = "\n".join(chunk_lines).strip()
            if not chunk_text:
                start = end + 1
                continue

            if len(chunk_text) > max_chars:
                # hard cap for DB + prompt budgets
                chunk_text = chunk_text[:max_chars]

            fp = _sha1(f"{snapshot_id}|{rf.path}|{start}|{end}|{_sha1(chunk_text)}")
            desired_fps.add(fp)

            if fp in existing_fps:
                skipped += 1
            else:
                db.add(
                    RepoChunk(
                        snapshot_id=snapshot_id,
                        path=rf.path,
                        start_line=start,
                        end_line=end,
                        symbol=None,
                        text=chunk_text,
                        fingerprint=fp,
                    )
                )
                existing_fps.add(fp)
                created += 1

            if end == len(lines):
                break
            start = end - overlap + 1

    db.commit()

    # Garbage-collect chunks that no longer correspond to current file content
    # (Optional but prevents growth if files shrink)
    stale = db.scalars(
        select(RepoChunk).where(RepoChunk.snapshot_id == snapshot_id).where(RepoChunk.fingerprint.not_in(desired_fps))
    ).all()
    for ch in stale:
        db.delete(ch)
        deleted += 1
    if deleted:
        db.commit()

    total = db.scalar(select(func.count(RepoChunk.id)).where(RepoChunk.snapshot_id == snapshot_id)) or 0
    return {"snapshot_id": snapshot_id, "created": created, "skipped": skipped, "deleted": deleted, "total_chunks": int(total)}


def search_chunks(db: Session, snapshot_id: int, q: str, limit: int | None = None) -> list[ChunkHit]:
    """
    Retrieve top chunks using SQLite FTS.
    If you move to embeddings later, this becomes your fallback.
    """
    q = (q or "").strip()
    if not q:
        return []

    top_k = int(limit or settings.repo_rag_top_k or 12)
    top_k = max(1, min(50, top_k))

    if db.bind is None or db.bind.url.get_backend_name() != "sqlite":
        # Fallback: naive LIKE on the chunk table
        rows = db.execute(
            select(RepoChunk)
            .where(RepoChunk.snapshot_id == snapshot_id)
            .where(RepoChunk.text.ilike(f"%{q}%"))
            .limit(top_k)
        ).scalars().all()
        return [
            ChunkHit(
                chunk_id=r.id,
                path=r.path,
                start_line=r.start_line,
                end_line=r.end_line,
                snippet=r.text[:700],
                rank=None,
            )
            for r in rows
        ]

    # SQLite FTS query
    sql = text(
        """
        SELECT c.id, c.path, c.start_line, c.end_line, c.text,
               bm25(repo_chunks_fts) AS rank
        FROM repo_chunks_fts
        JOIN repo_chunks c ON c.id = repo_chunks_fts.rowid
        WHERE c.snapshot_id = :sid
          AND repo_chunks_fts MATCH :q
        ORDER BY rank ASC
        LIMIT :k;
        """
    )

    rows = db.execute(sql, {"sid": snapshot_id, "q": q, "k": top_k}).fetchall()

    hits: list[ChunkHit] = []
    for r in rows:
        txt = r[4] or ""
        hits.append(
            ChunkHit(
                chunk_id=int(r[0]),
                path=str(r[1]),
                start_line=int(r[2]),
                end_line=int(r[3]),
                snippet=txt[:900],
                rank=float(r[5]) if r[5] is not None else None,
            )
        )
    return hits


def seed_queries() -> list[str]:
    seeds = (settings.repo_rag_query_seeds or "").strip()
    if not seeds:
        return []
    return [s.strip() for s in seeds.split(",") if s.strip()]
