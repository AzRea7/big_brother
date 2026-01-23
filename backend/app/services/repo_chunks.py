# backend/app/services/repo_chunks.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ..models import RepoChunk, RepoChunkEmbedding, RepoFile
from .repo_embeddings import (
    cosine_similarity,
    embed_texts,
    embedding_model_name,
    embeddings_enabled,
)


@dataclass(frozen=True)
class ChunkHit:
    """A single retrieved chunk hit.

    score is a *mode-agnostic quality score* in [0, 1] where higher is better.

    - Postgres FTS: ts_rank_cd already behaves roughly like [0,1] → clamped.
    - SQLite FTS5 bm25: lower is better → score = 1 / (1 + bm25)
    - Embeddings: cosine similarity in [-1,1] → score = clamp((cos+1)/2)
    """

    id: int
    snapshot_id: int
    path: str
    start_line: int
    end_line: int
    chunk_text: str
    symbols_json: Optional[str]
    score: Optional[float]


_SYMBOL_RE = re.compile(r"^\s*(class|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b")


def _extract_symbols(lines: list[str]) -> list[str]:
    out: list[str] = []
    for ln in lines:
        m = _SYMBOL_RE.match(ln)
        if m:
            out.append(m.group(2))
        if len(out) >= 25:
            break
    return out


def _dialect_name(db: Session) -> str:
    """Detect postgres vs sqlite without importing engine types."""
    try:
        bind = db.get_bind()
        if bind is None:
            return ""
        return str(getattr(bind.dialect, "name", "") or "").lower()
    except Exception:
        return ""


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def load_chunk_text(*, db: Session, chunk_id: int) -> dict[str, Any]:
    """Load a single chunk by id for UI/debug viewing."""
    c = db.get(RepoChunk, int(chunk_id))
    if not c:
        return {"ok": False, "error": "chunk_not_found", "chunk_id": int(chunk_id)}

    symbols: list[str] = []
    if c.symbols_json:
        try:
            symbols = json.loads(c.symbols_json)
        except Exception:
            symbols = []

    return {
        "ok": True,
        "chunk": {
            "id": int(c.id),
            "snapshot_id": int(c.snapshot_id),
            "path": str(c.path),
            "start_line": int(c.start_line),
            "end_line": int(c.end_line),
            "symbols": symbols,
            "chunk_text": str(c.chunk_text or ""),
            "created_at": c.created_at.isoformat() if getattr(c, "created_at", None) else None,
        },
    }


def chunk_snapshot(
    db: Session,
    snapshot_id: int,
    *,
    force: bool = False,
    max_lines: int = 140,
    overlap: int = 25,
    max_chars: int = 7000,
) -> dict[str, Any]:
    """Build RepoChunk rows from RepoFile content for a snapshot."""
    max_lines = max(40, int(max_lines))
    overlap = max(0, min(int(overlap), max_lines - 1))

    existing = db.execute(
        text("SELECT COUNT(1) FROM repo_chunks WHERE snapshot_id = :sid"),
        {"sid": snapshot_id},
    ).scalar()
    existing_n = int(existing or 0)
    if existing_n > 0 and not force:
        return {"snapshot_id": snapshot_id, "inserted": 0, "deleted": 0, "total": existing_n, "skipped": True}

    deleted = (
        db.execute(text("DELETE FROM repo_chunks WHERE snapshot_id = :sid"), {"sid": snapshot_id}).rowcount or 0
    )
    db.commit()

    files = db.scalars(
        select(RepoFile)
        .where(RepoFile.snapshot_id == snapshot_id)
        .where(RepoFile.content_kind == "text")
    ).all()

    inserted = 0
    for f in files:
        raw = (getattr(f, "content_text", None) or f.content or "")
        if not raw:
            continue

        lines = raw.splitlines()
        n = len(lines)
        if n == 0:
            continue

        i = 0
        while i < n:
            start = i
            end = min(n, i + max_lines)
            chunk_lines = lines[start:end]

            chunk_text = "\n".join(chunk_lines).strip("\n")
            if not chunk_text:
                i = end
                continue

            if len(chunk_text) > max_chars:
                chunk_text = chunk_text[:max_chars] + "\n...<truncated>..."

            symbols = _extract_symbols(chunk_lines)
            symbols_json = json.dumps(symbols, ensure_ascii=False) if symbols else None

            db.add(
                RepoChunk(
                    snapshot_id=snapshot_id,
                    path=f.path,
                    start_line=start + 1,
                    end_line=end,
                    chunk_text=chunk_text,
                    symbols_json=symbols_json,
                    created_at=datetime.utcnow(),
                )
            )
            inserted += 1

            if end >= n:
                break
            i = end - overlap

    db.commit()

    total = db.execute(
        text("SELECT COUNT(1) FROM repo_chunks WHERE snapshot_id = :sid"),
        {"sid": snapshot_id},
    ).scalar()

    return {"snapshot_id": snapshot_id, "inserted": inserted, "deleted": int(deleted), "total": int(total or 0)}


def _fts_available(db: Session) -> bool:
    try:
        row = db.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='repo_chunks_fts'")).fetchone()
        return bool(row)
    except Exception:
        return False


def _normalize_fts_query(q: str) -> str:
    q = (q or "").strip()
    if not q:
        return ""
    toks = re.split(r"\s+", q)
    cleaned: list[str] = []
    for t in toks:
        t = t.strip()
        if not t:
            continue
        t = t.replace('"', "")
        if re.search(r"[^A-Za-z0-9_:/.-]", t):
            cleaned.append(f'"{t}"')
        else:
            cleaned.append(t)
    return " OR ".join(cleaned)


def search_chunks_fts(
    db: Session,
    *,
    snapshot_id: int,
    query: str,
    top_k: int = 10,
    path_contains: Optional[str] = None,
) -> list[ChunkHit]:
    if not _fts_available(db):
        return []
    q = _normalize_fts_query(query)
    if not q:
        return []

    params: dict[str, Any] = {"sid": snapshot_id, "q": q, "k": int(top_k)}
    path_filter_sql = ""
    if path_contains:
        params["p"] = f"%{path_contains}%"
        path_filter_sql = " AND c.path LIKE :p "

    sql = f"""
    SELECT
      c.id, c.snapshot_id, c.path, c.start_line, c.end_line,
      c.chunk_text, c.symbols_json,
      bm25(repo_chunks_fts) as bm25_rank
    FROM repo_chunks_fts
    JOIN repo_chunks c ON c.id = repo_chunks_fts.rowid
    WHERE c.snapshot_id = :sid
      AND repo_chunks_fts MATCH :q
      {path_filter_sql}
    ORDER BY bm25_rank ASC
    LIMIT :k
    """

    rows = db.execute(text(sql), params).fetchall()
    out: list[ChunkHit] = []
    for r in rows:
        bm25_rank = float(r[7]) if r[7] is not None else None
        score = None
        if bm25_rank is not None and bm25_rank >= 0.0:
            score = 1.0 / (1.0 + bm25_rank)
        out.append(
            ChunkHit(
                id=int(r[0]),
                snapshot_id=int(r[1]),
                path=str(r[2]),
                start_line=int(r[3]),
                end_line=int(r[4]),
                chunk_text=str(r[5]),
                symbols_json=r[6],
                score=score,
            )
        )
    return out


def search_chunks_postgres_fts(
    db: Session,
    *,
    snapshot_id: int,
    query: str,
    top_k: int = 10,
    path_contains: Optional[str] = None,
) -> list[ChunkHit]:
    """
    Postgres full-text search.

    IMPORTANT:
    - websearch_to_tsquery behaves more like a real user search box (supports quoting, OR/AND).
    - plainto_tsquery behaves like strict AND of all terms, often too strict for code chunks.
    """
    q = (query or "").strip()
    if not q:
        return []

    params: dict[str, Any] = {"sid": snapshot_id, "q": q, "k": int(top_k)}
    path_filter_sql = ""
    if path_contains:
        params["p"] = f"%{path_contains}%"
        path_filter_sql = " AND c.path ILIKE :p "

    sql = f"""
    SELECT
      c.id, c.snapshot_id, c.path, c.start_line, c.end_line,
      c.chunk_text, c.symbols_json,
      ts_rank_cd(
        to_tsvector('english', coalesce(c.chunk_text,'')),
        websearch_to_tsquery('english', :q)
      ) AS rank
    FROM repo_chunks c
    WHERE c.snapshot_id = :sid
      {path_filter_sql}
      AND to_tsvector('english', coalesce(c.chunk_text,'')) @@ websearch_to_tsquery('english', :q)
    ORDER BY rank DESC
    LIMIT :k
    """

    rows = db.execute(text(sql), params).fetchall()
    out: list[ChunkHit] = []
    for r in rows:
        rank = float(r[7]) if r[7] is not None else None
        out.append(
            ChunkHit(
                id=int(r[0]),
                snapshot_id=int(r[1]),
                path=str(r[2]),
                start_line=int(r[3]),
                end_line=int(r[4]),
                chunk_text=str(r[5]),
                symbols_json=r[6],
                score=_clamp01(rank) if rank is not None else None,
            )
        )
    return out



async def build_embeddings_for_snapshot(
    db: Session,
    snapshot_id: int,
    *,
    force: bool = False,
    batch_size: int = 64,
    max_chunks: Optional[int] = None,
) -> dict[str, Any]:
    if not embeddings_enabled():
        return {"snapshot_id": snapshot_id, "embedded": 0, "skipped": 0, "error": "embeddings_disabled"}

    model = embedding_model_name()

    if force:
        db.execute(
            text(
                """
                DELETE FROM repo_chunk_embeddings
                WHERE model = :m AND chunk_id IN (
                  SELECT id FROM repo_chunks WHERE snapshot_id = :sid
                )
                """
            ),
            {"sid": snapshot_id, "m": model},
        )
        db.commit()

    sql = """
    SELECT c.id
    FROM repo_chunks c
    LEFT JOIN repo_chunk_embeddings e
      ON e.chunk_id = c.id AND e.model = :m
    WHERE c.snapshot_id = :sid
      AND e.id IS NULL
    ORDER BY c.id
    """
    ids = [int(r[0]) for r in db.execute(text(sql), {"sid": snapshot_id, "m": model}).fetchall()]
    if max_chunks is not None:
        ids = ids[: int(max_chunks)]

    if not ids:
        return {"snapshot_id": snapshot_id, "embedded": 0, "skipped": 0, "model": model}

    embedded = 0
    skipped = 0

    for i in range(0, len(ids), int(batch_size)):
        batch_ids = ids[i : i + int(batch_size)]
        chunks = db.scalars(select(RepoChunk).where(RepoChunk.id.in_(batch_ids))).all()

        texts: list[str] = []
        chunk_id_order: list[int] = []
        for c in chunks:
            t = (c.chunk_text or "").strip()
            if not t:
                skipped += 1
                continue
            t = t[:6000]
            texts.append(t)
            chunk_id_order.append(int(c.id))

        if not texts:
            continue

        vectors = await embed_texts(texts)
        if len(vectors) != len(chunk_id_order):
            raise RuntimeError("Embeddings provider returned mismatched vector count.")

        now = datetime.utcnow()
        for cid, vec in zip(chunk_id_order, vectors):
            db.add(
                RepoChunkEmbedding(
                    chunk_id=cid,
                    model=model,
                    vector_json=json.dumps(vec, separators=(",", ":"), ensure_ascii=False),
                    created_at=now,
                )
            )
            embedded += 1

        db.commit()

    return {"snapshot_id": snapshot_id, "embedded": embedded, "skipped": skipped, "model": model}


def _snapshot_has_embeddings(db: Session, *, snapshot_id: int, model: str) -> bool:
    row = db.execute(
        text(
            """
            SELECT 1
            FROM repo_chunk_embeddings e
            JOIN repo_chunks c ON c.id = e.chunk_id
            WHERE c.snapshot_id = :sid AND e.model = :m
            LIMIT 1
            """
        ),
        {"sid": snapshot_id, "m": model},
    ).fetchone()
    return bool(row)


async def search_chunks_embeddings(
    db: Session,
    *,
    snapshot_id: int,
    query: str,
    top_k: int = 10,
    path_contains: Optional[str] = None,
) -> list[ChunkHit]:
    if not embeddings_enabled():
        return []

    model = embedding_model_name()
    q = (query or "").strip()
    if not q:
        return []

    q_vec = (await embed_texts([q]))[0]

    params: dict[str, Any] = {"sid": snapshot_id, "m": model}
    path_filter_sql = ""
    if path_contains:
        params["p"] = f"%{path_contains}%"
        path_filter_sql = " AND c.path LIKE :p "

    sql = f"""
    SELECT
      c.id, c.snapshot_id, c.path, c.start_line, c.end_line,
      c.chunk_text, c.symbols_json,
      e.vector_json
    FROM repo_chunks c
    JOIN repo_chunk_embeddings e ON e.chunk_id = c.id AND e.model = :m
    WHERE c.snapshot_id = :sid
      {path_filter_sql}
    """

    rows = db.execute(text(sql), params).fetchall()

    scored: list[ChunkHit] = []
    for r in rows:
        vec = json.loads(r[7]) if r[7] else []
        cos = cosine_similarity([float(x) for x in q_vec], [float(x) for x in vec])
        score = _clamp01((float(cos) + 1.0) / 2.0)
        scored.append(
            ChunkHit(
                id=int(r[0]),
                snapshot_id=int(r[1]),
                path=str(r[2]),
                start_line=int(r[3]),
                end_line=int(r[4]),
                chunk_text=str(r[5]),
                symbols_json=r[6],
                score=score,
            )
        )

    scored.sort(key=lambda h: float(h.score or 0.0), reverse=True)
    return scored[: int(top_k)]


async def search_chunks(
    db: Session,
    *,
    snapshot_id: int,
    query: str,
    top_k: int = 10,
    mode: str = "auto",
    path_contains: Optional[str] = None,
) -> tuple[str, list[ChunkHit]]:
    mode = (mode or "auto").strip().lower()
    dialect = _dialect_name(db)
    top_k = max(1, min(int(top_k), 50))

    def _best_fts() -> tuple[str, list[ChunkHit]]:
        if dialect.startswith("postgres"):
            return "fts_pg", search_chunks_postgres_fts(
                db, snapshot_id=snapshot_id, query=query, top_k=top_k, path_contains=path_contains
            )
        if _fts_available(db):
            return "fts", search_chunks_fts(
                db, snapshot_id=snapshot_id, query=query, top_k=top_k, path_contains=path_contains
            )
        return "none", []

    if mode == "embeddings":
        return "embeddings", await search_chunks_embeddings(
            db, snapshot_id=snapshot_id, query=query, top_k=top_k, path_contains=path_contains
        )

    if mode == "fts":
        return _best_fts()

    fts_mode, fts_hits = _best_fts()

    if embeddings_enabled():
        model = embedding_model_name()
        if _snapshot_has_embeddings(db, snapshot_id=snapshot_id, model=model):
            min_hits = max(2, top_k // 3)
            best_score = max((h.score or 0.0) for h in fts_hits) if fts_hits else 0.0
            fts_looks_weak = (len(fts_hits) < min_hits) or (best_score < 0.15)

            if fts_looks_weak:
                emb_hits = await search_chunks_embeddings(
                    db, snapshot_id=snapshot_id, query=query, top_k=top_k, path_contains=path_contains
                )
                by_id: dict[int, ChunkHit] = {h.id: h for h in fts_hits}
                for h in emb_hits:
                    prev = by_id.get(h.id)
                    if prev is None or (h.score or 0.0) > (prev.score or 0.0):
                        by_id[h.id] = h

                merged = list(by_id.values())
                merged.sort(key=lambda h: float(h.score or 0.0), reverse=True)
                return "auto_mix", merged[:top_k]

    # ✅ FIX: return exactly (mode, hits)
    return fts_mode, fts_hits
