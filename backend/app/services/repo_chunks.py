# backend/app/services/repo_chunks.py
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoChunk, RepoChunkEmbedding, RepoFile
from ..ai.embeddings import embed_texts


@dataclass(frozen=True)
class ChunkHit:
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
    """
    Used to detect postgres vs sqlite without importing engine types.
    """
    try:
        bind = db.get_bind()
        if bind is None:
            return ""
        return str(getattr(bind.dialect, "name", "") or "").lower()
    except Exception:
        return ""


def load_chunk_text(*, db: Session, chunk_id: int) -> dict[str, Any]:
    """
    Load a single chunk by id for UI/debug viewing.
    """
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
    """
    Build RepoChunk rows from RepoFile content for a snapshot.

    If force=False and chunks already exist, this will NO-OP and just return counts.
    If force=True, it deletes and rebuilds.

    Returns: {"snapshot_id":..., "inserted":..., "deleted":..., "total":...}
    """
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


# -----------------------------
# SQLite FTS (existing behavior)
# -----------------------------
def _fts_available(db: Session) -> bool:
    """
    We create FTS in migrations.ensure_schema if SQLite supports it.
    This checks presence of the virtual table.
    """
    try:
        row = db.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='repo_chunks_fts'")
        ).fetchone()
        return bool(row)
    except Exception:
        return False


def _normalize_fts_query(q: str) -> str:
    """
    Minimal FTS hygiene:
    - split into tokens
    - quote tokens with special chars
    - join with OR so you get recall
    """
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
      bm25(repo_chunks_fts) as rank
    FROM repo_chunks_fts
    JOIN repo_chunks c ON c.id = repo_chunks_fts.rowid
    WHERE c.snapshot_id = :sid
      AND repo_chunks_fts MATCH :q
      {path_filter_sql}
    ORDER BY rank
    LIMIT :k
    """

    rows = db.execute(text(sql), params).fetchall()
    out: list[ChunkHit] = []
    for r in rows:
        out.append(
            ChunkHit(
                id=int(r[0]),
                snapshot_id=int(r[1]),
                path=str(r[2]),
                start_line=int(r[3]),
                end_line=int(r[4]),
                chunk_text=str(r[5]),
                symbols_json=r[6],
                score=float(-r[7]) if r[7] is not None else None,
            )
        )
    return out


# -----------------------------
# Postgres FTS (NEW)
# -----------------------------
def search_chunks_postgres_fts(
    db: Session,
    *,
    snapshot_id: int,
    query: str,
    top_k: int = 10,
    path_contains: Optional[str] = None,
) -> list[ChunkHit]:
    q = (query or "").strip()
    if not q:
        return []

    params: dict[str, Any] = {"sid": snapshot_id, "q": q, "k": int(top_k)}
    path_filter_sql = ""
    if path_contains:
        params["p"] = f"%{path_contains}%"
        path_filter_sql = " AND c.path ILIKE :p "

    # No extensions required; uses built-in Postgres text search.
    sql = f"""
    SELECT
      c.id, c.snapshot_id, c.path, c.start_line, c.end_line,
      c.chunk_text, c.symbols_json,
      ts_rank_cd(
        to_tsvector('english', coalesce(c.chunk_text,'')),
        plainto_tsquery('english', :q)
      ) AS rank
    FROM repo_chunks c
    WHERE c.snapshot_id = :sid
      {path_filter_sql}
      AND to_tsvector('english', coalesce(c.chunk_text,'')) @@ plainto_tsquery('english', :q)
    ORDER BY rank DESC
    LIMIT :k
    """

    rows = db.execute(text(sql), params).fetchall()
    out: list[ChunkHit] = []
    for r in rows:
        out.append(
            ChunkHit(
                id=int(r[0]),
                snapshot_id=int(r[1]),
                path=str(r[2]),
                start_line=int(r[3]),
                end_line=int(r[4]),
                chunk_text=str(r[5]),
                symbols_json=r[6],
                score=float(r[7]) if r[7] is not None else None,
            )
        )
    return out


# -----------------------------
# Embeddings (existing behavior)
# -----------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(min(len(a), len(b))):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom <= 1e-12:
        return 0.0
    return dot / denom


def embeddings_enabled() -> bool:
    return bool(getattr(settings, "EMBEDDINGS_ENABLED", False))


def embedding_model_name() -> str:
    return str(getattr(settings, "OPENAI_EMBEDDING_MODEL", "") or "text-embedding-3-small")


async def build_embeddings_for_snapshot(
    db: Session,
    *,
    snapshot_id: int,
    force: bool = False,
    batch_size: int = 64,
    max_chunks: Optional[int] = None,
) -> dict[str, Any]:
    """
    Generate embeddings for all chunks in a snapshot (if EMBEDDINGS_ENABLED=true).
    """
    if not embeddings_enabled():
        return {"snapshot_id": snapshot_id, "embedded": 0, "skipped": 0, "error": "EMBEDDINGS_ENABLED=false"}

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
        vec = json.loads(r[7])
        score = _cosine(q_vec, vec)
        scored.append(
            ChunkHit(
                id=int(r[0]),
                snapshot_id=int(r[1]),
                path=str(r[2]),
                start_line=int(r[3]),
                end_line=int(r[4]),
                chunk_text=str(r[5]),
                symbols_json=r[6],
                score=float(score),
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
    """
    mode:
      - auto: embeddings if enabled AND embeddings exist,
              else postgres fts if postgres,
              else sqlite fts if available,
              else none
      - embeddings: force embeddings
      - fts: force best-available text search
    """
    mode = (mode or "auto").strip().lower()
    dialect = _dialect_name(db)

    if mode == "embeddings":
        return "embeddings", await search_chunks_embeddings(
            db, snapshot_id=snapshot_id, query=query, top_k=top_k, path_contains=path_contains
        )

    if mode == "fts":
        if dialect.startswith("postgres"):
            return "fts_pg", search_chunks_postgres_fts(
                db, snapshot_id=snapshot_id, query=query, top_k=top_k, path_contains=path_contains
            )
        return "fts", search_chunks_fts(
            db, snapshot_id=snapshot_id, query=query, top_k=top_k, path_contains=path_contains
        )

    # auto:
    if embeddings_enabled():
        model = embedding_model_name()
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
        if row:
            hits = await search_chunks_embeddings(
                db, snapshot_id=snapshot_id, query=query, top_k=top_k, path_contains=path_contains
            )
            return "embeddings", hits

    if dialect.startswith("postgres"):
        hits = search_chunks_postgres_fts(
            db, snapshot_id=snapshot_id, query=query, top_k=top_k, path_contains=path_contains
        )
        return "fts_pg", hits

    if _fts_available(db):
        hits = search_chunks_fts(db, snapshot_id=snapshot_id, query=query, top_k=top_k, path_contains=path_contains)
        return "fts", hits

    return "none", []
