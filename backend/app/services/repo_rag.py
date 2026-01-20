# backend/app/services/repo_rag.py
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoChunk, RepoFile
from .repo_embeddings import embed_texts, loads_embedding, cosine_similarity


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


def _simple_keyword_score(text: str, q_terms: List[str]) -> float:
    """
    Cheap relevance scoring for FTS-like behavior without needing SQLite FTS tables.
    """
    t = (text or "").lower()
    score = 0.0
    for term in q_terms:
        if not term:
            continue
        score += float(t.count(term))
    return score


def _tokenize(q: str) -> List[str]:
    q = (q or "").strip().lower()
    parts = [p for p in q.replace("/", " ").replace(":", " ").split() if len(p) >= 2]
    return parts[:20]


def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n…(truncated)…\n"


def chunk_snapshot(db: Session, snapshot_id: int, force: bool = False) -> dict:
    """
    Build repo_chunks rows for a snapshot using RepoFile.content_text.
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
        for start_line, end_line, chunk_text in _line_chunks(
            lines, settings.REPO_CHUNK_LINES, settings.REPO_CHUNK_OVERLAP
        ):
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
async def search_chunks(
    db: Session,
    snapshot_id: int,
    query: str,
    top_k: int = 10,
    mode: str = "auto",
    path_contains: Optional[str] = None,
) -> Tuple[str, List[ChunkHit]]:
    """
    Returns (mode_used, hits)
    mode: auto | fts | embeddings

    Note:
      - We do not depend on SQLite FTS tables here.
      - "fts" means "keyword scoring fallback".
      - Embeddings mode requires RepoChunk.embedding_json to be populated by a separate embed step.
    """
    cap = int(getattr(settings, "RAG_MAX_TOP_K", 20) or 20)
    top_k = max(1, min(int(top_k), cap))
    mode = (mode or "auto").lower().strip()

    # base query
    q = db.query(RepoChunk).filter(RepoChunk.snapshot_id == snapshot_id)
    if path_contains:
        q = q.filter(RepoChunk.path.contains(path_contains))

    # pull candidates (cap for safety)
    candidates: List[RepoChunk] = q.order_by(RepoChunk.id.asc()).limit(2000).all()

    def to_hits(chunks: List[RepoChunk], scores: Optional[List[float]] = None) -> List[ChunkHit]:
        out: List[ChunkHit] = []
        if scores is None:
            scores = [None] * len(chunks)  # type: ignore
        for c, s in zip(chunks, scores):
            out.append(
                ChunkHit(
                    id=c.id,
                    path=c.path,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    score=s,
                    chunk_text=c.chunk_text,
                )
            )
        return out

    # ---- embeddings path ----
    if mode == "embeddings" or (mode == "auto" and (settings.EMBEDDINGS_PROVIDER or "off").lower() != "off"):
        emb_rows = [c for c in candidates if getattr(c, "embedding_json", None)]
        if emb_rows:
            qvec = (await embed_texts([query]))[0]
            qnorm = math.sqrt(sum(x * x for x in qvec)) or 1.0

            scored: List[Tuple[float, RepoChunk]] = []
            for c in emb_rows:
                vec = loads_embedding(getattr(c, "embedding_json", None) or "[]")

                # optional stored norm (if your model has this column)
                cnorm = None
                en = getattr(c, "embedding_norm", None)
                if en:
                    # if you store scaled ints, this recovers approx float
                    try:
                        cnorm = float(en) / 1_000_000.0
                    except Exception:
                        cnorm = None

                s = cosine_similarity(qvec, vec, a_norm=qnorm, b_norm=cnorm)
                scored.append((s, c))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:top_k]
            return "embeddings", to_hits([c for _, c in top], [float(s) for s, _ in top])

        # if forced embeddings but none exist, fall back
        if mode == "embeddings":
            chunks = _keyword_fallback(candidates, query, top_k)
            return "fts", to_hits(chunks)

    # ---- default keyword fallback ----
    chunks = _keyword_fallback(candidates, query, top_k)
    return "fts", to_hits(chunks)


def _keyword_fallback(candidates: List[RepoChunk], query: str, top_k: int) -> List[RepoChunk]:
    terms = _tokenize(query)
    scored: List[Tuple[float, RepoChunk]] = []
    for c in candidates:
        scored.append((_simple_keyword_score(c.chunk_text, terms), c))
    scored.sort(key=lambda x: x[0], reverse=True)

    nonzero = [c for s, c in scored if s > 0.0]
    if nonzero:
        return nonzero[:top_k]
    return [c for _, c in scored[:top_k]]
