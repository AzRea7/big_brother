# backend/app/services/repo_rag.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import text
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
    t = text.lower()
    score = 0.0
    for term in q_terms:
        if not term:
            continue
        # count occurrences (cheap)
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


async def search_chunks(
    db: Session,
    snapshot_id: int,
    query: str,
    top_k: int = 10,
    mode: str = "auto",
    path_contains: Optional[str] = None,
) -> Tuple[str, List[RepoChunk]]:
    """
    Returns (mode_used, chunks)
    mode: auto|fts|embeddings
    """
    top_k = max(1, min(int(top_k), settings.RAG_MAX_TOP_K))
    mode = (mode or "auto").lower().strip()

    # base query
    q = db.query(RepoChunk).filter(RepoChunk.snapshot_id == snapshot_id)
    if path_contains:
        q = q.filter(RepoChunk.path.contains(path_contains))

    # pull candidates (cap for safety)
    candidates = q.order_by(RepoChunk.id.asc()).limit(2000).all()

    if mode == "embeddings" or (mode == "auto" and settings.EMBEDDINGS_PROVIDER.lower() != "off"):
        # require embeddings exist on rows
        emb_rows = [c for c in candidates if c.embedding_json]
        if emb_rows:
            qvec = (await embed_texts([query]))[0]
            qnorm = math.sqrt(sum(x * x for x in qvec)) or 1.0

            scored: List[Tuple[float, RepoChunk]] = []
            for c in emb_rows:
                vec = loads_embedding(c.embedding_json or "[]")
                # we stored scaled int norms; recover approximate norm:
                cnorm = None
                if c.embedding_norm:
                    cnorm = float(c.embedding_norm) / 1_000_000.0
                s = cosine_similarity(qvec, vec, a_norm=qnorm, b_norm=cnorm)
                scored.append((s, c))

            scored.sort(key=lambda x: x[0], reverse=True)
            return "embeddings", [c for _, c in scored[:top_k]]

        # fallback if no embeddings present
        if mode == "embeddings":
            return "fts", _fts_fallback(candidates, query, top_k)

    # default: fts-like fallback
    return "fts", _fts_fallback(candidates, query, top_k)


def _fts_fallback(candidates: List[RepoChunk], query: str, top_k: int) -> List[RepoChunk]:
    terms = _tokenize(query)
    scored = []
    for c in candidates:
        scored.append((_simple_keyword_score(c.chunk_text, terms), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    # filter out pure zeros unless we’d return nothing
    nonzero = [c for s, c in scored if s > 0.0]
    if nonzero:
        return nonzero[:top_k]
    return [c for _, c in scored[:top_k]]
