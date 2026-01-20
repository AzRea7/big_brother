from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoChunk, RepoFile, RepoFinding, RepoSnapshot, Task

# --- Import compatibility (repo evolves; keep this file resilient) ---
# We want ONE canonical signal counter used everywhere.
try:
    # Newer location (recommended)
    from .code_signals import count_markers_in_text  # type: ignore
except Exception:  # pragma: no cover
    # Older location
    from ..ai.repo_tasks import count_markers_in_text  # type: ignore

try:
    # LLM task generator (JSON schema: {"tasks":[...]}), contains bounded prompt builder.
    from ..ai.repo_tasks import generate_repo_tasks_json  # type: ignore
except Exception:  # pragma: no cover
    from ..ai.repo_tasks import generate_repo_tasks_json  # type: ignore


# -------------------------
# Deterministic fallback (only used if LLM fails)
# -------------------------

_FALLBACK_RE = re.compile(r"\b(TODO|FIXME|HACK|XXX|BUG|NOTE)\b[:\-\s]*(.*)", re.IGNORECASE)

_ACTION_WORDS_RE = re.compile(
    r"\b(should|must|need|needs|fix|implement|add|remove|refactor|rename|handle|support|validate|sanitize|secure|test)\b",
    re.IGNORECASE,
)

_GARBAGE_RE = re.compile(r"^[\s\"',.:;()\[\]{}<>/\\-]*$")

_KIND_SCORE = {"BUG": 100, "FIXME": 90, "HACK": 80, "XXX": 70, "TODO": 60, "NOTE": 10}


@dataclass
class SuggestedTask:
    title: str
    notes: str
    link: str | None
    tags: str
    priority: int = 3
    estimated_minutes: int = 60
    blocks_me: bool = False
    starter: str | None = None
    dod: str | None = None


# -------------------------
# Repo links
# -------------------------

def _repo_link(snapshot: RepoSnapshot, path: str, line: int | None = None) -> str:
    base = f"repo://{snapshot.repo}?branch={snapshot.branch}&commit_sha={snapshot.commit_sha or ''}"
    return f"{base}#{path}:L{line}" if line is not None else f"{base}#{path}"


# -------------------------
# Signal counts (used by routes + prompt builder)
# -------------------------

def compute_signal_counts(db: Session, snapshot_id: int) -> dict[str, int]:
    files = db.scalars(select(RepoFile).where(RepoFile.snapshot_id == snapshot_id)).all()
    return compute_signal_counts_for_files(files)


def compute_signal_counts_for_files(files: list[RepoFile]) -> dict[str, int]:
    total = len(files)

    files_with: dict[str, int] = {
        "todo": 0,
        "fixme": 0,
        "hack": 0,
        "xxx": 0,
        "bug": 0,
        "note": 0,
        "dotdotdot": 0,
        "auth": 0,
        "timeout": 0,
        "retry": 0,
        "rate_limit": 0,
        "validation": 0,
        "logging": 0,
        "metrics": 0,
        "db": 0,
        "tests": 0,
        "ci": 0,
        "docker": 0,
        "config": 0,
        "secrets": 0,
        "nplus1": 0,
        "cors": 0,
        "csrf": 0,
    }

    impl = 0

    for f in files:
        if f.content_kind != "text" or not (getattr(f, "content_text", None) or f.content):
            continue

        text = (getattr(f, "content_text", None) or f.content or "")
        if not text:
            continue

        sig = count_markers_in_text(text)

        # classic
        if sig.get("todo_count", 0) > 0:
            files_with["todo"] += 1
        if sig.get("fixme_count", 0) > 0:
            files_with["fixme"] += 1
        if sig.get("hack_count", 0) > 0:
            files_with["hack"] += 1
        if sig.get("xxx_count", 0) > 0:
            files_with["xxx"] += 1
        if sig.get("bug_count", 0) > 0:
            files_with["bug"] += 1
        if sig.get("note_count", 0) > 0:
            files_with["note"] += 1
        if sig.get("dotdotdot_count", 0) > 0:
            files_with["dotdotdot"] += 1

        # production signals
        if sig.get("auth_signal", 0) > 0:
            files_with["auth"] += 1
        if sig.get("timeout_signal", 0) > 0:
            files_with["timeout"] += 1
        if sig.get("retry_signal", 0) > 0:
            files_with["retry"] += 1
        if sig.get("rate_limit_signal", 0) > 0:
            files_with["rate_limit"] += 1
        if sig.get("input_validation_signal", 0) > 0:
            files_with["validation"] += 1
        if sig.get("logging_signal", 0) > 0:
            files_with["logging"] += 1
        if sig.get("metrics_signal", 0) > 0:
            files_with["metrics"] += 1
        if sig.get("db_signal", 0) > 0:
            files_with["db"] += 1
        if sig.get("tests_signal", 0) > 0:
            files_with["tests"] += 1
        if sig.get("ci_signal", 0) > 0:
            files_with["ci"] += 1
        if sig.get("docker_signal", 0) > 0:
            files_with["docker"] += 1
        if sig.get("config_signal", 0) > 0:
            files_with["config"] += 1
        if sig.get("secrets_signal", 0) > 0:
            files_with["secrets"] += 1
        if sig.get("nplus1_signal", 0) > 0:
            files_with["nplus1"] += 1
        if sig.get("cors_signal", 0) > 0:
            files_with["cors"] += 1
        if sig.get("csrf_signal", 0) > 0:
            files_with["csrf"] += 1

        if any(x in text for x in ("raise NotImplementedError", "IMPLEMENT", "stub", "pass  #")):
            impl += 1

    return {
        "total_files": total,
        "files_with_todo": files_with["todo"],
        "files_with_fixme": files_with["fixme"],
        "files_with_hack": files_with["hack"],
        "files_with_xxx": files_with["xxx"],
        "files_with_bug": files_with["bug"],
        "files_with_note": files_with["note"],
        "files_with_dotdotdot": files_with["dotdotdot"],
        "files_with_impl_signals": impl,
        "files_with_auth": files_with["auth"],
        "files_with_timeout": files_with["timeout"],
        "files_with_retry": files_with["retry"],
        "files_with_rate_limit": files_with["rate_limit"],
        "files_with_validation": files_with["validation"],
        "files_with_logging": files_with["logging"],
        "files_with_metrics": files_with["metrics"],
        "files_with_db": files_with["db"],
        "files_with_tests": files_with["tests"],
        "files_with_ci": files_with["ci"],
        "files_with_docker": files_with["docker"],
        "files_with_config": files_with["config"],
        "files_with_secrets": files_with["secrets"],
        "files_with_nplus1": files_with["nplus1"],
        "files_with_cors": files_with["cors"],
        "files_with_csrf": files_with["csrf"],
    }


# -------------------------
# ✅ Level 2 RAG: chunk builder
# -------------------------

def _iter_line_chunks(lines: list[str], max_lines: int, overlap: int) -> Iterable[tuple[int, int, str]]:
    """
    Yields (start_line, end_line, chunk_text) with overlap, 1-indexed line numbers.
    """
    if max_lines <= 0:
        max_lines = 220
    if overlap < 0:
        overlap = 0
    if overlap >= max_lines:
        overlap = max_lines // 4

    i = 0
    n = len(lines)
    while i < n:
        start = i
        end = min(n, i + max_lines)
        chunk = "\n".join(lines[start:end])
        yield (start + 1, end, chunk)
        if end >= n:
            break
        i = max(0, end - overlap)


def build_chunks_for_snapshot(
    db: Session,
    snapshot_id: int,
    *,
    max_lines: int = 220,
    overlap: int = 40,
) -> dict[str, Any]:
    """
    Builds RepoChunk rows from RepoFile content for the snapshot.

    Safe behavior:
      - deletes existing chunks for snapshot, then rebuilds deterministically
      - only chunks files where content_kind == "text"
    """
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        return {"snapshot_id": snapshot_id, "inserted": 0, "deleted": 0, "files": 0}

    deleted = db.execute(delete(RepoChunk).where(RepoChunk.snapshot_id == snapshot_id)).rowcount or 0

    files = db.scalars(
        select(RepoFile)
        .where(RepoFile.snapshot_id == snapshot_id)
        .where(RepoFile.content_kind == "text")
    ).all()

    inserted = 0
    for f in files:
        text = (getattr(f, "content_text", None) or f.content or "")
        if not text:
            continue
        lines = text.splitlines()
        if not lines:
            continue

        for start_line, end_line, chunk_text in _iter_line_chunks(lines, max_lines=max_lines, overlap=overlap):
            if not chunk_text.strip():
                continue
            db.add(
                RepoChunk(
                    snapshot_id=snapshot_id,
                    path=f.path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_text=chunk_text,
                    symbols_json=None,
                )
            )
            inserted += 1

    db.commit()
    return {
        "snapshot_id": snapshot_id,
        "deleted": deleted,
        "inserted": inserted,
        "files": len(files),
        "max_lines": max_lines,
        "overlap": overlap,
    }


# -------------------------
# ✅ Level 2 RAG: retrieval (FTS + optional embeddings)
# -------------------------

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _tokenize(q: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(q or "")][:24]


def _fts_score(text: str, tokens: list[str]) -> float:
    """
    Lightweight scoring:
      - counts token occurrences
      - small bonus if tokens appear early
    """
    if not text or not tokens:
        return 0.0
    t = text.lower()
    score = 0.0
    for tok in tokens:
        c = t.count(tok)
        if c:
            score += 1.0 + min(3.0, float(c))  # cap per token
            first = t.find(tok)
            if 0 <= first < 200:
                score += 0.5
            if 0 <= first < 80:
                score += 0.5
    return score


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _chunk_has_embedding(c: RepoChunk) -> bool:
    """
    Embeddings are optional. If you later add RepoChunk.embedding_json or RepoChunk.embedding,
    this will start working automatically.
    """
    return bool(getattr(c, "embedding_json", None) or getattr(c, "embedding", None))


def _get_chunk_embedding(c: RepoChunk) -> Optional[list[float]]:
    raw = getattr(c, "embedding_json", None)
    if raw:
        try:
            v = json.loads(raw)
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                return [float(x) for x in v]
        except Exception:
            return None
    raw2 = getattr(c, "embedding", None)
    if isinstance(raw2, list) and raw2 and isinstance(raw2[0], (int, float)):
        return [float(x) for x in raw2]
    return None


def _embed_query_optional(query: str) -> Optional[list[float]]:
    """
    Optional query embedding hook.

    If you already added a real embedding service elsewhere, wire it by setting:
      - settings.EMBEDDINGS_ENABLED = true
      - and providing a function in backend/app/ai/embeddings.py: embed_text(query)->list[float]

    This file won’t crash if you haven't added it; it simply falls back to FTS.
    """
    if not bool(getattr(settings, "EMBEDDINGS_ENABLED", False)):
        return None

    try:
        from ..ai.embeddings import embed_text  # type: ignore
    except Exception:
        return None

    try:
        v = embed_text(query)  # type: ignore[misc]
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            return [float(x) for x in v]
    except Exception:
        return None
    return None


def search_chunks(
    db: Session,
    snapshot_id: int,
    *,
    query: str,
    top_k: int = 10,
    mode: str = "auto",
    path_contains: Optional[str] = None,
) -> dict[str, Any]:
    """
    Returns:
      {
        snapshot_id, query, mode_used,
        results: [{id,path,start_line,end_line,chunk_text,symbols_json}, ...]
      }
    """
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        return {"snapshot_id": snapshot_id, "query": query, "mode_used": "none", "results": []}

    q = (query or "").strip()
    if len(q) < 2:
        return {"snapshot_id": snapshot_id, "query": query, "mode_used": "none", "results": []}

    top_k = max(1, min(int(top_k), 30))

    stmt = select(RepoChunk).where(RepoChunk.snapshot_id == snapshot_id)
    if path_contains:
        stmt = stmt.where(RepoChunk.path.ilike(f"%{path_contains}%"))  # type: ignore[attr-defined]

    chunks = db.scalars(stmt).all()
    if not chunks:
        return {"snapshot_id": snapshot_id, "query": query, "mode_used": "none", "results": []}

    tokens = _tokenize(q)

    # Decide mode
    mode_used = mode
    query_vec: Optional[list[float]] = None
    if mode == "auto":
        # Only use embeddings if both (a) you enabled embeddings and (b) chunks actually have embeddings stored.
        if bool(getattr(settings, "EMBEDDINGS_ENABLED", False)) and any(_chunk_has_embedding(c) for c in chunks):
            query_vec = _embed_query_optional(q)
            if query_vec is not None:
                mode_used = "embeddings"
            else:
                mode_used = "fts"
        else:
            mode_used = "fts"

    if mode == "embeddings":
        query_vec = _embed_query_optional(q)
        if query_vec is None:
            mode_used = "fts"

    scored: list[tuple[float, RepoChunk]] = []

    if mode_used == "embeddings" and query_vec is not None:
        for c in chunks:
            v = _get_chunk_embedding(c)
            if v is None or len(v) != len(query_vec):
                continue
            scored.append((_cosine(query_vec, v), c))
    else:
        for c in chunks:
            scored.append((_fts_score(c.chunk_text, tokens), c))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [c for s, c in scored[:top_k] if s > 0.0] or [c for _, c in scored[:top_k]]

    results = [
        {
            "id": c.id,
            "snapshot_id": c.snapshot_id,
            "path": c.path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "chunk_text": c.chunk_text,
            "symbols_json": getattr(c, "symbols_json", None),
        }
        for c in picked
    ]

    return {"snapshot_id": snapshot_id, "query": query, "mode_used": mode_used, "results": results}


# -------------------------
# Fallback helpers
# -------------------------

def _is_actionable_note(msg: str) -> bool:
    m = (msg or "").strip()
    if not m:
        return False
    if _GARBAGE_RE.match(m):
        return False
    return bool(_ACTION_WORDS_RE.search(m))


def _clean_msg(msg: str) -> str:
    m = (msg or "").strip()
    if not m:
        return "Unspecified"
    if _GARBAGE_RE.match(m):
        return "Unspecified"
    return m[:160]


def _priority_for_kind(kind: str) -> int:
    k = kind.upper()
    if k in ("BUG", "FIXME"):
        return 5
    if k in ("HACK", "XXX"):
        return 4
    if k == "TODO":
        return 3
    return 2


def _minutes_for_kind(kind: str) -> int:
    k = kind.upper()
    if k in ("BUG", "FIXME"):
        return 90
    if k in ("HACK", "XXX"):
        return 75
    if k == "TODO":
        return 60
    return 45


def _blocks_for_kind(kind: str) -> bool:
    return kind.upper() in ("BUG", "FIXME", "HACK")


def _task_key(title: str, link: str | None) -> str:
    return (title or "").strip().lower() + "|" + (link or "")


# -------------------------
# Deterministic fallback task suggestion (filtered + ranked)
# -------------------------

def _suggest_tasks_from_signals(db: Session, snapshot_id: int, project: str, limit: int = 25) -> list[SuggestedTask]:
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        return []

    files = db.scalars(
        select(RepoFile)
        .where(RepoFile.snapshot_id == snapshot_id)
        .where(RepoFile.content_kind == "text")
    ).all()

    found: list[dict[str, Any]] = []

    for f in files:
        text = (getattr(f, "content_text", None) or f.content or "")
        if not text:
            continue

        lines = text.splitlines()
        for i, line in enumerate(lines[:2500], start=1):
            m = _FALLBACK_RE.search(line)
            if not m:
                continue

            kind = m.group(1).upper()
            raw_msg = (m.group(2) or "").strip()

            if kind == "NOTE" and not _is_actionable_note(raw_msg):
                continue

            msg = _clean_msg(raw_msg)
            if msg == "Unspecified" and kind == "NOTE":
                continue

            found.append(
                {
                    "kind": kind,
                    "msg": msg,
                    "path": f.path,
                    "line": i,
                    "line_text": line.strip(),
                    "score": _KIND_SCORE.get(kind, 0),
                }
            )

    if not found:
        return [
            SuggestedTask(
                title="Repo scan: identify top 3 high-impact improvements",
                notes=(
                    "No actionable TODO/FIXME/HACK/XXX/BUG markers were found in the stored snapshot content.\n\n"
                    "Starter (5–10 min): Skim API entrypoints (/main.py, /routes), auth, timeouts, retries, "
                    "error handling, and tests.\n"
                    "DoD: Create 3 concrete tasks with repo links and measurable acceptance criteria."
                ),
                link=_repo_link(snap, "onehaven/"),
                tags="repo,autogen,scan",
                priority=3,
                estimated_minutes=60,
                blocks_me=False,
                starter="Skim /main.py, /routes, services, and tests; write down 3 concrete improvements.",
                dod="3 concrete tasks exist with repo links and clear DoD.",
            )
        ]

    found.sort(key=lambda x: (-x["score"], x["path"], x["line"]))

    out: list[SuggestedTask] = []
    for item in found[:limit]:
        kind = item["kind"]
        msg = item["msg"]
        path = item["path"]
        line_no = item["line"]
        line_text = item["line_text"]

        out.append(
            SuggestedTask(
                title=f"{kind}: {msg}",
                notes=(
                    f"Found {kind} in `{path}` line ~{line_no}:\n\n"
                    f"{line_text}\n\n"
                    "Starter (2–5 min): Open the file at the linked line, understand the intent, and locate related callsites.\n"
                    "DoD: The marker is resolved (or converted into a real ticket/reference), behavior is correct, "
                    "and tests are updated if needed."
                ),
                link=_repo_link(snap, path, line_no),
                tags=f"repo,autogen,code-signal,{kind.lower()}",
                priority=_priority_for_kind(kind),
                estimated_minutes=_minutes_for_kind(kind),
                blocks_me=_blocks_for_kind(kind),
                starter="Open the file at the linked line; reproduce/understand in 2–5 minutes.",
                dod="Marker resolved; correct behavior; test added/updated when applicable.",
            )
        )

    return out


# -------------------------
# ✅ NEW: Finding-driven retrieval → LLM prompt
# -------------------------

def _finding_query(f: RepoFinding) -> str:
    """
    Turn a finding into a retrieval query.
    """
    parts = [
        f.title or "",
        f.category or "",
        f.path or "",
        (f.evidence or "")[:180],
        (f.recommendation or "")[:180],
    ]
    q = " ".join(p for p in parts if p).strip()
    return q[:600]


def _format_finding_block(f: RepoFinding) -> str:
    return (
        f"[FINDING]\n"
        f"id={f.id} severity={f.severity} category={f.category}\n"
        f"path={f.path} line={f.line}\n"
        f"title={f.title}\n"
        f"evidence={(f.evidence or '')[:400]}\n"
        f"recommendation={(f.recommendation or '')[:400]}\n"
        f"acceptance={(getattr(f, 'acceptance', None) or '')[:300]}\n"
    )


async def _llm_generate_tasks_from_findings_with_retrieval(
    db: Session,
    snapshot: RepoSnapshot,
    *,
    project: str,
    max_findings: int = 10,
    chunks_per_finding: int = 3,
) -> list[SuggestedTask]:
    """
    Generates tasks using:
      findings → query → retrieve chunks → send excerpts to LLM

    Important: this uses your existing generate_repo_tasks_json() contract by creating
    synthetic "file_summaries" with excerpt text drawn from chunks.

    If chunks do not exist yet, it raises and caller will fall back cleanly.
    """
    if not getattr(settings, "LLM_ENABLED", False):
        return []
    if not getattr(settings, "OPENAI_MODEL", None):
        return []

    # We need chunks to exist for RAG.
    chunk_count = db.scalar(select(RepoChunk.id).where(RepoChunk.snapshot_id == snapshot.id).limit(1))
    if chunk_count is None:
        raise RuntimeError("No RepoChunk rows found. Run /debug/repo/chunks/build first.")

    findings = db.scalars(
        select(RepoFinding)
        .where(RepoFinding.snapshot_id == snapshot.id)
        .where(RepoFinding.is_resolved == False)  # noqa: E712
        .order_by(RepoFinding.severity.desc(), RepoFinding.id.desc())
        .limit(max_findings)
    ).all()

    if not findings:
        return []

    # Build synthetic evidence “files”
    file_summaries: list[dict[str, Any]] = []
    for f in findings:
        q = _finding_query(f)
        search = search_chunks(
            db=db,
            snapshot_id=snapshot.id,
            query=q,
            top_k=chunks_per_finding,
            mode="auto",
            path_contains=f.path,  # keep it scoped unless path is empty
        )
        chunks = search.get("results", []) or []
        if not chunks:
            # Try a second time without path constraint
            search = search_chunks(db=db, snapshot_id=snapshot.id, query=q, top_k=chunks_per_finding, mode="auto")
            chunks = search.get("results", []) or []

        finding_header = _format_finding_block(f)

        for c in chunks:
            excerpt = f"{finding_header}\n[CODE_CHUNK]\npath={c.get('path')} lines={c.get('start_line')}-{c.get('end_line')}\n{c.get('chunk_text','')}"
            excerpt = excerpt[: int(getattr(settings, "REPO_TASK_EXCERPT_CHARS", 1200))]

            sig = count_markers_in_text(excerpt)
            file_summaries.append(
                {
                    "path": str(c.get("path") or f.path or "unknown"),
                    "excerpt": excerpt,
                    **sig,
                }
            )

    # Also provide global signal counts (stable)
    files = db.scalars(
        select(RepoFile)
        .where(RepoFile.snapshot_id == snapshot.id)
        .where(RepoFile.content_kind == "text")
    ).all()
    signal_counts = compute_signal_counts_for_files(files)

    raw = await generate_repo_tasks_json(
        repo_name=snapshot.repo,
        branch=snapshot.branch,
        commit_sha=snapshot.commit_sha,
        snapshot_id=snapshot.id,
        signal_counts=signal_counts,
        file_summaries=file_summaries,
    )

    # generate_repo_tasks_json returns dict; if it ever returns str, handle leniently
    if isinstance(raw, str):
        raw = json.loads(raw)

    out: list[SuggestedTask] = []
    for t in (raw or {}).get("tasks", []):
        path = str(t.get("path") or "").strip() or "unknown"
        line = t.get("line")
        link = _repo_link(snapshot, path, int(line) if isinstance(line, int) else None)
        out.append(
            SuggestedTask(
                title=str(t.get("title", "")).strip()[:240] or "Untitled repo task",
                notes=str(t.get("notes", "")).strip(),
                link=link,
                tags=str(t.get("tags", "repo,autogen")).strip(),
                priority=int(t.get("priority", 3)) if str(t.get("priority", "3")).isdigit() else 3,
                estimated_minutes=int(t.get("estimated_minutes", 60))
                if str(t.get("estimated_minutes", "60")).isdigit()
                else 60,
                blocks_me=bool(t.get("blocks_me", False)),
                starter=str(t.get("starter", "")).strip() or None,
                dod=str(t.get("dod", "")).strip() or None,
            )
        )
    return out


# -------------------------
# DB materialization helpers
# -------------------------

def _write_tasks_deduped(
    db: Session,
    *,
    snapshot: RepoSnapshot,
    project: str,
    suggestions: list[SuggestedTask],
    limit: int,
) -> tuple[int, int]:
    existing = db.scalars(select(Task).where(Task.project == project)).all()
    existing_keys = {_task_key(t.title or "", t.link) for t in existing}

    created = 0
    skipped = 0

    for s in suggestions[:limit]:
        key = _task_key(s.title, s.link)
        if key in existing_keys:
            skipped += 1
            continue

        db.add(
            Task(
                title=s.title,
                notes=s.notes,
                project=project,
                tags=s.tags,
                link=s.link,
                priority=s.priority,
                estimated_minutes=s.estimated_minutes,
                blocks_me=s.blocks_me,
                starter=s.starter,
                dod=s.dod,
                completed=False,
            )
        )
        existing_keys.add(key)
        created += 1

    db.commit()
    return created, skipped


# -------------------------
# Compatibility entrypoint expected by routes/repo.py
# -------------------------

def tasks_from_findings(db: Session, snapshot_id: int, project: str) -> dict[str, Any]:
    """
    Preferred behavior:
      1) If LLM enabled AND chunks exist → generate tasks using finding-driven retrieval (better tasks).
      2) Otherwise → fall back to canonical deterministic conversion (repo_llm_findings.tasks_from_findings)
    """
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        return {"snapshot_id": snapshot_id, "count": 0, "created": 0, "skipped": 0, "mode": "missing_snapshot"}

    # Try new retrieval-driven mode first (best quality)
    suggestions: list[SuggestedTask] = []
    mode_used = "fallback"

    if bool(getattr(settings, "LLM_ENABLED", False)) and bool(getattr(settings, "OPENAI_MODEL", None)):
        try:
            # NOTE: sync endpoint; we run async LLM in a simple event loop-less way by delegating to repo_llm_findings fallback
            # if environment doesn't support await here.
            #
            # If your FastAPI is async-friendly and you want true async all the way, you can move this call to an async route.
            import asyncio  # local import avoids global dependency issues

            suggestions = asyncio.run(
                _llm_generate_tasks_from_findings_with_retrieval(
                    db=db,
                    snapshot=snap,
                    project=project,
                    max_findings=int(getattr(settings, "REPO_TASK_FINDINGS_MAX", 10)),
                    chunks_per_finding=int(getattr(settings, "REPO_TASK_CHUNKS_PER_FINDING", 3)),
                )
            )
            if suggestions:
                mode_used = "findings_rag_llm"
        except Exception:
            suggestions = []

    if suggestions:
        limit = int(getattr(settings, "REPO_TASK_COUNT", 8))
        created, skipped = _write_tasks_deduped(
            db=db,
            snapshot=snap,
            project=project,
            suggestions=suggestions,
            limit=limit,
        )
        return {
            "snapshot_id": snapshot_id,
            "count": len(suggestions),
            "created": created,
            "skipped": skipped,
            "mode": mode_used,
        }

    # Fall back to canonical implementation (your existing stable conversion)
    from .repo_llm_findings import tasks_from_findings as _impl  # type: ignore

    out = _impl(db=db, snapshot_id=snapshot_id, project=project)
    if isinstance(out, dict):
        out.setdefault("mode", "repo_llm_findings_fallback")
    return out
