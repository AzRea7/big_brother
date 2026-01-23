# backend/app/services/repo_taskgen.py
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
    from .code_signals import compute_signal_counts_for_files, count_markers_in_text
except Exception:
    # Fallback older location
    from .repo_findings import compute_signal_counts_for_files, count_markers_in_text  # type: ignore

# LLM task generator
from ..ai.repo_tasks import build_prompt as build_repo_tasks_prompt
from ..ai.llm import chat_completion_json

# LLM config helper (kept as before)
from .repo_findings import _repo_link  # type: ignore


@dataclass(frozen=True)
class SuggestedTask:
    title: str
    notes: str
    link: Optional[str]
    tags: str
    priority: int
    estimated_minutes: int
    blocks_me: bool
    starter: Optional[str] = None
    dod: Optional[str] = None


def _task_key(title: str, link: Optional[str]) -> str:
    return f"{(title or '').strip().lower()}::{(link or '').strip().lower()}"


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _clean_text(s: Any, max_len: int) -> str:
    return str(s or "").strip()[:max_len]


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


def _finding_queries(f: RepoFinding) -> tuple[str, str]:
    """Return (exact_query, semantic_query).

    - exact_query: high precision (path + title keywords)
    - semantic_query: wider recall (category + recommendation keywords)

    This keeps FTS useful even before embeddings are deployed everywhere.
    """
    title = str(f.title or "").strip()
    category = str(f.category or "").strip()
    path = str(f.path or "").strip()
    rec = str(f.recommendation or "").strip()
    ev = str(f.evidence or "").strip()

    exact_parts = [path, title]
    exact_q = " ".join(p for p in exact_parts if p).strip()[:400]

    semantic_parts = [category, title, rec[:180], ev[:120]]
    semantic_q = " ".join(p for p in semantic_parts if p).strip()[:600]

    # Ensure neither query is empty if we have *something*.
    if not exact_q:
        exact_q = semantic_q[:400]
    if not semantic_q:
        semantic_q = exact_q[:600]

    return exact_q, semantic_q


def _format_finding_block(f: RepoFinding) -> str:
    return (
        f"[FINDING]\n"
        f"id={int(f.id)}\n"
        f"category={_clean_text(f.category, 60)}\n"
        f"severity={_safe_int(f.severity, 3)}\n"
        f"title={_clean_text(f.title, 240)}\n"
        f"path={_clean_text(f.path, 400)} line={f.line if isinstance(f.line, int) else 'null'}\n"
        f"evidence={_clean_text(f.evidence, 400)}\n"
        f"recommendation={_clean_text(f.recommendation, 400)}\n"
        f"acceptance={_clean_text(f.acceptance, 400)}\n"
    )


async def generate_repo_tasks_json(
    *,
    repo_name: str,
    branch: str,
    commit_sha: str | None,
    snapshot_id: int,
    signal_counts: dict[str, Any],
    file_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt = build_repo_tasks_prompt(
        repo_name=repo_name,
        branch=branch,
        commit_sha=commit_sha,
        snapshot_id=snapshot_id,
        signal_counts=signal_counts,
        file_summaries=file_summaries,
    )
    raw = await chat_completion_json(
        system="",
        user=prompt,
        model=str(getattr(settings, "OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"),
        max_tokens=int(getattr(settings, "REPO_TASK_MAX_TOKENS", 1000)),
        temperature=float(getattr(settings, "REPO_TASK_TEMPERATURE", 0.2)),
    )
    return raw if isinstance(raw, dict) else {"tasks": []}


def search_chunks(
    db: Session,
    snapshot_id: int,
    *,
    query: str,
    top_k: int = 10,
    mode: str = "auto",
    path_contains: Optional[str] = None,
) -> dict[str, Any]:
    """Thin wrapper over services.repo_chunks.search_chunks().

    This keeps older callers stable while making the retrieval provider interface
    centralized (FTS + optional embeddings).
    """
    from .repo_chunks import search_chunks as _search  # canonical provider

    q = (query or "").strip()
    if len(q) < 2:
        return {"snapshot_id": snapshot_id, "query": query, "mode_used": "none", "results": []}

    top_k = max(1, min(int(top_k), 30))

    # repo_chunks.search_chunks is async (because embeddings can be async).
    import asyncio

    try:
        mode_used, hits = asyncio.run(
            _search(
                db,
                snapshot_id=snapshot_id,
                query=q,
                top_k=top_k,
                mode=mode,
                path_contains=path_contains,
            )
        )
    except RuntimeError:
        # If we're already inside an event loop (e.g., called from async context),
        # fall back to creating a task on that loop.
        loop = asyncio.get_event_loop()
        mode_used, hits = loop.run_until_complete(
            _search(
                db,
                snapshot_id=snapshot_id,
                query=q,
                top_k=top_k,
                mode=mode,
                path_contains=path_contains,
            )
        )

    results: list[dict[str, Any]] = []
    for h in hits:
        results.append(
            {
                "id": int(h.id),
                "snapshot_id": int(h.snapshot_id),
                "path": str(h.path),
                "start_line": int(h.start_line),
                "end_line": int(h.end_line),
                "chunk_text": str(h.chunk_text or ""),
                "symbols_json": h.symbols_json,
                "score": float(h.score) if h.score is not None else None,
            }
        )

    return {"snapshot_id": snapshot_id, "query": q, "mode_used": str(mode_used), "results": results}


# -------------------------
# Fallback helpers
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
            )
        )
        created += 1
        existing_keys.add(key)

    db.commit()
    return created, skipped


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
    # Use the canonical async provider directly (avoid nested asyncio.run inside async context).
    from .repo_chunks import search_chunks as _search_chunks_async  # canonical provider

    async def _search_dicts(*, q: str, k: int, path_hint: Optional[str]) -> list[dict[str, Any]]:
        mode_used, hits = await _search_chunks_async(
            db,
            snapshot_id=snapshot.id,
            query=q,
            top_k=k,
            mode="auto",
            path_contains=path_hint,
        )
        out: list[dict[str, Any]] = []
        for h in hits:
            out.append(
                {
                    "id": int(h.id),
                    "path": str(h.path),
                    "start_line": int(h.start_line),
                    "end_line": int(h.end_line),
                    "chunk_text": str(h.chunk_text or ""),
                    "symbols_json": h.symbols_json,
                    "score": float(h.score) if h.score is not None else None,
                    "mode_used": str(mode_used),
                }
            )
        return out

    for f in findings:
        exact_q, semantic_q = _finding_queries(f)

        # Retrieval pack (max ~3 chunks): same-file, related-file, fallback.
        merged: list[dict[str, Any]] = []

        def _extend_unique(items: list[dict[str, Any]]) -> None:
            seen = {(c.get("path"), c.get("start_line"), c.get("end_line")) for c in merged}
            for c in items:
                key = (c.get("path"), c.get("start_line"), c.get("end_line"))
                if key in seen:
                    continue
                merged.append(c)
                seen.add(key)
                if len(merged) >= chunks_per_finding:
                    break

        # 1) Same-file precision
        if f.path:
            _extend_unique(await _search_dicts(q=exact_q, k=1, path_hint=f.path))

        # 2) Related-file hint (light heuristic)
        related_hint: Optional[str] = None
        cat = str(f.category or "").lower()
        if cat in {"auth", "security"}:
            related_hint = "auth"
        elif cat in {"reliability", "observability"}:
            related_hint = "middleware"
        elif cat in {"correctness", "testing"}:
            related_hint = "tests"
        elif cat in {"performance"}:
            related_hint = "cache"
        elif cat in {"maintainability"}:
            related_hint = "config"

        if related_hint:
            _extend_unique(await _search_dicts(q=semantic_q, k=1, path_hint=related_hint))

        # 3) Fallback wide recall
        if len(merged) < chunks_per_finding:
            _extend_unique(await _search_dicts(q=semantic_q, k=chunks_per_finding, path_hint=None))

        if not merged:
            continue

        finding_header = _format_finding_block(f)
        finding_tag = f"[FINDING id={int(f.id)} category={str(f.category or 'unknown')} severity={int(f.severity or 3)}]"

        for c in merged[:chunks_per_finding]:
            excerpt = (
                f"{finding_tag}\n{finding_header}\n"
                f"[CODE_CHUNK]\npath={c.get('path')} lines={c.get('start_line')}-{c.get('end_line')} score={c.get('score')}\n"
                f"{c.get('chunk_text','')}"
            )
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
