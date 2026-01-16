# backend/app/services/repo_taskgen.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoFile, RepoSnapshot, Task
from ..ai.repo_tasks import count_markers_in_text, generate_repo_tasks_json

# -------------------------
# Signal extraction (fast, deterministic)
# -------------------------

# For deterministic fallback tasks we still detect the “classic” markers.
# (We broaden it slightly so XXX/BUG/NOTE can produce fallback tasks too.)
_FALLBACK_RE = re.compile(r"\b(TODO|FIXME|HACK|XXX|BUG|NOTE)\b[:\-\s]*(.*)", re.IGNORECASE)


def compute_signal_counts(db: Session, snapshot_id: int) -> dict[str, int]:
    files = db.scalars(select(RepoFile).where(RepoFile.snapshot_id == snapshot_id)).all()
    total = len(files)

    files_with = {
        "todo": 0,
        "fixme": 0,
        "hack": 0,
        "xxx": 0,
        "bug": 0,
        "note": 0,
        "dotdotdot": 0,
    }

    # also keep “impl signals” because it’s useful (stubs, NotImplementedError, etc.)
    impl = 0

    for f in files:
        if f.content_kind != "text" or not f.content:
            continue

        sig = count_markers_in_text(f.content)

        if sig["todo_count"] > 0:
            files_with["todo"] += 1
        if sig["fixme_count"] > 0:
            files_with["fixme"] += 1
        if sig["hack_count"] > 0:
            files_with["hack"] += 1
        if sig["xxx_count"] > 0:
            files_with["xxx"] += 1
        if sig["bug_count"] > 0:
            files_with["bug"] += 1
        if sig["note_count"] > 0:
            files_with["note"] += 1
        if sig["dotdotdot_count"] > 0:
            files_with["dotdotdot"] += 1

        if any(x in f.content for x in ("raise NotImplementedError", "IMPLEMENT", "stub", "pass  #")):
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
    }


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


def _repo_link(snapshot: RepoSnapshot, path: str, line: int | None = None) -> str:
    base = f"repo://{snapshot.repo}?branch={snapshot.branch}&commit_sha={snapshot.commit_sha or ''}"
    if line is not None:
        return f"{base}#{path}:L{line}"
    return f"{base}#{path}"


def _suggest_tasks_from_signals(db: Session, snapshot_id: int, project: str, limit: int = 25) -> list[SuggestedTask]:
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        return []

    files = db.scalars(
        select(RepoFile)
        .where(RepoFile.snapshot_id == snapshot_id)
        .where(RepoFile.content_kind == "text")
    ).all()

    out: list[SuggestedTask] = []

    for f in files:
        if not f.content:
            continue

        lines = f.content.splitlines()
        for i, line in enumerate(lines[:2500], start=1):
            m = _FALLBACK_RE.search(line)
            if not m:
                continue

            kind = m.group(1).upper()
            msg = (m.group(2) or "").strip()
            msg = msg[:160] if msg else "Unspecified"

            title = f"{kind}: {msg}"
            notes = (
                f"Found {kind} in `{f.path}` line ~{i}:\n\n"
                f"{line.strip()}\n\n"
                "Starter (2 min): Open the file and locate this line.\n"
                "DoD: The marker is resolved, and (if behavior changes) you add/adjust a test."
            )
            tags = "repo,autogen,code-signal," + kind.lower()

            out.append(
                SuggestedTask(
                    title=title,
                    notes=notes,
                    link=_repo_link(snap, f.path, i),
                    tags=tags,
                    priority=4 if kind in ("FIXME", "HACK", "BUG") else 3,
                    estimated_minutes=45 if kind in ("TODO", "NOTE") else 60,
                    blocks_me=(kind in ("FIXME", "HACK", "BUG")),
                    starter="Open the file at the linked line; reproduce/understand the issue in 2–5 minutes.",
                    dod="The code no longer contains this marker, behavior is correct, and a regression test exists if applicable.",
                )
            )
            if len(out) >= limit:
                return out

    if not out:
        out.append(
            SuggestedTask(
                title="Repo scan: identify top 3 high-impact improvements",
                notes=(
                    "No TODO/FIXME/HACK/XXX/BUG/NOTE markers were found in the stored snapshot content. "
                    "Do a quick scan of: API entrypoints, auth, error handling, and tests to propose 3 improvements."
                ),
                link=_repo_link(snap, "onehaven/"),
                tags="repo,autogen,scan",
                priority=3,
                estimated_minutes=60,
                blocks_me=False,
                starter="Skim /main.py, /routes, and test folder to find weak spots.",
                dod="3 concrete tasks exist with repo links and acceptance criteria.",
            )
        )

    return out


# -------------------------
# LLM task generation (bounded + high-signal)
# -------------------------

async def _llm_generate_tasks(snapshot: RepoSnapshot, files: list[RepoFile], project: str) -> list[SuggestedTask]:
    """
    Uses the *bounded* prompt builder (top N by signal strength, capped excerpts, capped total chars).
    This prevents local-model context overflows (your LM Studio n_ctx issue).
    """
    if not settings.LLM_ENABLED:
        return []

    # If the LLM client is configured to talk to an OpenAI-compatible endpoint (LM Studio),
    # it should work even if OPENAI_API_KEY is a dummy value, depending on your server.
    if not settings.OPENAI_MODEL:
        return []

    # Build per-file summaries with signal counts + excerpts.
    # IMPORTANT: we do NOT push “40 files x 1200 chars” anymore.
    file_summaries: list[dict[str, Any]] = []
    for f in files:
        if f.content_kind != "text" or not f.content:
            continue

        sig = count_markers_in_text(f.content)

        file_summaries.append(
            {
                "path": f.path,
                # send a larger raw excerpt here; the prompt builder will trim to 400–800
                "excerpt": (f.content or "")[:4000],
                **sig,
            }
        )

    signal_counts = compute_signal_counts_for_files(files)

    # Call the bounded repo-task generator
    raw = await generate_repo_tasks_json(
        repo_name=snapshot.repo,
        branch=snapshot.branch,
        commit_sha=snapshot.commit_sha,
        snapshot_id=snapshot.id,
        signal_counts=signal_counts,
        file_summaries=file_summaries,
    )

    tasks: list[SuggestedTask] = []
    for t in raw.get("tasks", []):
        path = t.get("path") or "unknown"
        line = t.get("line")
        link = _repo_link(snapshot, path, int(line) if isinstance(line, int) else None)

        tasks.append(
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

    return tasks


def compute_signal_counts_for_files(files: list[RepoFile]) -> dict[str, int]:
    """
    Same as compute_signal_counts(), but works from an in-memory RepoFile list.
    Useful for LLM path where we already loaded files.
    """
    total = len(files)

    files_with = {
        "todo": 0,
        "fixme": 0,
        "hack": 0,
        "xxx": 0,
        "bug": 0,
        "note": 0,
        "dotdotdot": 0,
    }
    impl = 0

    for f in files:
        if f.content_kind != "text" or not f.content:
            continue

        sig = count_markers_in_text(f.content)

        if sig["todo_count"] > 0:
            files_with["todo"] += 1
        if sig["fixme_count"] > 0:
            files_with["fixme"] += 1
        if sig["hack_count"] > 0:
            files_with["hack"] += 1
        if sig["xxx_count"] > 0:
            files_with["xxx"] += 1
        if sig["bug_count"] > 0:
            files_with["bug"] += 1
        if sig["note_count"] > 0:
            files_with["note"] += 1
        if sig["dotdotdot_count"] > 0:
            files_with["dotdotdot"] += 1

        if any(x in f.content for x in ("raise NotImplementedError", "IMPLEMENT", "stub", "pass  #")):
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
    }


# -------------------------
# Materialization (DB writes + dedupe)
# -------------------------

def _task_key(title: str, link: str | None) -> str:
    return (title or "").strip().lower() + "|" + (link or "")


async def generate_tasks_from_snapshot(db: Session, snapshot_id: int, project: str = "haven") -> tuple[int, int]:
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        return 0, 0

    files = db.scalars(
        select(RepoFile)
        .where(RepoFile.snapshot_id == snapshot_id)
        .where(RepoFile.content_kind == "text")
    ).all()

    # 1) LLM path (bounded)
    suggestions: list[SuggestedTask] = []
    if settings.LLM_ENABLED:
        try:
            suggestions = await _llm_generate_tasks(snap, files, project=project)
        except Exception:
            suggestions = []

    # 2) deterministic fallback
    if not suggestions:
        suggestions = _suggest_tasks_from_signals(db, snapshot_id=snapshot_id, project=project, limit=25)

    # 3) dedupe vs existing tasks in this project
    existing = db.scalars(select(Task).where(Task.project == project)).all()
    existing_keys = {_task_key(t.title or "", t.link) for t in existing}

    created = 0
    skipped = 0

    for s in suggestions:
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
