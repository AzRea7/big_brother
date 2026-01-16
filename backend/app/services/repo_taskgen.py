# backend/app/services/repo_taskgen.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoFile, RepoSnapshot, Task


# -------------------------
# Signal extraction (fast, deterministic)
# -------------------------

_TODO_RE = re.compile(r"\b(TODO|FIXME|HACK)\b[:\-\s]*(.*)", re.IGNORECASE)


def compute_signal_counts(db: Session, snapshot_id: int) -> dict[str, int]:
    files = db.scalars(select(RepoFile).where(RepoFile.snapshot_id == snapshot_id)).all()
    total = len(files)

    todo = 0
    fixme = 0
    impl = 0

    for f in files:
        if f.content_kind != "text" or not f.content:
            continue
        # cheap signals
        if "TODO" in f.content:
            todo += 1
        if "FIXME" in f.content:
            fixme += 1
        if any(x in f.content for x in ("pass  #", "raise NotImplementedError", "IMPLEMENT", "stub")):
            impl += 1

    return {
        "total_files": total,
        "files_with_todo": todo,
        "files_with_fixme": fixme,
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
    # keep your style: repo:// links
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
            m = _TODO_RE.search(line)
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
                "DoD: The TODO/FIXME/HACK is resolved, and (if behavior changes) you add/adjust a test."
            )
            tags = "repo,autogen,code-signal," + ("todo" if kind == "TODO" else kind.lower())

            out.append(
                SuggestedTask(
                    title=title,
                    notes=notes,
                    link=_repo_link(snap, f.path, i),
                    tags=tags,
                    priority=4 if kind in ("FIXME", "HACK") else 3,
                    estimated_minutes=45 if kind == "TODO" else 60,
                    blocks_me=(kind in ("FIXME", "HACK")),
                    starter="Open the file at the linked line; reproduce/understand the issue in 2–5 minutes.",
                    dod="The code no longer contains this marker, behavior is correct, and a regression test exists if applicable.",
                )
            )
            if len(out) >= limit:
                return out

    # fallback if no TODO-style signals exist
    if not out:
        out.append(
            SuggestedTask(
                title="Repo scan: identify top 3 high-impact improvements",
                notes=(
                    "No TODO/FIXME/HACK markers were found in the stored snapshot content. "
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
# LLM task generation (optional)
# -------------------------

async def _llm_generate_tasks(snapshot: RepoSnapshot, files: list[RepoFile], project: str) -> list[SuggestedTask]:
    """
    Uses OpenAI-compatible chat endpoint if configured.
    Returns SuggestedTask list.
    """
    if not settings.OPENAI_API_KEY or not settings.OPENAI_MODEL:
        return []

    # keep prompt small: send paths + first N chars
    payload_files: list[dict[str, Any]] = []
    for f in files[:40]:
        txt = (f.content or "")
        payload_files.append(
            {
                "path": f.path,
                "excerpt": txt[:1200],
            }
        )

    system = (
        "You are a senior software engineer generating actionable development tasks from a repo snapshot. "
        "Tasks must be specific, testable, and grounded in provided file excerpts. "
        "Return JSON only."
    )
    user = {
        "repo": snapshot.repo,
        "branch": snapshot.branch,
        "commit_sha": snapshot.commit_sha,
        "project": project,
        "instructions": [
            "Generate 8-15 tasks.",
            "Each task must include: title, notes, tags, priority(1-5), estimated_minutes, blocks_me, path, line(optional), starter, dod.",
            "Tags must include: repo, autogen.",
            "Do not invent files not provided.",
        ],
        "files": payload_files,
    }

    base_url = (settings.OPENAI_BASE_URL or "https://api.openai.com").rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}

    timeout = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            url,
            headers=headers,
            json={
                "model": settings.OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user)},
                ],
                "temperature": 0.2,
            },
        )
        r.raise_for_status()
        data = r.json()

    content = data["choices"][0]["message"]["content"]
    raw = json.loads(content)

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

    # 1) LLM path (if enabled)
    suggestions: list[SuggestedTask] = []
    if settings.LLM_ENABLED:
        try:
            suggestions = await _llm_generate_tasks(snap, files, project=project)
        except Exception:
            # hard fallback to signals
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
