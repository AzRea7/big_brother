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
from ..models import RepoFile, Task


@dataclass
class RepoTask:
    title: str
    notes: str
    starter: str | None
    dod: str | None
    tags: str
    link: str


_TODO_RE = re.compile(r"\b(TODO|FIXME|HACK)\b[:\-\s]*(.*)", re.IGNORECASE)


def compute_signal_counts(db: Session, snapshot_id: int) -> dict[str, int]:
    files = db.scalars(select(RepoFile).where(RepoFile.snapshot_id == snapshot_id)).all()
    total = len(files)
    with_todo = 0
    with_fixme = 0
    with_impl = 0

    impl_signals = ("router", "service", "adapter", "repo", "model", "schema", "migration", "scheduler", "docker", "compose")

    for f in files:
        txt = f.content or ""
        if _TODO_RE.search(txt):
            if "todo" in txt.lower():
                with_todo += 1
            if "fixme" in txt.lower():
                with_fixme += 1
        p = (f.path or "").lower()
        if any(s in p for s in impl_signals):
            with_impl += 1

    return {
        "total_files": total,
        "files_with_todo": with_todo,
        "files_with_fixme": with_fixme,
        "files_with_impl_signals": with_impl,
    }


def _rank_paths(paths: list[str]) -> list[str]:
    """
    Prefer code + architecture files. This is crude but works well.
    """
    def score(p: str) -> int:
        pl = p.lower()
        s = 0
        if "backend" in pl or "/app/" in pl:
            s += 40
        if any(x in pl for x in ["/routes/", "/services/", "/adapters/", "/models", "/schemas", "/db", "dockerfile", "compose", "pyproject", "readme"]):
            s += 30
        if pl.endswith((".py", ".md", ".yml", ".yaml", ".toml")):
            s += 10
        if "test" in pl:
            s += 5
        return s

    return sorted(paths, key=score, reverse=True)


def _build_prompt(repo: str, branch: str, files: list[RepoFile]) -> str:
    max_chars = settings.REPO_TASKGEN_MAX_CHARS_PER_FILE

    blobs: list[str] = []
    for f in files:
        body = (f.content or "")[:max_chars]
        blobs.append(f"---\nFILE: {f.path}\n---\n{body}\n")

    instruction = f"""
You are generating engineering tasks from a code repository snapshot.

Repo: {repo}
Branch: {branch}

Output MUST be valid JSON with this schema:

{{
  "tasks": [
    {{
      "title": "...",
      "notes": "...",
      "starter": "...",
      "dod": "...",
      "tags": "repo,autogen,<other_tags>",
      "link": "repo://{repo}?branch={branch}#<path>"
    }}
  ]
}}

Rules:
- Produce {settings.REPO_TASKGEN_LIMIT} tasks max.
- Every task must include tags containing BOTH "repo" and "autogen".
- Every link must point to a real file path seen below (use #<path>).
- Tasks must be specific, concrete, and testable (mention what to change + where + acceptance checks).
- Prefer tasks that improve correctness, security, performance, observability, and developer UX.
- Avoid vague tasks like "review X" unless you specify exact checks and outputs.
"""

    return instruction.strip() + "\n\n" + "\n".join(blobs)


async def _openai_json(prompt: str) -> dict[str, Any]:
    if not settings.LLM_API_KEY:
        raise RuntimeError("LLM enabled but LLM_API_KEY is missing")

    url = settings.LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a senior staff software engineer. Be precise and practical."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    timeout = httpx.Timeout(connect=10.0, read=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        j = r.json()

    content = j["choices"][0]["message"]["content"]
    return json.loads(content)


def _validate_tasks(tasks: list[dict[str, Any]], valid_paths: set[str], repo: str, branch: str) -> list[RepoTask]:
    out: list[RepoTask] = []

    for t in tasks:
        title = (t.get("title") or "").strip()
        notes = (t.get("notes") or "").strip()
        starter = (t.get("starter") or "").strip() or None
        dod = (t.get("dod") or "").strip() or None
        tags = (t.get("tags") or "").strip()
        link = (t.get("link") or "").strip()

        if not title or not notes or not tags or not link:
            continue
        tl = tags.lower()
        if "repo" not in tl or "autogen" not in tl:
            continue

        prefix = f"repo://{repo}?branch={branch}#"
        if not link.startswith(prefix):
            continue
        path = link.split("#", 1)[-1]
        if path not in valid_paths:
            continue

        out.append(RepoTask(title=title, notes=notes, starter=starter, dod=dod, tags=tags, link=link))

    return out


def _fallback_todo_tasks(db: Session, snapshot_id: int, limit: int) -> list[RepoTask]:
    files = db.scalars(
        select(RepoFile)
        .where(RepoFile.snapshot_id == snapshot_id)
        .where(RepoFile.content_kind == "text")
    ).all()

    tasks: list[RepoTask] = []
    for f in files:
        if not f.content:
            continue
        for i, line in enumerate(f.content.splitlines()[:2000]):
            m = _TODO_RE.search(line)
            if not m:
                continue
            msg = (m.group(2) or "").strip() or "Unspecified"
            msg = msg[:160]
            title = f"{m.group(1).upper()}: {msg}"
            notes = f"Found {m.group(1).upper()} in `{f.path}` line ~{i+1}:\n\n{line.strip()}"
            link = f"repo://snapshot/{snapshot_id}#{f.path}"
            tasks.append(RepoTask(title=title, notes=notes, starter="Open the file and locate the TODO.", dod="TODO resolved or converted into tracked work + tests.", tags="repo,autogen,todo", link=link))
            if len(tasks) >= limit:
                return tasks
    return tasks


async def generate_tasks_from_snapshot(db: Session, snapshot_id: int, project: str = "haven") -> tuple[int, int]:
    """
    Returns (created, skipped_duplicates)
    """
    snap = db.execute(select(RepoSnapshot).where(RepoSnapshot.id == snapshot_id)).scalars().first()
    if not snap:
        raise ValueError(f"snapshot_id={snapshot_id} not found")

    files = db.scalars(
        select(RepoFile)
        .where(RepoFile.snapshot_id == snapshot_id)
        .where(RepoFile.content_kind == "text")
    ).all()

    paths = [f.path for f in files if f.path]
    ranked = _rank_paths(paths)
    pick = ranked[: settings.REPO_TASKGEN_MAX_FILES_IN_PROMPT]
    picked_files = [f for f in files if f.path in set(pick)]

    valid_paths = set(paths)

    repo = snap.repo
    branch = snap.branch

    tasks: list[RepoTask] = []

    if settings.LLM_ENABLED:
        prompt = _build_prompt(repo=repo, branch=branch, files=picked_files)
        try:
            j = await _openai_json(prompt)
            raw_tasks = j.get("tasks", [])
            if isinstance(raw_tasks, list):
                tasks = _validate_tasks(raw_tasks, valid_paths=valid_paths, repo=repo, branch=branch)
        except Exception:
            tasks = []

    if not tasks:
        tasks = _fallback_todo_tasks(db, snapshot_id=snapshot_id, limit=settings.REPO_TASKGEN_LIMIT)

    # De-dupe on (title, project)
    existing = set(
        db.scalars(select(Task.title).where(Task.project == project)).all()
    )

    created = 0
    skipped = 0

    for t in tasks[: settings.REPO_TASKGEN_LIMIT]:
        if t.title in existing:
            skipped += 1
            continue

        db.add(
            Task(
                title=t.title,
                notes=t.notes,
                project=project,
                tags=t.tags,
                link=t.link,
                starter=t.starter,
                dod=t.dod,
                priority=4,
                estimated_minutes=90,
                blocks_me=False,
                completed=False,
            )
        )
        created += 1

    db.commit()
    return created, skipped
