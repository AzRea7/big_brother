# backend/app/services/code_signals.py
from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import RepoFile, Task


@dataclass
class SuggestedTask:
    title: str
    notes: str
    link: str | None
    tags: str


_TODO_RE = re.compile(r"\b(TODO|FIXME|HACK)\b[:\-\s]*(.*)", re.IGNORECASE)


def suggest_tasks_from_snapshot(db: Session, snapshot_id: int, project: str = "haven", limit: int = 30) -> list[SuggestedTask]:
    files = db.scalars(
        select(RepoFile)
        .where(RepoFile.snapshot_id == snapshot_id)
        .where(RepoFile.content_kind == "text")
    ).all()

    suggestions: list[SuggestedTask] = []

    for f in files:
        if not f.content:
            continue

        lines = f.content.splitlines()
        for i, line in enumerate(lines[:2000]):
            m = _TODO_RE.search(line)
            if not m:
                continue

            kind = m.group(1).upper()
            msg = (m.group(2) or "").strip()
            msg = msg[:160] if msg else "Unspecified"

            title = f"{kind}: {msg}"
            notes = (
                f"Found {kind} in `{f.path}` line ~{i+1}:\n\n"
                f"{line.strip()}\n\n"
                "Starter (2 min): Open the file and locate this line.\n"
                "DoD: The TODO/FIXME is resolved or replaced with a tracked task + tests where needed."
            )
            tags = "code-signal,todo" if kind == "TODO" else "code-signal,fixme"

            suggestions.append(SuggestedTask(title=title, notes=notes, link=None, tags=tags))
            if len(suggestions) >= limit:
                return suggestions

    return suggestions


def materialize_suggestions_as_tasks(
    db: Session,
    snapshot_id: int,
    project: str = "haven",
    limit: int = 30,
    priority: int = 3,
    estimated_minutes: int = 45,
) -> dict[str, int]:
    suggestions = suggest_tasks_from_snapshot(db, snapshot_id=snapshot_id, project=project, limit=limit)

    created = 0
    for s in suggestions:
        db.add(
            Task(
                title=s.title,
                notes=s.notes,
                project=project,
                tags=s.tags,
                priority=priority,
                estimated_minutes=estimated_minutes,
                blocks_me=False,
                completed=False,
            )
        )
        created += 1

    db.commit()
    return {"created": created}
