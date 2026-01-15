# backend/app/routes/debug.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select

from ..db import get_db
from ..models import Task
from ..services.planner import generate_daily_plan
from ..services.notifier import send_webhook, send_email

router = APIRouter(prefix="/debug", tags=["debug"])


@router.post("/run/daily")
async def run_daily(
    project: str | None = Query(default=None),
    mode: str = Query(default="single"),
    db: Session = Depends(get_db),
):
    out = await generate_daily_plan(db=db, focus_project=project, mode=mode)
    return {"generated_at": out.generated_at.isoformat(), "content": out.content}


@router.post("/send/daily")
async def send_daily(
    project: str | None = Query(default=None),
    mode: str = Query(default="single"),
    db: Session = Depends(get_db),
):
    out = await generate_daily_plan(db=db, focus_project=project, mode=mode)
    subject = f"Daily Plan — Goal Autopilot ({mode}:{project or 'auto'})"
    await send_webhook(out.content)

    # use project for the UI link in the email; if split, default to onestream
    ui_project = (project or "onestream") if mode != "split" else "onestream"
    send_email(subject, out.content, project=ui_project)

    return {"ok": True, "generated_at": out.generated_at.isoformat()}


class SeedNoteItem(BaseModel):
    task_id: int
    notes: str | None = None
    starter: str | None = None
    dod: str | None = None
    link: str | None = None
    tags: str | None = None
    project: str | None = None
    impact_score: int | None = None
    confidence: int | None = None
    energy: str | None = None


class SeedNotesPayload(BaseModel):
    items: list[SeedNoteItem]


@router.post("/tasks/seed_notes")
def seed_notes(payload: SeedNotesPayload, db: Session = Depends(get_db)):
    updated = 0
    for item in payload.items:
        t = db.get(Task, item.task_id)
        if not t:
            continue

        for field in [
            "notes",
            "starter",
            "dod",
            "link",
            "tags",
            "project",
            "impact_score",
            "confidence",
            "energy",
        ]:
            val = getattr(item, field)
            if val is not None:
                setattr(t, field, val)

        updated += 1

    db.commit()
    return {"ok": True, "updated": updated}


@router.post("/tasks/cleanup_microtasks")
def cleanup_microtasks(db: Session = Depends(get_db)):
    """
    Deletes bad historical junk:
    microtasks whose parent is ALSO a microtask.
    This prevents infinite “MICRO of MICRO” clutter when old data exists.
    """
    tasks = list(db.scalars(select(Task)).all())
    by_id = {t.id: t for t in tasks}

    to_delete = []
    for t in tasks:
        if (t.title or "").startswith("[MICRO]") and t.parent_task_id:
            parent = by_id.get(t.parent_task_id)
            if parent and (parent.title or "").startswith("[MICRO]"):
                to_delete.append(t.id)

    deleted = 0
    for tid in to_delete:
        obj = db.get(Task, tid)
        if obj:
            db.delete(obj)
            deleted += 1

    db.commit()
    return {"ok": True, "deleted": deleted}
