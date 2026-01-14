from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Task
from ..services.planner import generate_daily_plan
from ..services.notifier import send_webhook, send_email

router = APIRouter(prefix="/debug", tags=["debug"])


@router.post("/run/daily")
async def run_daily(
    project: str | None = Query(default=None),
    mode: str = Query(default="single"),  # single | split
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
    send_email(subject, out.content)
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
