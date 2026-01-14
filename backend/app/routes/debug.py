from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..planner import generate_daily_plan
from ..models import Task

router = APIRouter(prefix="/debug", tags=["debug"])


@router.post("/run/daily")
async def run_daily(project: str | None = Query(default=None), db: Session = get_db()):
    out = await generate_daily_plan(db=db, focus_project=project)
    return {"generated_at": out.generated_at.isoformat(), "content": out.content}


# If you already implemented /debug/send/daily, keep it; otherwise leave as-is in your codebase.


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
def seed_notes(payload: SeedNotesPayload, db: Session = get_db()):
    updated = 0
    for item in payload.items:
        t = db.get(Task, item.task_id)
        if not t:
            continue
        if item.notes is not None:
            t.notes = item.notes
        if item.starter is not None:
            t.starter = item.starter
        if item.dod is not None:
            t.dod = item.dod
        if item.link is not None:
            t.link = item.link
        if item.tags is not None:
            t.tags = item.tags
        if item.project is not None:
            t.project = item.project
        if item.impact_score is not None:
            t.impact_score = item.impact_score
        if item.confidence is not None:
            t.confidence = item.confidence
        if item.energy is not None:
            t.energy = item.energy
        updated += 1

    db.commit()
    return {"ok": True, "updated": updated}
