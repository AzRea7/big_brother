from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Task
from ..services.planner import generate_daily_plan  # IMPORTANT: correct module path

router = APIRouter(prefix="/debug", tags=["debug"])


@router.post("/run/daily")
async def run_daily(
    project: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    out = await generate_daily_plan(db=db, focus_project=project)
    return {"generated_at": out.generated_at.isoformat(), "content": out.content}


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

        for field in ["notes", "starter", "dod", "link", "tags", "project", "impact_score", "confidence", "energy"]:
            val = getattr(item, field)
            if val is not None:
                setattr(t, field, val)

        updated += 1

    db.commit()
    return {"ok": True, "updated": updated}
