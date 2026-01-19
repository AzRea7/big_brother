# backend/app/routes/debug.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select

from ..db import get_db
from ..models import Task
from ..services.planner import generate_daily_plan
from ..services.notifier import send_webhook, send_email
from ..services.security import require_api_key, debug_is_allowed

from ..services.repo_rag import chunk_snapshot, search_chunks, load_chunk_text

router = APIRouter(prefix="/debug", tags=["debug"])


def _guard_debug(request: Request) -> None:
    """
    Enforce production safety:
    - Optionally disable debug endpoints entirely in prod
    - Require X-API-Key for access (at least in prod; configurable in require_api_key)
    """
    if not debug_is_allowed():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")

    require_api_key(request)


router.dependencies.append(Depends(_guard_debug))


class DailyPlanRequest(BaseModel):
    project: str | None = None
    mode: str | None = None


@router.post("/run/daily")
async def run_daily_plan(payload: DailyPlanRequest, db: Session = Depends(get_db)) -> dict:
    mode = (payload.mode or "single").strip().lower()
    project = (payload.project or "onestream").strip().lower()

    plan = await generate_daily_plan(
        db=db,
        focus_project=(project if mode != "split" else None),
        mode=mode,
    )

    msg = plan.content
    await send_webhook(msg)
    send_email(f"Daily Plan — Goal Autopilot ({project})", msg, project=project)
    return {"ok": True, "project": project, "mode": mode}


@router.get("/tasks/recent")
def recent_tasks(limit: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)) -> dict:
    q = select(Task).order_by(Task.id.desc()).limit(limit)
    rows = db.execute(q).scalars().all()
    return {"count": len(rows), "tasks": [t.to_dict() for t in rows]}


@router.post("/send/daily")
async def send_daily(
    project: str | None = Query(default=None),
    mode: str = Query(default="single"),
    db: Session = Depends(get_db),
):
    out = await generate_daily_plan(db=db, focus_project=project, mode=mode)
    subject = f"Daily Plan — Goal Autopilot ({mode}:{project or 'auto'})"
    await send_webhook(out.content)

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


# -----------------------------
# Level 2 RAG (Chunking + Retrieval)
# -----------------------------

@router.post("/repo/chunks/build")
def build_repo_chunks(
    snapshot_id: int = Query(..., ge=1),
    force: bool = Query(False),
    db: Session = Depends(get_db),
):
    """
    Build repo chunks for a snapshot.
    If force=true, it rebuilds from scratch.
    """
    return chunk_snapshot(db, snapshot_id, force=force)


@router.get("/repo/chunks/search")
def rag_search(
    snapshot_id: int = Query(..., ge=1),
    q: str = Query(..., min_length=2),
    top_k: int = Query(16, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return {"snapshot_id": snapshot_id, "query": q, "hits": search_chunks(db, snapshot_id, q, top_k=top_k)}


@router.get("/repo/chunks/get")
def rag_get_chunk(
    snapshot_id: int = Query(..., ge=1),
    path: str = Query(..., min_length=1),
    start_line: int = Query(..., ge=1),
    end_line: int = Query(..., ge=1),
    db: Session = Depends(get_db),
):
    txt = load_chunk_text(db, snapshot_id, path, start_line, end_line)
    return {"snapshot_id": snapshot_id, "path": path, "start_line": start_line, "end_line": end_line, "chunk_text": txt}
