# backend/app/routes/tasks.py
from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from ..db import get_db
from ..config import settings
from ..models import Task
from ..schemas import TaskCreate, TaskOut, TaskUpdate

router = APIRouter(prefix="/tasks", tags=["tasks"])


def _has_repo_tag(tags: str | None) -> bool:
    if not tags:
        return False
    parts = [p.strip() for p in tags.split(",") if p.strip()]
    return "repo" in parts


def _is_haven(project: str | None) -> bool:
    return (project or "").strip().lower() == "haven"


@router.get("", response_model=list[TaskOut])
def list_tasks(
    project: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    stmt = select(Task)
    if project:
        stmt = stmt.where(Task.project == project)

    tasks = db.execute(stmt.order_by(Task.created_at.desc())).scalars().all()

    # Enforce: haven shows only repo tasks
    if settings.HAVEN_REPO_ONLY and _is_haven(project):
        tasks = [t for t in tasks if _has_repo_tag(t.tags) or (t.link or "").startswith("repo://")]

    return tasks


@router.post("", response_model=TaskOut)
def create_task(payload: TaskCreate, db: Session = Depends(get_db)):
    # Enforce: haven cannot be manually polluted
    if settings.HAVEN_REPO_ONLY and _is_haven(payload.project):
        if not _has_repo_tag(payload.tags) and not (payload.link or "").startswith("repo://"):
            raise HTTPException(
                status_code=400,
                detail="HAVEN_REPO_ONLY is enabled: haven tasks must be repo-generated (tag 'repo' or repo:// link).",
            )

    t = Task(
        project=payload.project,
        goal_id=payload.goal_id,
        title=payload.title,
        notes=payload.notes,
        due_date=payload.due_date,
        priority=payload.priority,
        estimated_minutes=payload.estimated_minutes,
        blocks_me=payload.blocks_me,
        tags=payload.tags,
        link=payload.link,
        starter=payload.starter,
        dod=payload.dod,
        impact_score=payload.impact_score,
        confidence=payload.confidence,
        energy=payload.energy,
        parent_task_id=payload.parent_task_id,
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@router.patch("/{task_id}", response_model=TaskOut)
def update_task(task_id: int, payload: TaskUpdate, db: Session = Depends(get_db)):
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")

    for field in [
        "title",
        "goal_id",
        "notes",
        "due_date",
        "priority",
        "estimated_minutes",
        "blocks_me",
        "completed",
        "project",
        "tags",
        "link",
        "starter",
        "dod",
        "impact_score",
        "confidence",
        "energy",
        "parent_task_id",
    ]:
        val = getattr(payload, field, None)
        if val is not None:
            setattr(t, field, val)

    # completed_at bookkeeping
    if payload.completed is True and not t.completed_at:
        t.completed_at = datetime.utcnow()
    if payload.completed is False:
        t.completed_at = None

    db.commit()
    db.refresh(t)
    return t


@router.post("/{task_id}/complete", response_model=TaskOut)
def complete_task(task_id: int, db: Session = Depends(get_db)):
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")

    t.completed = True
    t.completed_at = datetime.utcnow()
    db.commit()
    db.refresh(t)
    return t


@router.post("/{task_id}/reopen", response_model=TaskOut)
def reopen_task(task_id: int, db: Session = Depends(get_db)):
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")

    t.completed = False
    t.completed_at = None
    db.commit()
    db.refresh(t)
    return t


@router.post("/{task_id}/complete_and_refresh", response_model=TaskOut)
def complete_and_refresh(task_id: int, db: Session = Depends(get_db)):
    """
    Keep this endpoint because your OpenAPI shows it exists.
    For now: completes only (UI calls exist). You can later hook it to regenerate plan.
    """
    return complete_task(task_id, db)
