from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from ..db import get_db
from ..models import Task, TaskDependency
from ..schemas import TaskCreate, TaskOut, TaskUpdate

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", response_model=TaskOut)
def create_task(payload: TaskCreate, db: Session = get_db()):
    t = Task(
        title=payload.title,
        goal_id=payload.goal_id,
        notes=payload.notes,
        due_date=payload.due_date,
        priority=payload.priority,
        estimated_minutes=payload.estimated_minutes,
        blocks_me=payload.blocks_me,

        project=payload.project,
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


@router.get("", response_model=list[TaskOut])
def list_tasks(include_completed: bool = Query(False), db: Session = get_db()):
    stmt = select(Task)
    if not include_completed:
        stmt = stmt.where(Task.completed == False)  # noqa: E712
    rows = list(db.scalars(stmt.order_by(Task.created_at.desc())).all())
    return rows


@router.patch("/{task_id}", response_model=TaskOut)
def update_task(task_id: int, payload: TaskUpdate, db: Session = get_db()):
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")

    data = payload.model_dump(exclude_unset=True)

    # handle completed -> completed_at
    if "completed" in data:
        new_completed = data["completed"]
        if new_completed is True and not t.completed:
            t.completed = True
            t.completed_at = datetime.utcnow()
        elif new_completed is False and t.completed:
            t.completed = False
            t.completed_at = None
        del data["completed"]

    for k, v in data.items():
        setattr(t, k, v)

    db.commit()
    db.refresh(t)
    return t


@router.post("/{task_id}/complete", response_model=TaskOut)
def complete_task(task_id: int, db: Session = get_db()):
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    if not t.completed:
        t.completed = True
        t.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(t)
    return t


@router.post("/{task_id}/uncomplete", response_model=TaskOut)
def uncomplete_task(task_id: int, db: Session = get_db()):
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    if t.completed:
        t.completed = False
        t.completed_at = None
        db.commit()
        db.refresh(t)
    return t


# --- Dependencies API (minimal, but enough to unblock planner intelligence) ---
@router.post("/{task_id}/depends_on/{depends_on_task_id}")
def add_dependency(task_id: int, depends_on_task_id: int, db: Session = get_db()):
    if task_id == depends_on_task_id:
        raise HTTPException(status_code=400, detail="Task cannot depend on itself")

    t = db.get(Task, task_id)
    dep = db.get(Task, depends_on_task_id)
    if not t or not dep:
        raise HTTPException(status_code=404, detail="Task not found")

    exists = db.scalars(
        select(TaskDependency).where(
            TaskDependency.task_id == task_id,
            TaskDependency.depends_on_task_id == depends_on_task_id,
        )
    ).first()
    if exists:
        return {"ok": True, "already": True}

    row = TaskDependency(task_id=task_id, depends_on_task_id=depends_on_task_id)
    db.add(row)
    db.commit()
    return {"ok": True}


@router.delete("/{task_id}/depends_on/{depends_on_task_id}")
def remove_dependency(task_id: int, depends_on_task_id: int, db: Session = get_db()):
    row = db.scalars(
        select(TaskDependency).where(
            TaskDependency.task_id == task_id,
            TaskDependency.depends_on_task_id == depends_on_task_id,
        )
    ).first()
    if not row:
        return {"ok": True, "deleted": False}
    db.delete(row)
    db.commit()
    return {"ok": True, "deleted": True}
