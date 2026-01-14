from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from ..db import get_db
from ..models import Task
from ..schemas import TaskCreate, TaskOut, TaskUpdate

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", response_model=TaskOut)
def create_task(payload: TaskCreate, db: Session = Depends(get_db)) -> TaskOut:
    t = Task(**payload.model_dump())
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@router.get("", response_model=list[TaskOut])
def list_tasks(
    include_completed: bool = Query(default=False),
    db: Session = Depends(get_db),
) -> list[TaskOut]:
    stmt = select(Task)
    if not include_completed:
        stmt = stmt.where(Task.completed == False)  # noqa: E712
    return list(db.scalars(stmt).all())


@router.patch("/{task_id}", response_model=TaskOut)
def update_task(task_id: int, payload: TaskUpdate, db: Session = Depends(get_db)) -> TaskOut:
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")

    data = payload.model_dump(exclude_unset=True)

    # handle completed toggle
    if "completed" in data:
        new_val = bool(data["completed"])
        if new_val and not t.completed:
            t.completed = True
            t.completed_at = datetime.utcnow()
        elif (not new_val) and t.completed:
            t.completed = False
            t.completed_at = None
        data.pop("completed", None)

    for k, v in data.items():
        setattr(t, k, v)

    db.commit()
    db.refresh(t)
    return t


@router.post("/{task_id}/complete", response_model=TaskOut)
def complete_task(task_id: int, db: Session = Depends(get_db)) -> TaskOut:
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    if not t.completed:
        t.completed = True
        t.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(t)
    return t


@router.post("/{task_id}/reopen", response_model=TaskOut)
def reopen_task(task_id: int, db: Session = Depends(get_db)) -> TaskOut:
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    if t.completed:
        t.completed = False
        t.completed_at = None
        db.commit()
        db.refresh(t)
    return t
