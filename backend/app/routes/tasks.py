# backend/app/routes/tasks.py
from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Task
from ..schemas import TaskCreate, TaskOut, TaskUpdate
from ..services.planner import generate_daily_plan

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
    project: str | None = Query(default=None),
    include_completed: bool = Query(default=False),
    db: Session = Depends(get_db),
) -> list[TaskOut]:
    stmt = select(Task)
    if project:
        stmt = stmt.where(Task.project == project)
    if not include_completed:
        stmt = stmt.where(Task.completed == False)  # noqa: E712
    return list(db.scalars(stmt).all())


@router.patch("/{task_id}", response_model=TaskOut)
def update_task(task_id: int, payload: TaskUpdate, db: Session = Depends(get_db)) -> TaskOut:
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")

    data = payload.model_dump(exclude_unset=True)

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


@router.post("/{task_id}/complete_and_refresh")
async def complete_and_refresh(
    task_id: int,
    project: str | None = Query(default=None),
    mode: str = Query(default="single"),
    db: Session = Depends(get_db),
):
    """
    Mark task complete, then regenerate the plan so the user gets the next steps instantly.
    """
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")

    if not t.completed:
        t.completed = True
        t.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(t)

    # default to task's project if not provided
    focus_project = project or t.project
    out = await generate_daily_plan(db=db, focus_project=focus_project, mode=mode)
    return {
        "task": TaskOut.model_validate(t).model_dump(),
        "new_plan": {"generated_at": out.generated_at.isoformat(), "content": out.content},
    }


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
