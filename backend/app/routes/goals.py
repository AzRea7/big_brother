from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from ..db import SessionLocal
from ..models import Goal
from ..schemas import GoalCreate, GoalOut

router = APIRouter(prefix="/goals", tags=["goals"])


@router.post("", response_model=GoalOut)
def create_goal(payload: GoalCreate):
    db: Session = SessionLocal()
    try:
        g = Goal(title=payload.title, why=payload.why, target_date=payload.target_date)
        db.add(g)
        db.commit()
        db.refresh(g)
        return g
    finally:
        db.close()


@router.get("", response_model=list[GoalOut])
def list_goals():
    db: Session = SessionLocal()
    try:
        goals = list(db.scalars(select(Goal).order_by(Goal.created_at.desc())).all())
        return goals
    finally:
        db.close()


@router.post("/{goal_id}/archive", response_model=GoalOut)
def archive_goal(goal_id: int):
    db: Session = SessionLocal()
    try:
        g = db.get(Goal, goal_id)
        if not g:
            raise HTTPException(status_code=404, detail="Goal not found")
        g.is_archived = True
        db.commit()
        db.refresh(g)
        return g
    finally:
        db.close()
