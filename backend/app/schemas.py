from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class GoalCreate(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    why: Optional[str] = None
    target_date: Optional[date] = None


class GoalOut(BaseModel):
    id: int
    title: str
    why: Optional[str]
    target_date: Optional[date]
    is_archived: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TaskCreate(BaseModel):
    title: str = Field(min_length=2, max_length=250)
    goal_id: Optional[int] = None
    notes: Optional[str] = None
    due_date: Optional[date] = None
    priority: int = Field(default=3, ge=1, le=5)
    estimated_minutes: int = Field(default=30, ge=5, le=480)
    blocks_me: bool = False

    # new
    project: Optional[str] = None
    tags: Optional[str] = None
    link: Optional[str] = None
    starter: Optional[str] = None
    dod: Optional[str] = None
    impact_score: Optional[int] = Field(default=None, ge=1, le=5)
    confidence: Optional[int] = Field(default=None, ge=1, le=5)
    energy: Optional[str] = None
    parent_task_id: Optional[int] = None


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=2, max_length=250)
    goal_id: Optional[int] = None
    notes: Optional[str] = None
    due_date: Optional[date] = None
    priority: Optional[int] = Field(default=None, ge=1, le=5)
    estimated_minutes: Optional[int] = Field(default=None, ge=5, le=480)
    blocks_me: Optional[bool] = None
    completed: Optional[bool] = None

    # new
    project: Optional[str] = None
    tags: Optional[str] = None
    link: Optional[str] = None
    starter: Optional[str] = None
    dod: Optional[str] = None
    impact_score: Optional[int] = Field(default=None, ge=1, le=5)
    confidence: Optional[int] = Field(default=None, ge=1, le=5)
    energy: Optional[str] = None
    parent_task_id: Optional[int] = None


class TaskOut(BaseModel):
    id: int
    goal_id: Optional[int]
    title: str
    notes: Optional[str]
    due_date: Optional[date]
    priority: int
    estimated_minutes: int
    blocks_me: bool
    completed: bool
    completed_at: Optional[datetime]
    created_at: datetime

    # new
    project: Optional[str]
    tags: Optional[str]
    link: Optional[str]
    starter: Optional[str]
    dod: Optional[str]
    impact_score: Optional[int]
    confidence: Optional[int]
    energy: Optional[str]
    parent_task_id: Optional[int]

    class Config:
        from_attributes = True


class PlanOut(BaseModel):
    generated_at: datetime
    content: str
