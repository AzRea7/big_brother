from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Literal

from pydantic import BaseModel, Field, conint


ProjectName = Literal["haven", "onestream", "personal"]
Energy = Literal["low", "med", "high"]


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

    priority: conint(ge=1, le=5) = 3
    estimated_minutes: conint(ge=5, le=480) = 30
    blocks_me: bool = False

    # new metadata
    project: Optional[ProjectName] = None
    tags: Optional[str] = None
    link: Optional[str] = None
    starter: Optional[str] = None
    dod: Optional[str] = None

    impact_score: Optional[conint(ge=1, le=5)] = None
    confidence: Optional[conint(ge=1, le=5)] = None
    energy: Optional[Energy] = None

    parent_task_id: Optional[int] = None


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=2, max_length=250)

    goal_id: Optional[int] = None
    notes: Optional[str] = None
    due_date: Optional[date] = None

    priority: Optional[conint(ge=1, le=5)] = None
    estimated_minutes: Optional[conint(ge=5, le=480)] = None
    blocks_me: Optional[bool] = None

    completed: Optional[bool] = None

    # new metadata
    project: Optional[ProjectName] = None
    tags: Optional[str] = None
    link: Optional[str] = None
    starter: Optional[str] = None
    dod: Optional[str] = None

    impact_score: Optional[conint(ge=1, le=5)] = None
    confidence: Optional[conint(ge=1, le=5)] = None
    energy: Optional[Energy] = None

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
