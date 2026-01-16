from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class GoalCreate(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    why: str | None = None
    target_date: date | None = None
    project: str | None = None  # haven | onestream | personal


class GoalOut(BaseModel):
    id: int
    title: str
    why: str | None
    target_date: date | None
    project: str | None
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

class RepoSyncOut(BaseModel):
    snapshot_id: int
    repo: str
    branch: str
    commit_sha: Optional[str] = None
    file_count: int
    stored_content_files: int
    warnings: list[str] = []


class RepoSignalCountsOut(BaseModel):
    snapshot_id: int
    total_files: int
    files_with_todo: int
    files_with_fixme: int
    files_with_impl_signals: int


class RepoTaskGenOut(BaseModel):
    snapshot_id: int
    created_tasks: int
    skipped_duplicates: int