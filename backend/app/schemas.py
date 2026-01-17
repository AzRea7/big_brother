# backend/app/schemas.py
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class GoalCreate(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    why: str | None = None
    target_date: date | None = None
    project: str | None = None


class GoalOut(BaseModel):
    id: int
    title: str
    why: str | None
    target_date: date | None
    project: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


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

    model_config = {"from_attributes": True}


class PlanOut(BaseModel):
    generated_at: datetime
    content: str


class RepoSyncOut(BaseModel):
    snapshot_id: int
    repo: str
    branch: str
    commit_sha: str | None
    file_count: int
    stored_content_files: int
    warnings: list[str] = Field(default_factory=list)


class RepoSignalCountsOut(BaseModel):
    snapshot_id: int
    signals: dict[str, int]


class RepoFindingOut(BaseModel):
    id: int
    snapshot_id: int
    path: str
    line: Optional[int] = None
    category: str
    severity: int
    title: str
    evidence: Optional[str] = None
    recommendation: Optional[str] = None
    acceptance: Optional[str] = None
    fingerprint: str
    created_at: datetime
    is_resolved: bool

    model_config = {"from_attributes": True}


class RepoScanOut(BaseModel):
    snapshot_id: int
    inserted: int
    total_findings: int


class RepoTaskGenOut(BaseModel):
    snapshot_id: int
    generated_at: datetime
    tasks: list[TaskCreate]
