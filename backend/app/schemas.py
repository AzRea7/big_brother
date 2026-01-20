# backend/app/schemas.py
from __future__ import annotations

from datetime import date, datetime
from typing import Optional, List

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


class RepoSnapshotOut(BaseModel):
    id: int
    repo: str
    branch: str
    commit_sha: Optional[str]
    file_count: int
    stored_content_files: int
    created_at: datetime

    class Config:
        from_attributes = True


class RepoChunkOut(BaseModel):
    id: int
    snapshot_id: int
    path: str
    start_line: int
    end_line: int
    chunk_text: str
    symbols_json: Optional[str] = None
    score: Optional[float] = None  # bm25 rank or cosine similarity score depending on mode

    class Config:
        from_attributes = True


class ChunkSearchRequest(BaseModel):
    snapshot_id: int
    query: str
    top_k: int = Field(default=10, ge=1, le=30)
    mode: str = Field(default="auto", description="auto|fts|embeddings")
    path_contains: Optional[str] = None


class ChunkSearchResponse(BaseModel):
    snapshot_id: int
    query: str
    mode_used: str
    results: List[RepoChunkOut]


# ---- PR workflow ----
class PatchValidateRequest(BaseModel):
    patch_text: str
    snapshot_id: int


class PatchValidateResponse(BaseModel):
    valid: bool
    error: Optional[str] = None
    files_changed: int = 0
    lines_changed: int = 0


class PatchApplyRequest(BaseModel):
    snapshot_id: int
    patch_text: str
    run_tests: bool = True


class PatchApplyResponse(BaseModel):
    run_id: int
    valid: bool
    validation_error: Optional[str]
    applied: bool
    apply_error: Optional[str]
    tests_ran: bool
    tests_ok: bool
    test_output: Optional[str]


class PatchOpenPRRequest(BaseModel):
    run_id: int
    title: str = Field(min_length=4, max_length=300)
    body: Optional[str] = None
    base_branch: Optional[str] = None  # default: snapshot.branch


class PatchOpenPRResponse(BaseModel):
    ok: bool
    pr_url: Optional[str] = None
    pr_number: Optional[int] = None
    error: Optional[str] = None


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
