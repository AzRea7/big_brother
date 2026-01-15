# backend/app/models.py
from __future__ import annotations

from datetime import datetime, date
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class Goal(Base):
    __tablename__ = "goals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    why: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    target_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # lane/project for goal filtering
    project: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # haven | onestream | personal


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    goal_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("goals.id"), nullable=True)

    title: Mapped[str] = mapped_column(String(250), nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    due_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    priority: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    estimated_minutes: Mapped[int] = mapped_column(Integer, default=30, nullable=False)
    blocks_me: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # task metadata
    project: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    link: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    starter: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    dod: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # precision scoring
    impact_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    confidence: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    energy: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    # microtasks
    parent_task_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class TaskDependency(Base):
    __tablename__ = "task_dependencies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[int] = mapped_column(Integer, nullable=False)
    depends_on_task_id: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class PlanRun(Base):
    __tablename__ = "plan_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    top_task_ids: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)


# ---------------------------
# Level 1: GitHub repo snapshots
# ---------------------------

class RepoSnapshot(Base):
    __tablename__ = "repo_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    repo: Mapped[str] = mapped_column(String(200), nullable=False)     # "AzRea7/OneHaven"
    branch: Mapped[str] = mapped_column(String(80), nullable=False)    # "main"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Optional metadata (depends on endpoints used)
    commit_sha: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)

    # quick stats
    file_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    stored_content_files: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # store sync warnings/errors without failing the entire app
    warnings_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class RepoFile(Base):
    __tablename__ = "repo_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    snapshot_id: Mapped[int] = mapped_column(Integer, ForeignKey("repo_snapshots.id"), nullable=False)

    path: Mapped[str] = mapped_column(String(600), nullable=False)  # "onehaven/backend/app/main.py"
    sha: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Content is optional (only for small-ish text files)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # "text", "binary", "skipped"
    content_kind: Mapped[str] = mapped_column(String(16), default="skipped", nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
