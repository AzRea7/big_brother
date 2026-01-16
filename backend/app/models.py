# backend/app/models.py
from __future__ import annotations

from datetime import datetime, date
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

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

    tasks: Mapped[list["Task"]] = relationship("Task", back_populates="goal")


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

    goal: Mapped[Goal | None] = relationship("Goal", back_populates="tasks")


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
    repo: Mapped[str] = mapped_column(String(255))
    branch: Mapped[str] = mapped_column(String(255))
    commit_sha: Mapped[str | None] = mapped_column(String(80), nullable=True)

    file_count: Mapped[int] = mapped_column(Integer, default=0)
    stored_content_files: Mapped[int] = mapped_column(Integer, default=0)

    warnings_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # ✅ SQLAlchemy needs an FK to infer RepoSnapshot.files join
    files: Mapped[list["RepoFile"]] = relationship(
        "RepoFile",
        back_populates="snapshot",
        cascade="all, delete-orphan",
    )


class RepoFile(Base):
    __tablename__ = "repo_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # ✅ this is the missing piece
    snapshot_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("repo_snapshots.id"),
        index=True,
        nullable=False,
    )

    path: Mapped[str] = mapped_column(String(1024))
    sha: Mapped[str | None] = mapped_column(String(120), nullable=True)
    size: Mapped[int | None] = mapped_column(Integer, nullable=True)

    is_text: Mapped[bool] = mapped_column(Boolean, default=True)
    skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    skip_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # keeping both fields since your migrations/table have "content" and "content_kind"
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_kind: Mapped[str] = mapped_column(String(30), default="skipped")  # text|binary|skipped
    content_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    snapshot: Mapped[RepoSnapshot] = relationship("RepoSnapshot", back_populates="files")
