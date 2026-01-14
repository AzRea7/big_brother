from __future__ import annotations

from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class Goal(Base):
    __tablename__ = "goals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    why: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    target_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), nullable=False
    )

    tasks: Mapped[list["Task"]] = relationship("Task", back_populates="goal")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    goal_id: Mapped[Optional[int]] = mapped_column(ForeignKey("goals.id"), nullable=True)
    goal: Mapped[Optional[Goal]] = relationship("Goal", back_populates="tasks")

    title: Mapped[str] = mapped_column(String(250), nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    due_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    priority: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    estimated_minutes: Mapped[int] = mapped_column(Integer, default=30, nullable=False)
    blocks_me: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), nullable=False
    )

    # --- Improvement 1: task metadata ---
    # Use a small enum-ish string; we enforce values in code.
    project: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # "haven"|"onestream"|"personal"
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # "data,backend,demo"
    link: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # URL or repo path
    starter: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 2-min starter, precise
    dod: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # definition of done, precise

    # --- Improvement 3: precision scoring ---
    impact_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)      # 1-5
    confidence: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)       # 1-5
    energy: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)         # "low"|"med"|"high"

    # --- Improvement 4: microtask generation ---
    parent_task_id: Mapped[Optional[int]] = mapped_column(ForeignKey("tasks.id"), nullable=True)
    parent: Mapped[Optional["Task"]] = relationship("Task", remote_side="Task.id", back_populates="children")
    children: Mapped[list["Task"]] = relationship("Task", back_populates="parent", cascade="all, delete-orphan")

    # --- Improvement 2: dependencies (many-to-many via join table) ---
    depends_on: Mapped[list["TaskDependency"]] = relationship(
        "TaskDependency",
        foreign_keys="TaskDependency.task_id",
        cascade="all, delete-orphan",
        back_populates="task",
    )
    required_by: Mapped[list["TaskDependency"]] = relationship(
        "TaskDependency",
        foreign_keys="TaskDependency.depends_on_task_id",
        cascade="all, delete-orphan",
        back_populates="depends_on_task",
    )


class TaskDependency(Base):
    __tablename__ = "task_dependencies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    depends_on_task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), nullable=False)

    task: Mapped[Task] = relationship("Task", foreign_keys=[task_id], back_populates="depends_on")
    depends_on_task: Mapped[Task] = relationship("Task", foreign_keys=[depends_on_task_id], back_populates="required_by")


# --- Improvement 5: plan history ---
class PlanRun(Base):
    __tablename__ = "plan_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), nullable=False
    )
    project: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    top_task_ids: Mapped[str] = mapped_column(Text, nullable=False)  # comma-separated ids
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # later we can compute this automatically
    completed_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
