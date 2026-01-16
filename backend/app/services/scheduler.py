# backend/app/services/scheduler.py
from __future__ import annotations

import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..config import settings
from ..db import SessionLocal
from .planner import generate_daily_plan
from .notifier import send_webhook, send_email
from .github_sync import create_repo_snapshot
from .repo_taskgen import generate_tasks_from_snapshot

_scheduler: BackgroundScheduler | None = None


def start_scheduler() -> None:
    global _scheduler
    if _scheduler:
        return

    _scheduler = BackgroundScheduler(timezone="America/Detroit")

    # Daily Plan
    _scheduler.add_job(
        _run_daily_plan_job,
        CronTrigger(hour=settings.DAILY_PLAN_HOUR, minute=settings.DAILY_PLAN_MINUTE),
        id="daily_plan",
        replace_existing=True,
    )

    # Midday Nudge
    _scheduler.add_job(
        _run_midday_nudge_job,
        CronTrigger(hour=settings.MIDDAY_NUDGE_HOUR, minute=settings.MIDDAY_NUDGE_MINUTE),
        id="midday_nudge",
        replace_existing=True,
    )

    # Repo -> tasks pipeline (Haven)
    _scheduler.add_job(
        _run_repo_pipeline_job,
        CronTrigger(hour=settings.REPO_SYNC_HOUR, minute=settings.REPO_SYNC_MINUTE),
        id="repo_pipeline",
        replace_existing=True,
    )

    _scheduler.start()


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def _run_daily_plan_job() -> None:
    asyncio.run(_async_send(subject_prefix="Daily Plan"))


def _run_midday_nudge_job() -> None:
    asyncio.run(_async_send(subject_prefix="Midday Nudge"))


def _run_repo_pipeline_job() -> None:
    asyncio.run(_async_repo_pipeline())


async def _async_send(subject_prefix: str) -> None:
    db = SessionLocal()
    try:
        mode = settings.DAILY_PLAN_MODE.strip().lower()
        project = settings.DAILY_PLAN_PROJECT.strip().lower()

        plan = await generate_daily_plan(
            db=db,
            focus_project=(project if mode != "split" else None),
            mode=mode,
        )

        if mode == "split":
            subject = f"{subject_prefix} — Goal Autopilot (split)"
            ui_link_project = "onestream"
        else:
            subject = f"{subject_prefix} — Goal Autopilot ({project})"
            ui_link_project = project

        msg = plan.content

        await send_webhook(msg)
        send_email(subject, msg, project=ui_link_project)
    finally:
        db.close()


async def _async_repo_pipeline() -> None:
    """
    1) snapshot repo
    2) generate tasks (LLM if enabled)
    3) store tasks to /tasks?project=haven (repo-only)
    """
    db = SessionLocal()
    try:
        project = (settings.REPO_SYNC_PROJECT or "haven").strip().lower()

        res = await create_repo_snapshot(db=db, repo=None, branch=None)
        created, skipped = await generate_tasks_from_snapshot(db=db, snapshot_id=res.snapshot_id, project=project)

        msg = f"Repo pipeline ran for {project}: snapshot={res.snapshot_id}, created={created}, skipped_dupes={skipped}"
        await send_webhook(msg)
    finally:
        db.close()
