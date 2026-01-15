# backend/app/services/scheduler.py
from __future__ import annotations

import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..config import settings
from ..db import SessionLocal
from .planner import generate_daily_plan
from .notifier import send_webhook, send_email


_scheduler: BackgroundScheduler | None = None


def start_scheduler() -> None:
    global _scheduler
    if _scheduler:
        return

    _scheduler = BackgroundScheduler(timezone=settings.TZ)

    _scheduler.add_job(
        func=_run_daily_plan_job,
        trigger=CronTrigger(hour=settings.DAILY_PLAN_HOUR, minute=settings.DAILY_PLAN_MINUTE),
        id="daily_plan",
        replace_existing=True,
    )

    _scheduler.add_job(
        func=_run_midday_nudge_job,
        trigger=CronTrigger(hour=settings.MIDDAY_NUDGE_HOUR, minute=settings.MIDDAY_NUDGE_MINUTE),
        id="midday_nudge",
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
            ui_link_project = "onestream"  # default landing
        else:
            subject = f"{subject_prefix} — Goal Autopilot ({project})"
            ui_link_project = project

        msg = plan.content

        await send_webhook(msg)
        send_email(subject, msg, project=ui_link_project)

    finally:
        db.close()
