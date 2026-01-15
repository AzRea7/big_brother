from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..ai.llm import LLMClient
from ..ai.prompts import USER_PROMPT_TEMPLATE
from ..models import Goal, PlanRun, Task, TaskDependency
from ..schemas import PlanOut
from .microtasks import generate_microtasks

_ID_RE = re.compile(r"\*\*\[(\d+)\]\*\*")


def _fmt_goals(goals: list[Goal]) -> str:
    if not goals:
        return "- (none)"
    lines: list[str] = []
    for g in goals:
        td = f" target={g.target_date}" if g.target_date else ""
        why = f" | why={g.why}" if g.why else ""
        proj = f" project={g.project}" if g.project else ""
        lines.append(f"- [{g.id}] {g.title}{td}{why}{proj}")
    return "\n".join(lines)


def _fmt_tasks(tasks: list[Task]) -> str:
    if not tasks:
        return "- (none)"

    lines: list[str] = []
    for t in tasks:
        dd = f" due={t.due_date}" if t.due_date else ""
        blk = " blocks_me=true" if t.blocks_me else ""
        proj = f" project={t.project}" if t.project else ""

        meta_bits: list[str] = []
        if t.impact_score is not None:
            meta_bits.append(f"impact={t.impact_score}")
        if t.confidence is not None:
            meta_bits.append(f"confidence={t.confidence}")
        if t.energy:
            meta_bits.append(f"energy={t.energy}")

        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
        tags = f" tags={t.tags}" if t.tags else ""
        link = f" link={t.link}" if t.link else ""
        starter = f" starter={t.starter}" if (t.starter and t.starter.strip()) else ""
        dod = f" dod={t.dod}" if (t.dod and t.dod.strip()) else ""

        lines.append(
            f"- [{t.id}] {t.title} (p={t.priority}, est={t.estimated_minutes}m{dd}{blk}{proj}){meta}{tags}{link}{starter}{dod}"
        )

    return "\n".join(lines)


def _dependency_blocked_ids(db: Session, task_ids: list[int]) -> set[int]:
    # Avoid selecting TaskDependency.* to survive schema drift (sqlite missing created_at etc.)
    if not task_ids:
        return set()

    rows = db.execute(
        select(TaskDependency.task_id, TaskDependency.depends_on_task_id).where(TaskDependency.task_id.in_(task_ids))
    ).all()

    if not rows:
        return set()

    depends_ids = {depends_on_id for (_, depends_on_id) in rows}
    if not depends_ids:
        return set()

    incomplete_dep_ids = {
        t.id
        for t in db.scalars(
            select(Task).where(
                Task.id.in_(list(depends_ids)),
                Task.completed == False,  # noqa: E712
            )
        ).all()
    }

    blocked: set[int] = set()
    for task_id, depends_on_id in rows:
        if depends_on_id in incomplete_dep_ids:
            blocked.add(task_id)

    return blocked


def _deterministic_sort_key(t: Task) -> tuple:
    due = t.due_date.isoformat() if t.due_date else "9999-12-31"
    # Microtasks first in deterministic ordering (they are the glue that unblocks)
    is_micro = 0 if (t.title or "").startswith("[MICRO]") else 1
    return (-t.priority, is_micro, (0 if t.blocks_me else 1), due, t.created_at.isoformat())


def _fetch_lane(db: Session, project: str) -> tuple[list[Goal], list[Task]]:
    goals = list(
        db.scalars(
            select(Goal)
            .where(Goal.project == project)
            .where(Goal.is_archived == False)  # noqa: E712
            .order_by(Goal.created_at.desc())
        ).all()
    )

    tasks = list(
        db.scalars(
            select(Task)
            .where(Task.project == project)
            .where(Task.completed == False)  # noqa: E712
            .order_by(Task.created_at.desc())
        ).all()
    )

    if not tasks:
        return goals, []

    task_ids = [t.id for t in tasks]
    blocked_ids = _dependency_blocked_ids(db, task_ids)

    unblocked = [t for t in tasks if t.id not in blocked_ids]
    blocked = [t for t in tasks if t.id in blocked_ids]

    tasks_sorted = sorted(unblocked, key=_deterministic_sort_key) + sorted(blocked, key=_deterministic_sort_key)
    return goals, tasks_sorted


def _extract_top_ids(content: str, top_n: int) -> list[int]:
    ids = []
    for m in _ID_RE.finditer(content):
        try:
            ids.append(int(m.group(1)))
        except Exception:
            continue
        if len(ids) >= top_n:
            break
    return ids


def _deterministic_plan(pool: list[Task], top_n: int) -> tuple[str, list[Task]]:
    top_tasks = pool[:top_n]
    start_times = ["09:00", "10:00", "11:00"]

    lines: list[str] = []
    lines.append("1) Top 3")
    for i, t in enumerate(top_tasks):
        lines.append(f"- **[{t.id}]** {t.title} @ {start_times[i]} ({t.estimated_minutes}m)")
        lines.append(f"  Starter (2 min): {t.starter or 'MISSING — create starter in task.'}")
        lines.append(f"  DoD: {t.dod or 'MISSING — define measurable outcome.'}")
        nxt = t.notes or "MISSING — add notes."
        if t.link:
            nxt = f"{nxt} (Link: {t.link})"
        lines.append(f"  Next: {nxt}")
        lines.append("")

    # Doable Now: microtasks first, then normal tasks (up to 8 total)
    micro = [t for t in pool if (t.title or "").startswith("[MICRO]")]
    normal = [t for t in pool if not (t.title or "").startswith("[MICRO]")]
    doables = []
    for t in micro:
        if t.id not in {x.id for x in top_tasks}:
            doables.append(t)
        if len(doables) >= 8:
            break
    if len(doables) < 8:
        for t in normal:
            if t.id not in {x.id for x in top_tasks}:
                doables.append(t)
            if len(doables) >= 8:
                break

    lines.append("2) Doable Now")
    if doables:
        for t in doables:
            lines.append(f"- **[{t.id}]** {t.title} ({t.estimated_minutes}m)")
    else:
        lines.append("- No tasks available")

    lines.append("")
    lines.append("3) Quick Wins")
    lines.append("- None")
    lines.append("")
    lines.append("4) Motivation")
    lines.append("- Momentum beats motivation. Do the first 2 minutes.")

    return "\n".join(lines).strip(), top_tasks


async def _generate_lane_plan(db: Session, project: str, top_n: int) -> tuple[str, list[Task]]:
    goals, tasks_sorted = _fetch_lane(db, project)
    pool = tasks_sorted[:25]

    llm = LLMClient()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        today=date.today().isoformat(),
        focus_project=project,
        goals=_fmt_goals(goals),
        tasks=_fmt_tasks(pool),
    )

    if llm.enabled():
        try:
            content = await llm.generate(user_prompt)

            chosen_ids = _extract_top_ids(content, top_n=top_n)
            if chosen_ids:
                chosen_tasks = list(db.scalars(select(Task).where(Task.id.in_(chosen_ids))).all())
                chosen_map = {t.id: t for t in chosen_tasks}
                top_tasks = [chosen_map[i] for i in chosen_ids if i in chosen_map]
            else:
                top_tasks = pool[:top_n]

            for t in top_tasks:
                await generate_microtasks(db, t, llm)

            return content, top_tasks

        except Exception as e:
            fallback, top_tasks = _deterministic_plan(pool, top_n=top_n)
            content = (
                f"1) Top 3\n"
                f"- **[0]** LLM failed — fallback plan used @ 00:00 (0m)\n"
                f"  Starter (2 min): MISSING — fix LLM config or disable LLM\n"
                f"  DoD: Planner runs without errors using deterministic mode\n"
                f"  Next: Error was {type(e).__name__}: {e}\n\n"
                f"{fallback}"
            )
            for t in top_tasks:
                await generate_microtasks(db, t, llm)
            return content, top_tasks

    content, top_tasks = _deterministic_plan(pool, top_n=top_n)
    for t in top_tasks:
        await generate_microtasks(db, t, llm)
    return content, top_tasks


async def generate_daily_plan(
    db: Session,
    focus_project: Optional[str] = None,
    mode: str = "single",
) -> PlanOut:
    if mode == "split":
        haven_content, haven_top = await _generate_lane_plan(db, "haven", top_n=2)
        os_content, os_top = await _generate_lane_plan(db, "onestream", top_n=1)

        content = "\n".join(
            [
                "Daily Plan — Split Mode (Haven + OneStream)",
                "",
                haven_content,
                "",
                os_content,
            ]
        ).strip()

        top_ids = ",".join([*(str(t.id) for t in haven_top), *(str(t.id) for t in os_top)])

        db.add(PlanRun(project="split", top_task_ids=top_ids, content=content))
        db.commit()

        return PlanOut(generated_at=datetime.utcnow(), content=content)

    if not focus_project:
        focus_project = "haven"

    lane_content, lane_top = await _generate_lane_plan(db, focus_project, top_n=3)

    content = "\n".join(
        [
            f"Daily Plan — Goal Autopilot ({focus_project})",
            "",
            lane_content,
        ]
    ).strip()

    top_ids = ",".join(str(t.id) for t in lane_top)
    db.add(PlanRun(project=focus_project, top_task_ids=top_ids, content=content))
    db.commit()

    return PlanOut(generated_at=datetime.utcnow(), content=content)
