from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import select, or_

from ..models import Goal, Task, TaskDependency, PlanRun
from ..schemas import PlanOut
from ..ai.llm import LLMClient
from ..ai.prompts import USER_PROMPT_TEMPLATE
from .microtasks import generate_microtasks


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
    if not task_ids:
        return set()

    deps = list(
        db.scalars(select(TaskDependency).where(TaskDependency.task_id.in_(task_ids))).all()
    )
    if not deps:
        return set()

    depends_ids = {d.depends_on_task_id for d in deps}

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
    for d in deps:
        if d.depends_on_task_id in incomplete_dep_ids:
            blocked.add(d.task_id)
    return blocked


def _deterministic_sort_key(t: Task):
    due_ord = t.due_date.toordinal() if t.due_date else 10**9

    impact = t.impact_score or 0
    conf = t.confidence or 0

    missing_starter = 0 if (t.starter and t.starter.strip()) else 1
    missing_dod = 0 if (t.dod and t.dod.strip()) else 1
    missing_link = 0 if (t.link and t.link.strip()) else 1

    return (
        due_ord,
        0 if t.blocks_me else 1,
        -(impact * 10 + conf),
        -t.priority,
        t.estimated_minutes,
        missing_starter,
        missing_dod,
        missing_link,
    )


def _extract_top_task_ids(plan_text: str, top_n: int) -> list[int]:
    """
    Parses IDs from the required output format:
      - **[id]** Title @ ...
    Returns up to top_n unique IDs, in order.
    """
    ids: list[int] = []
    for m in re.finditer(r"\*\*\[(\d+)\]\*\*", plan_text):
        tid = int(m.group(1))
        if tid not in ids:
            ids.append(tid)
        if len(ids) >= top_n:
            break
    return ids


def _fetch_lane(db: Session, project: str) -> tuple[list[Goal], list[Task]]:
    goals = list(
        db.scalars(
            select(Goal).where(
                Goal.is_archived == False,  # noqa: E712
                Goal.project == project,
            )
        ).all()
    )
    goal_ids = [g.id for g in goals]

    # ✅ Backward compatible:
    # - tasks explicitly tagged with lane
    # - OR tasks with project NULL but whose goal is in lane
    stmt = select(Task).where(
        Task.completed == False,  # noqa: E712
        or_(
            Task.project == project,
            (Task.project == None) & (Task.goal_id.in_(goal_ids)) if goal_ids else False,  # noqa: E711
        ),
    )

    tasks = list(db.scalars(stmt).all())

    blocked_ids = _dependency_blocked_ids(db, [t.id for t in tasks])
    tasks = [t for t in tasks if t.id not in blocked_ids]

    tasks_sorted = sorted(tasks, key=_deterministic_sort_key)
    return goals, tasks_sorted


async def _generate_lane_plan(db: Session, project: str, top_n: int) -> tuple[str, list[Task]]:
    today = date.today()
    goals, tasks_sorted = _fetch_lane(db, project)
    pool = tasks_sorted[:25]
    llm = LLMClient()

    user_prompt = USER_PROMPT_TEMPLATE.format(
        today=today.isoformat(),
        focus_project=project,
        goals=_fmt_goals(goals),
        tasks=_fmt_tasks(pool),
    )

    if llm.enabled():
        content = await llm.generate(user_prompt)
        chosen_ids = _extract_top_task_ids(content, top_n=top_n)

        chosen_tasks: list[Task] = []
        if chosen_ids:
            for tid in chosen_ids:
                t = db.get(Task, tid)
                if t and not t.completed:
                    chosen_tasks.append(t)

        # Fallback: if parsing fails, use deterministic top_n from pool
        if not chosen_tasks:
            chosen_tasks = pool[:top_n]

    else:
        # deterministic fallback
        chosen_tasks = pool[:top_n]
        start_times = ["09:00", "10:00", "11:00"]
        lines: list[str] = []
        lines.append(f"Lane: {project}")
        lines.append("1) Top 3")
        for i, t in enumerate(chosen_tasks):
            lines.append(f"- **[{t.id}]** {t.title} @ {start_times[i]} ({t.estimated_minutes}m)")
            lines.append(f"  Starter (2 min): {t.starter or 'MISSING — create starter in task.'}")
            lines.append(f"  DoD: {t.dod or 'MISSING — define measurable outcome.'}")
            nxt = t.notes or "MISSING — add notes."
            if t.link:
                nxt = f"{nxt} (Link: {t.link})"
            lines.append(f"  Next: {nxt}")
            lines.append("")
        content = "\n".join(lines).strip()

    # ✅ Microtasks for the ACTUAL chosen tasks
    for t in chosen_tasks:
        await generate_microtasks(db, t, llm)

    return content, chosen_tasks


async def generate_daily_plan(
    db: Session,
    focus_project: Optional[str] = None,
    mode: str = "single",
) -> PlanOut:
    """
    mode:
      - single: one lane only (requires focus_project)
      - split: 2 haven + 1 onestream
    """
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

    # single lane
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
