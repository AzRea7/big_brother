from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import select

from ..models import Goal, Task, TaskDependency, PlanRun
from ..schemas import PlanOut
from ..ai.llm import LLMClient
from ..ai.prompts import USER_PROMPT_TEMPLATE
from .microtasks import generate_microtasks


def _fmt_goals(goals: list[Goal]) -> str:
    if not goals:
        return "- (none)"
    lines = []
    for g in goals:
        td = f" target={g.target_date}" if g.target_date else ""
        why = f" | why={g.why}" if g.why else ""
        lines.append(f"- [{g.id}] {g.title}{td}{why}")
    return "\n".join(lines)


def _fmt_tasks(tasks: list[Task]) -> str:
    if not tasks:
        return "- (none)"
    lines = []
    for t in tasks:
        dd = f" due={t.due_date}" if t.due_date else ""
        blk = " blocks_me=true" if t.blocks_me else ""
        proj = f" project={t.project}" if t.project else ""
        meta = []
        if t.impact_score is not None:
            meta.append(f"impact={t.impact_score}")
        if t.confidence is not None:
            meta.append(f"confidence={t.confidence}")
        if t.energy:
            meta.append(f"energy={t.energy}")
        meta_s = f" ({', '.join(meta)})" if meta else ""
        link = f" link={t.link}" if t.link else ""
        starter = f" starter={t.starter}" if t.starter else ""
        dod = f" dod={t.dod}" if t.dod else ""
        tags = f" tags={t.tags}" if t.tags else ""
        lines.append(
            f"- [{t.id}] {t.title} (p={t.priority}, est={t.estimated_minutes}m{dd}{blk}{proj}){meta_s}{tags}{link}{starter}{dod}"
        )
    return "\n".join(lines)


def _dependency_blocked_ids(db: Session, task_ids: list[int]) -> set[int]:
    """
    Returns set of task_ids that are blocked because they depend on an incomplete task.
    """
    if not task_ids:
        return set()

    deps = list(
        db.scalars(
            select(TaskDependency).where(TaskDependency.task_id.in_(task_ids))
        ).all()
    )
    if not deps:
        return set()

    depends_ids = {d.depends_on_task_id for d in deps}
    incomplete = {
        t.id
        for t in db.scalars(
            select(Task).where(Task.id.in_(list(depends_ids)), Task.completed == False)  # noqa: E712
        ).all()
    }

    blocked: set[int] = set()
    for d in deps:
        if d.depends_on_task_id in incomplete:
            blocked.add(d.task_id)
    return blocked


def _deterministic_sort_key(t: Task):
    due_ord = t.due_date.toordinal() if t.due_date else 10**9

    # higher is better for impact/confidence; default to 0
    impact = t.impact_score or 0
    conf = t.confidence or 0

    # Prefer tasks with starter/dod/link (precision fuel)
    has_starter = 0 if (t.starter and t.starter.strip()) else 1
    has_dod = 0 if (t.dod and t.dod.strip()) else 1
    has_link = 0 if (t.link and t.link.strip()) else 1
    has_notes = 0 if (t.notes and t.notes.strip()) else 1

    # Sort order: earlier due, blocks_me first, higher impact/confidence, higher priority, shorter, more “precise”
    return (
        due_ord,
        0 if t.blocks_me else 1,
        -(impact * 10 + conf),
        -t.priority,
        t.estimated_minutes,
        has_starter,
        has_dod,
        has_link,
        has_notes,
    )


async def generate_daily_plan(db: Session, focus_project: Optional[str] = None) -> PlanOut:
    today = date.today()

    goals = list(db.scalars(select(Goal).where(Goal.is_archived == False)).all())  # noqa: E712

    task_stmt = select(Task).where(Task.completed == False)  # noqa: E712

    # Optional focus by project: "haven" | "onestream" | "personal"
    if focus_project:
        task_stmt = task_stmt.where(Task.project == focus_project)

    tasks = list(db.scalars(task_stmt).all())

    # Remove blocked tasks (dependency-aware)
    blocked = _dependency_blocked_ids(db, [t.id for t in tasks])
    tasks_unblocked = [t for t in tasks if t.id not in blocked]

    # Deterministic pre-rank
    tasks_sorted = sorted(tasks_unblocked, key=_deterministic_sort_key)

    # Limit context
    top_candidate_pool = tasks_sorted[:25]

    llm = LLMClient()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        today=today.isoformat(),
        focus_project=focus_project or "(none)",
        goals=_fmt_goals(goals),
        tasks=_fmt_tasks(top_candidate_pool),
    )

    if llm.enabled():
        content = await llm.generate(user_prompt)
    else:
        # fallback: simple deterministic plan
        top = top_candidate_pool[:3]
        lines = [f"1) Top 3"]
        start_times = ["09:00", "10:00", "11:00"]
        for i, t in enumerate(top):
            lines.append(f"- **[{t.id}]** {t.title} @ {start_times[i]} ({t.estimated_minutes}m)")
            lines.append(f"  Starter (2 min): {(t.starter or 'Open the relevant file and write the first 5 lines.')}")
            lines.append(f"  DoD: {(t.dod or 'Make a visible commit/test proving it works.')}")
            lines.append(f"  Next: {(t.notes or 'Move it forward one concrete step.')}")
            lines.append("")
        lines.append("2) Doable Now")
        for t in top_candidate_pool[3:11]:
            lines.append(f"- **[{t.id}]** {t.title}")
        lines.append("")
        lines.append("3) Quick Wins")
        lines.append("- Reduce one task note into a precise starter + DoD.")
        lines.append("- Mark one task complete.")
        lines.append("- Add one dependency to prevent wrong ordering.")
        lines.append("")
        lines.append("4) Motivation")
        lines.append("Tiny real progress beats perfect plans — ship something measurable today.")
        content = "\n".join(lines).strip()

    # Save plan run history (Improvement 5)
    # Try to infer top_task_ids from the deterministic ranking (safe even if LLM output differs)
    top_ids = ",".join(str(t.id) for t in top_candidate_pool[:3]) if top_candidate_pool else ""
    pr = PlanRun(project=focus_project, top_task_ids=top_ids, content=content)
    db.add(pr)
    db.commit()

    # Microtask generation (Improvement 4) for top 3 candidates (not parsing LLM)
    for t in top_candidate_pool[:3]:
        await generate_microtasks(db, t, llm)

    return PlanOut(generated_at=datetime.utcnow(), content=content)
