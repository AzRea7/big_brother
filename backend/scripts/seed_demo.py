from __future__ import annotations

import os
import sys
from datetime import date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.db import init_db, SessionLocal  # noqa: E402
from app.models import Goal, Task  # noqa: E402


def _add_task(
    db,
    *,
    title: str,
    goal_id: int,
    days_from_now: int | None = None,
    priority: int = 3,
    minutes: int = 30,
    blocks: bool = False,
    notes: str | None = None,
):
    due = (date.today() + timedelta(days=days_from_now)) if days_from_now is not None else None
    db.add(
        Task(
            title=title,
            goal_id=goal_id,
            due_date=due,
            priority=priority,
            estimated_minutes=minutes,
            blocks_me=blocks,
            notes=notes,
        )
    )


def main():
    init_db()
    db = SessionLocal()
    try:
        # Clear old demo data (optional but helps if you rerun)
        # Comment out if you don't want wipes.
        db.query(Task).delete()
        db.query(Goal).delete()
        db.commit()

        # GOALS (tracks)
        g_career = Goal(
            title="Transition into OneStream Cloud or AI Engineering",
            why="Increase leverage, grow faster, move into platform/AI work",
            target_date=date.today() + timedelta(days=90),
        )
        g_stack = Goal(
            title="Build OneStream platform map (support → cloud brain)",
            why="Become the support engineer who can trace systems end-to-end",
            target_date=date.today() + timedelta(days=60),
        )
        g_internal_tool = Goal(
            title="Ship 1 internal tool that reduces support pain",
            why="Demonstrate capability and create pull from engineering managers",
            target_date=date.today() + timedelta(days=45),
        )
        g_certs = Goal(
            title="Earn AZ-104 and start AI-102 path",
            why="Azure-first credibility + cross-cloud advantage",
            target_date=date.today() + timedelta(days=120),
        )
        g_haven = Goal(
            title="Ship Haven / Lead Consolidation MVP",
            why="Build a real revenue engine for realtor firms",
            target_date=date.today() + timedelta(days=45),
        )

        db.add_all([g_career, g_stack, g_internal_tool, g_certs, g_haven])
        db.commit()
        db.refresh(g_career)
        db.refresh(g_stack)
        db.refresh(g_internal_tool)
        db.refresh(g_certs)
        db.refresh(g_haven)

        # --- Step 1: Identify the 3 teams + 1 senior/manager/director each ---
        # Cloud
        _add_task(db, title="Identify 1 Cloud senior engineer (Teams/org chart)", goal_id=g_career.id, days_from_now=2, priority=5, minutes=15, blocks=True)
        _add_task(db, title="Identify 1 Cloud manager (Teams/org chart)", goal_id=g_career.id, days_from_now=2, priority=5, minutes=15, blocks=True)
        _add_task(db, title="Identify 1 Cloud director (Teams/org chart)", goal_id=g_career.id, days_from_now=3, priority=5, minutes=15, blocks=True)

        # Data
        _add_task(db, title="Identify 1 Data senior engineer (Teams/org chart)", goal_id=g_career.id, days_from_now=3, priority=5, minutes=15, blocks=True)
        _add_task(db, title="Identify 1 Data manager (Teams/org chart)", goal_id=g_career.id, days_from_now=4, priority=4, minutes=15)
        _add_task(db, title="Identify 1 Data director (Teams/org chart)", goal_id=g_career.id, days_from_now=5, priority=4, minutes=15)

        # AI
        _add_task(db, title="Identify 1 AI senior engineer (Teams/org chart)", goal_id=g_career.id, days_from_now=4, priority=5, minutes=15, blocks=True)
        _add_task(db, title="Identify 1 AI manager (Teams/org chart)", goal_id=g_career.id, days_from_now=6, priority=4, minutes=15)
        _add_task(db, title="Identify 1 AI director (Teams/org chart)", goal_id=g_career.id, days_from_now=7, priority=4, minutes=15)

        # Message script tasks
        script = (
            "Message template:\n"
            "“Hey — I’m on Support Engineering, but I’m trying to understand how OneStream runs its cloud and ML stack. "
            "I’d love to learn how your team actually ships. Would you mind if I asked you a few questions sometime?”"
        )
        _add_task(db, title="Send 2 curiosity messages (Cloud/Data/AI)", goal_id=g_career.id, days_from_now=7, priority=5, minutes=20, blocks=True, notes=script)
        _add_task(db, title="Schedule 2 short chats from replies", goal_id=g_career.id, days_from_now=14, priority=5, minutes=30, blocks=True)

        # --- Step 2: Become the support engineer who traces cloud problems ---
        _add_task(db, title="Next cloud-related ticket: ask 'What system does this touch?'", goal_id=g_stack.id, days_from_now=1, priority=5, minutes=10, blocks=True)
        _add_task(db, title="Trace a ticket across Storage/AKS/App Service/Data/ML endpoint", goal_id=g_stack.id, days_from_now=3, priority=5, minutes=30, blocks=True)
        _add_task(db, title="Write a 5-line production issue summary (what, where, why, fix, prevent)", goal_id=g_stack.id, days_from_now=3, priority=4, minutes=15)

        # --- Step 3: Build something inside OneStream ---
        _add_task(db, title="Pick one pain: repeated failures, slow diagnostics, data anomalies, misconfigs", goal_id=g_internal_tool.id, days_from_now=2, priority=5, minutes=20, blocks=True)
        _add_task(db, title="Prototype tool (Python/log analyzer/dashboard) for that pain", goal_id=g_internal_tool.id, days_from_now=10, priority=5, minutes=120, blocks=True)
        _add_task(db, title="Demo tool to team lead + 1 cloud engineer + 1 data/AI person", goal_id=g_internal_tool.id, days_from_now=14, priority=5, minutes=30, blocks=True)

        # --- Certifications ---
        _add_task(db, title="Register for AZ-104 exam date", goal_id=g_certs.id, days_from_now=7, priority=5, minutes=10, blocks=True)
        _add_task(db, title="AZ-104 study: Identity + RBAC + Networking (1 focused block)", goal_id=g_certs.id, days_from_now=2, priority=4, minutes=60)
        _add_task(db, title="Plan next cert: AZ-305 (cloud path) OR AI-102 (AI path)", goal_id=g_certs.id, days_from_now=14, priority=4, minutes=20)

        # --- Haven / business ---
        _add_task(db, title="Implement lead ingestion stub + normalization", goal_id=g_haven.id, days_from_now=2, priority=5, minutes=120, blocks=True)
        _add_task(db, title="Add simple scoring + top 10 endpoint", goal_id=g_haven.id, days_from_now=5, priority=4, minutes=120)
        _add_task(db, title="Reach out to 1 realtor to pilot the app", goal_id=g_haven.id, days_from_now=1, priority=5, minutes=20, blocks=True)
        _add_task(db, title="Get 1 realtor using it (feedback + iteration)", goal_id=g_haven.id, days_from_now=14, priority=5, minutes=30, blocks=True)

        # --- Weekly review habit ---
        _add_task(db, title="Weekly review: what moved, what blocked, set next Top 3", goal_id=g_stack.id, days_from_now=6, priority=4, minutes=25)

        db.commit()
        print("✅ Seeded FULL strategy goals + tasks.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
