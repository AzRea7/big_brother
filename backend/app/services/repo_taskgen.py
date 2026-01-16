from __future__ import annotations

import re
from typing import Any

from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoFile, RepoSnapshot, Task
from ..ai.repo_tasks import suggest_repo_tasks_llm


def compute_signal_counts(db: Session, snapshot_id: int) -> dict[str, int]:
    total = db.query(RepoFile).filter(RepoFile.snapshot_id == snapshot_id).count()

    # look in text content only
    rows = db.query(RepoFile.path, RepoFile.content).filter(
        RepoFile.snapshot_id == snapshot_id,
        RepoFile.content_kind == "text",
        RepoFile.content.isnot(None),
    ).all()

    todo = 0
    fixme = 0
    impl = 0

    impl_patterns = [
        r"\bTODO\b",
        r"\bFIXME\b",
        r"pass\s*(#\s*TODO|#\s*FIXME)?",
        r"raise\s+NotImplementedError",
        r"NotImplementedError\(",
        r"IMPLEMENT\s+ME",
    ]

    for _, content in rows:
        c = content or ""
        if "TODO" in c:
            todo += 1
        if "FIXME" in c:
            fixme += 1
        if any(re.search(p, c) for p in impl_patterns):
            impl += 1

    return {
        "total_files": total,
        "files_with_todo": todo,
        "files_with_fixme": fixme,
        "files_with_impl_signals": impl,
    }


def _select_relevant_files(db: Session, snapshot_id: int, limit: int = 20) -> list[RepoFile]:
    """
    Heuristic: grab a small set of files that contain strong signals.
    Your current signal_counts shows 9 impl-signal files; we pull around that.
    """
    candidates = db.query(RepoFile).filter(
        RepoFile.snapshot_id == snapshot_id,
        RepoFile.content_kind == "text",
        RepoFile.skipped == False,  # noqa: E712
        RepoFile.content.isnot(None),
    ).all()

    scored: list[tuple[int, RepoFile]] = []
    for rf in candidates:
        c = rf.content or ""
        score = 0
        score += 3 if "TODO" in c else 0
        score += 3 if "FIXME" in c else 0
        score += 4 if "NotImplementedError" in c else 0
        score += 1 if "test" in rf.path.lower() else 0
        score += 1 if rf.path.endswith(".py") else 0
        if score > 0:
            scored.append((score, rf))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [rf for _, rf in scored[:limit]]


def _file_summary(rf: RepoFile, max_chars: int = 1200) -> dict[str, Any]:
    c = (rf.content or "")
    c = c[:max_chars]
    signals: list[str] = []
    if "TODO" in c:
        signals.append("TODO")
    if "FIXME" in c:
        signals.append("FIXME")
    if "NotImplementedError" in c:
        signals.append("NotImplementedError")
    return {
        "path": rf.path,
        "signals": signals,
        "snippet": c,
    }


def _seed_tasks(repo: str, branch: str, snapshot_id: int, seeds: list[str]) -> list[dict[str, Any]]:
    # deterministic “starter pack” if LLM disabled
    base = [
        {
            "title": "Security pass: tighten auth + secrets hygiene",
            "notes": "Review debug endpoints, API-key handling, headers, CORS, and ensure secrets are never logged. Add rate limiting where appropriate. Acceptance: no secret logs, debug endpoints guarded.",
            "priority": 5,
            "estimated_minutes": 90,
            "link": f"repo://seed/security?snapshot={snapshot_id}&repo={repo}&branch={branch}",
            "tags": "repo,security,autogen,seed",
        },
        {
            "title": "Add E2E test: repo sync → task gen → complete persists",
            "notes": "Add an end-to-end test that triggers sync_and_generate, asserts tasks exist, completes one, and confirms persistence. Acceptance: test passes in CI.",
            "priority": 5,
            "estimated_minutes": 120,
            "link": f"repo://seed/e2e?snapshot={snapshot_id}&repo={repo}&branch={branch}",
            "tags": "repo,e2e,autogen,seed",
        },
        {
            "title": "Observability: request IDs + structured logs + safe errors",
            "notes": "Add request_id correlation, structured logging, and ensure error responses don't leak secrets. Acceptance: request_id present, errors sanitized, add regression test.",
            "priority": 4,
            "estimated_minutes": 90,
            "link": f"repo://seed/observability?snapshot={snapshot_id}&repo={repo}&branch={branch}",
            "tags": "repo,observability,autogen,seed",
        },
    ]
    wanted = set([s.strip().lower() for s in seeds if s.strip()])
    if not wanted:
        return base
    out = []
    for t in base:
        if any(w in t["tags"].lower() or w in t["title"].lower() for w in wanted):
            out.append(t)
    return out or base


def generate_tasks_from_snapshot(db: Session, *, snapshot_id: int, project: str) -> tuple[int, int]:
    """
    Returns (created_tasks, skipped_duplicates)
    Dedupe key = (project, title, link)
    """
    snap = db.query(RepoSnapshot).filter(RepoSnapshot.id == snapshot_id).first()
    if not snap:
        raise ValueError(f"snapshot_id not found: {snapshot_id}")

    # guardrail: only allow haven repo tasks unless you disable
    if settings.HAVEN_REPO_ONLY and project != "haven":
        raise RuntimeError("Repo task generation restricted to project=haven (set HAVEN_REPO_ONLY=false to change).")

    signals = compute_signal_counts(db, snapshot_id)
    relevant = _select_relevant_files(db, snapshot_id, limit=20)
    summaries = [_file_summary(rf) for rf in relevant]

    tasks_payload: list[dict[str, Any]]
    if settings.LLM_ENABLED:
        tasks_payload = (
            # LLM-first: if it fails, you WANT it to fail loudly during dev
            # so you fix prompt/response instead of silently seeding forever.
            __import__("asyncio").run(suggest_repo_tasks_llm(  # type: ignore
                repo=snap.repo,
                branch=snap.branch,
                snapshot_id=snap.id,
                signals=signals,
                file_summaries=summaries,
            ))
        )
    else:
        seeds = [s.strip() for s in (settings.REPO_TASK_SEEDS or "").split(",")]
        tasks_payload = _seed_tasks(snap.repo, snap.branch, snap.id, seeds)

    created = 0
    skipped = 0

    for t in tasks_payload:
        title = (t.get("title") or "").strip()
        link = (t.get("link") or "").strip()
        if not title:
            continue

        # dedupe
        exists = db.query(Task).filter(
            Task.project == project,
            Task.title == title,
            Task.link == link,
        ).first()
        if exists:
            skipped += 1
            continue

        db.add(
            Task(
                title=title,
                notes=t.get("notes"),
                priority=int(t.get("priority") or 3),
                estimated_minutes=int(t.get("estimated_minutes") or 45),
                project=project,
                tags=t.get("tags") or "repo,autogen",
                link=link,
                blocks_me=bool(t.get("blocks_me") or False),
                starter=t.get("starter"),
                dod=t.get("dod"),
            )
        )
        created += 1

    db.commit()
    return created, skipped
