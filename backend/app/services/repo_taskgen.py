# app/services/repo_taskgen.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..ai.llm import LLMClient
from ..config import settings
from ..models import RepoFile, RepoSnapshot, Task

# Basic signal regexes
_TODO_RE = re.compile(r"\bTODO\b[:]?", re.IGNORECASE)
_FIXME_RE = re.compile(r"\bFIXME\b[:]?", re.IGNORECASE)
_STUB_RE = re.compile(r"\b(pass|TODO\(\)|raise NotImplementedError|NotImplementedError)\b")


def _parse_csv_list(s: str | None) -> list[str]:
    if not s:
        return []
    return [p.strip() for p in (s or "").split(",") if p.strip()]


@dataclass
class RepoTaskDraft:
    title: str
    notes: str | None
    priority: int
    estimated_minutes: int
    blocks_me: bool
    tags: str
    link: str
    starter: str | None = None
    dod: str | None = None


def compute_signal_counts(db: Session, snapshot_id: int) -> dict[str, int]:
    """
    Used by /repo UI to show how noisy the repo is (TODO/FIXME/stubs).
    Your routes/repo.py imports this directly, so it must exist.
    """
    files = (
        db.execute(
            select(RepoFile).where(
                RepoFile.snapshot_id == snapshot_id,
                RepoFile.skipped == False,  # noqa: E712
            )
        )
        .scalars()
        .all()
    )
    total = len(files)
    todo = 0
    fixme = 0
    impl = 0

    for f in files:
        if not f.content:
            continue
        if _TODO_RE.search(f.content):
            todo += 1
        if _FIXME_RE.search(f.content):
            fixme += 1
        if _STUB_RE.search(f.content):
            impl += 1

    return {
        "total_files": total,
        "files_with_todo": todo,
        "files_with_fixme": fixme,
        "files_with_impl_signals": impl,
    }


def _repo_summary(db: Session, snapshot_id: int) -> dict[str, Any]:
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        return {"snapshot_id": snapshot_id, "repo": None, "branch": None, "file_count": 0, "top_folders": []}

    files = db.execute(select(RepoFile.path).where(RepoFile.snapshot_id == snapshot_id)).scalars().all()
    folder_counts: dict[str, int] = {}
    for p in files:
        parts = p.split("/")
        top = parts[0] if parts else "<root>"
        folder_counts[top] = folder_counts.get(top, 0) + 1

    top_folders = sorted(
        [{"folder": k, "count": v} for k, v in folder_counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:10]

    warnings: list[str] = []
    try:
        warnings = list(snap.warnings or [])
    except Exception:
        warnings = []

    return {
        "snapshot_id": snapshot_id,
        "repo": snap.repo,
        "branch": snap.branch,
        "commit_sha": snap.commit_sha,
        "file_count": snap.file_count,
        "stored_content_files": snap.stored_content_files,
        "top_folders": top_folders,
        "warnings": warnings,
    }


def _extract_signal_hits(db: Session, snapshot_id: int, limit_files: int = 60) -> list[dict[str, Any]]:
    """
    Returns a list of 'hits' with (path, line, kind, snippet).
    We cap file count to keep LLM prompts reasonable.
    """
    files = (
        db.execute(
            select(RepoFile)
            .where(RepoFile.snapshot_id == snapshot_id, RepoFile.skipped == False)  # noqa: E712
            .order_by(RepoFile.path.asc())
        )
        .scalars()
        .all()
    )

    hits: list[dict[str, Any]] = []
    scanned = 0
    for f in files:
        if scanned >= limit_files:
            break
        scanned += 1

        if not f.content:
            continue
        lines = f.content.splitlines()
        for i, line in enumerate(lines, start=1):
            kind = None
            if _TODO_RE.search(line):
                kind = "TODO"
            elif _FIXME_RE.search(line):
                kind = "FIXME"
            elif _STUB_RE.search(line):
                kind = "IMPL"

            if kind:
                hits.append(
                    {
                        "path": f.path,
                        "line": i,
                        "kind": kind,
                        "snippet": line.strip()[:220],
                    }
                )
            if len(hits) >= 200:
                break
        if len(hits) >= 200:
            break

    return hits


def _deterministic_seed_tasks(snapshot_id: int, seeds: list[str], summary: dict[str, Any]) -> list[RepoTaskDraft]:
    """
    Used when LLM is off or fails. Creates useful tasks that still work end-to-end.
    """
    repo = summary.get("repo") or "repo"
    branch = summary.get("branch") or "main"

    mapping: dict[str, RepoTaskDraft] = {
        "security": RepoTaskDraft(
            title="Security pass: tighten auth + secrets hygiene",
            notes="Review debug endpoints, API-key handling, headers, CORS, and ensure secrets are never logged. Add rate limiting where appropriate.",
            priority=5,
            estimated_minutes=90,
            blocks_me=True,
            tags="repo,seed,security",
            link=f"repo://seed/security?snapshot={snapshot_id}&repo={repo}&branch={branch}",
            starter="Search for API key checks, debug routes exposure, logging of headers/env. Identify missing auth/rate-limit.",
            dod="Document risks + implement fixes (auth guardrails, masked logs, rate limit). Add 1 regression test.",
        ),
        "e2e": RepoTaskDraft(
            title="Add E2E test: repo sync → task gen → complete persists",
            notes="Add an end-to-end test (or smoke script) that calls sync_and_generate, asserts tasks exist, completes one, and confirms persistence.",
            priority=5,
            estimated_minutes=120,
            blocks_me=True,
            tags="repo,seed,e2e",
            link=f"repo://seed/e2e?snapshot={snapshot_id}&repo={repo}&branch={branch}",
            starter="Write a pytest that triggers sync_and_generate, then uses /tasks and /complete endpoints.",
            dod="One E2E test passes in CI. Includes assertions on repo:// links and repo tags.",
        ),
        "observability": RepoTaskDraft(
            title="Observability: request IDs + structured logs + safe errors",
            notes="Add request_id correlation, structured logging, and ensure error responses don't leak secrets. Add 1 regression test.",
            priority=4,
            estimated_minutes=90,
            blocks_me=False,
            tags="repo,seed,observability",
            link=f"repo://seed/observability?snapshot={snapshot_id}&repo={repo}&branch={branch}",
            starter="Add request_id middleware + log fields; ensure exception handler sanitizes.",
            dod="Logs show request_id consistently; errors are sanitized; add 1 test for error response shape.",
        ),
    }

    drafts: list[RepoTaskDraft] = []
    for seed in seeds:
        key = seed.strip().lower()
        if key in mapping:
            drafts.append(mapping[key])

    # fallback: unknown seeds become generic tasks
    for seed in seeds:
        key = seed.strip().lower()
        if key and key not in mapping:
            drafts.append(
                RepoTaskDraft(
                    title=f"Repo improvement: {key}",
                    notes=f"Seed-driven improvement task: {key}. Convert into concrete work items tied to code areas.",
                    priority=3,
                    estimated_minutes=60,
                    blocks_me=False,
                    tags="repo,seed",
                    link=f"repo://seed/{key}?snapshot={snapshot_id}&repo={repo}&branch={branch}",
                    starter="Scan repo for relevant modules; draft implementation plan.",
                    dod="Produce 2-3 concrete sub-tasks with file anchors.",
                )
            )

    # if no seeds, still create something
    if not drafts:
        drafts.append(
            RepoTaskDraft(
                title="Repo health pass: resolve top TODO/FIXME signals",
                notes="Scan for TODO/FIXME/NotImplementedError hotspots and convert the highest-impact items into tracked tasks with links.",
                priority=3,
                estimated_minutes=60,
                blocks_me=False,
                tags="repo,autogen",
                link=f"repo://snapshot/{snapshot_id}",
                starter="Use repo signal counts and top hits to pick 3 items.",
                dod="3 tasks exist with repo:// links and clear DoD.",
            )
        )

    return drafts


def _task_exists(db: Session, project: str, link: str, title: str) -> bool:
    # Dedupe by link OR by (project,title) for safety
    existing = (
        db.execute(
            select(Task).where(
                Task.project == project,
                (Task.link == link) | (Task.title == title),
            )
        )
        .scalars()
        .first()
    )
    return existing is not None


async def _llm_generate_seed_tasks(
    *,
    snapshot_id: int,
    seeds: list[str],
    summary: dict[str, Any],
    hits: list[dict[str, Any]],
) -> list[RepoTaskDraft]:
    llm = LLMClient()

    system = (
        "You are an expert software engineering tech lead. "
        "Generate actionable engineering tasks from repo context. "
        "Return strict JSON: {tasks:[{title,notes,priority,estimated_minutes,blocks_me,tags,link,starter,dod}]}. "
        "No markdown."
    )

    user_payload = {
        "repo_summary": summary,
        "seeds": seeds,
        "signals": hits[:80],
        "rules": {
            "project": "haven",
            "must_include_repo_link": True,
            "repo_link_format": "repo://seed/<seed>?snapshot=<id> OR repo://file/<path>#L<line>",
            "tags_must_include": ["repo", "autogen"],
            "task_count": 5,
        },
    }

    raw = await llm.chat(
        system=system,
        user=json.dumps(user_payload, ensure_ascii=False),
        temperature=0.2,
        max_tokens=900,
    )

    # Parse strict JSON (with a salvage attempt)
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"LLM did not return JSON. First 400 chars:\n{raw[:400]}")
        data = json.loads(raw[start : end + 1])

    tasks = (data or {}).get("tasks") or []
    drafts: list[RepoTaskDraft] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        try:
            drafts.append(
                RepoTaskDraft(
                    title=str(t["title"])[:300],
                    notes=(t.get("notes") or None),
                    priority=int(t.get("priority", 3)),
                    estimated_minutes=int(t.get("estimated_minutes", 60)),
                    blocks_me=bool(t.get("blocks_me", False)),
                    tags=str(t.get("tags") or "repo,autogen"),
                    link=str(t.get("link") or f"repo://snapshot/{snapshot_id}"),
                    starter=t.get("starter"),
                    dod=t.get("dod"),
                )
            )
        except Exception:
            continue

    return drafts


async def generate_tasks_from_snapshot(db: Session, snapshot_id: int, project: str = "haven") -> tuple[int, int]:
    """
    1) Build summary + extract signal hits
    2) Expand with seeds:
       - if LLM enabled -> ask LLM for tasks based on summary+seeds+signals
       - else -> deterministic seed tasks
    3) Insert into Task table, tagged 'repo'
    """
    summary = _repo_summary(db, snapshot_id)
    hits = _extract_signal_hits(db, snapshot_id)
    seeds = _parse_csv_list(getattr(settings, "REPO_TASK_SEEDS", ""))

    drafts: list[RepoTaskDraft]
    if settings.LLM_ENABLED:
        try:
            drafts = await _llm_generate_seed_tasks(snapshot_id=snapshot_id, seeds=seeds, summary=summary, hits=hits)
        except Exception:
            drafts = _deterministic_seed_tasks(snapshot_id, seeds, summary)
    else:
        drafts = _deterministic_seed_tasks(snapshot_id, seeds, summary)

    created = 0
    skipped = 0

    for d in drafts:
        # enforce tags include repo/autogen
        tagset = {x.strip() for x in (d.tags or "").split(",") if x.strip()}
        tagset.add("repo")
        tagset.add("autogen")
        d.tags = ",".join(sorted(tagset))

        if getattr(settings, "HAVEN_REPO_ONLY", False) and project == "haven":
            if not (d.link or "").startswith("repo://"):
                skipped += 1
                continue

        if _task_exists(db, project, d.link, d.title):
            skipped += 1
            continue

        db.add(
            Task(
                goal_id=None,
                title=d.title,
                notes=d.notes,
                due_date=None,
                priority=max(1, min(5, int(d.priority))),
                estimated_minutes=max(5, int(d.estimated_minutes)),
                blocks_me=bool(d.blocks_me),
                completed=False,
                project=project,
                tags=d.tags,
                link=d.link,
                starter=d.starter,
                dod=d.dod,
            )
        )
        created += 1

    db.commit()
    return created, skipped
