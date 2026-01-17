# backend/app/services/repo_taskgen.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoFile, RepoSnapshot, Task

# --- Import compatibility (repo evolves; keep this file resilient) ---
# We want ONE canonical signal counter used everywhere.
try:
    # Newer location (recommended)
    from .code_signals import count_markers_in_text  # type: ignore
except Exception:  # pragma: no cover
    # Older location
    from ..ai.repo_tasks import count_markers_in_text  # type: ignore

try:
    # LLM task generator (JSON schema: {"tasks":[...]}), contains bounded prompt builder.
    from ..ai.repo_tasks import generate_repo_tasks_json  # type: ignore
except Exception:  # pragma: no cover
    # If you ever moved it elsewhere, patch the import here.
    from ..ai.repo_tasks import generate_repo_tasks_json  # type: ignore


# -------------------------
# Deterministic fallback (only used if LLM fails)
# -------------------------

# Classic markers + broadened set.
_FALLBACK_RE = re.compile(r"\b(TODO|FIXME|HACK|XXX|BUG|NOTE)\b[:\-\s]*(.*)", re.IGNORECASE)

# NOTE is usually documentation. Only convert to tasks if it looks actionable.
_ACTION_WORDS_RE = re.compile(
    r"\b(should|must|need|needs|fix|implement|add|remove|refactor|rename|handle|support|validate|sanitize|secure|test)\b",
    re.IGNORECASE,
)

# Skip messages that are basically punctuation/quotes/etc. (prevents NOTE: "," garbage).
_GARBAGE_RE = re.compile(r"^[\s\"',.:;()\[\]{}<>/\\-]*$")

# Order fallback results by severity/importance.
_KIND_SCORE = {"BUG": 100, "FIXME": 90, "HACK": 80, "XXX": 70, "TODO": 60, "NOTE": 10}


@dataclass
class SuggestedTask:
    title: str
    notes: str
    link: str | None
    tags: str
    priority: int = 3
    estimated_minutes: int = 60
    blocks_me: bool = False
    starter: str | None = None
    dod: str | None = None


def _repo_link(snapshot: RepoSnapshot, path: str, line: int | None = None) -> str:
    base = f"repo://{snapshot.repo}?branch={snapshot.branch}&commit_sha={snapshot.commit_sha or ''}"
    return f"{base}#{path}:L{line}" if line is not None else f"{base}#{path}"


# -------------------------
# Public: signal counts (used by routes + prompt builder)
# -------------------------

def compute_signal_counts(db: Session, snapshot_id: int) -> dict[str, int]:
    """
    Backwards-compatible DB-based signal counter.

    Routes import compute_signal_counts(db, snapshot_id).
    Internally we delegate to compute_signal_counts_for_files to avoid duplicated logic.
    """
    files = db.scalars(select(RepoFile).where(RepoFile.snapshot_id == snapshot_id)).all()
    return compute_signal_counts_for_files(files)


def compute_signal_counts_for_files(files: list[RepoFile]) -> dict[str, int]:
    """
    Produces file-level counts for both:
      - classic TODO/FIXME/HACK/etc.
      - production signals (auth/timeout/retry/rate_limit/validation/logging/metrics/db/tests/etc.)

    Design choice:
      - These are "files_with_X" counts (not total matches) because it's stable and
        helps ranking/snapshot selection without being skewed by long files.
    """
    total = len(files)

    files_with: dict[str, int] = {
        # classic marker files-with
        "todo": 0,
        "fixme": 0,
        "hack": 0,
        "xxx": 0,
        "bug": 0,
        "note": 0,
        "dotdotdot": 0,
        # production signal files-with
        "auth": 0,
        "timeout": 0,
        "retry": 0,
        "rate_limit": 0,
        "validation": 0,
        "logging": 0,
        "metrics": 0,
        "db": 0,
        "tests": 0,
        "ci": 0,
        "docker": 0,
        "config": 0,
        "secrets": 0,
        "nplus1": 0,
        "cors": 0,
        "csrf": 0,
    }

    impl = 0

    for f in files:
        if f.content_kind != "text" or not (getattr(f, "content_text", None) or f.content):
            continue

        text = (getattr(f, "content_text", None) or f.content or "")
        if not text:
            continue

        sig = count_markers_in_text(text)

        # classic
        if sig.get("todo_count", 0) > 0:
            files_with["todo"] += 1
        if sig.get("fixme_count", 0) > 0:
            files_with["fixme"] += 1
        if sig.get("hack_count", 0) > 0:
            files_with["hack"] += 1
        if sig.get("xxx_count", 0) > 0:
            files_with["xxx"] += 1
        if sig.get("bug_count", 0) > 0:
            files_with["bug"] += 1
        if sig.get("note_count", 0) > 0:
            files_with["note"] += 1
        if sig.get("dotdotdot_count", 0) > 0:
            files_with["dotdotdot"] += 1

        # production signals
        if sig.get("auth_signal", 0) > 0:
            files_with["auth"] += 1
        if sig.get("timeout_signal", 0) > 0:
            files_with["timeout"] += 1
        if sig.get("retry_signal", 0) > 0:
            files_with["retry"] += 1
        if sig.get("rate_limit_signal", 0) > 0:
            files_with["rate_limit"] += 1
        if sig.get("input_validation_signal", 0) > 0:
            files_with["validation"] += 1
        if sig.get("logging_signal", 0) > 0:
            files_with["logging"] += 1
        if sig.get("metrics_signal", 0) > 0:
            files_with["metrics"] += 1
        if sig.get("db_signal", 0) > 0:
            files_with["db"] += 1
        if sig.get("tests_signal", 0) > 0:
            files_with["tests"] += 1
        if sig.get("ci_signal", 0) > 0:
            files_with["ci"] += 1
        if sig.get("docker_signal", 0) > 0:
            files_with["docker"] += 1
        if sig.get("config_signal", 0) > 0:
            files_with["config"] += 1
        if sig.get("secrets_signal", 0) > 0:
            files_with["secrets"] += 1
        if sig.get("nplus1_signal", 0) > 0:
            files_with["nplus1"] += 1
        if sig.get("cors_signal", 0) > 0:
            files_with["cors"] += 1
        if sig.get("csrf_signal", 0) > 0:
            files_with["csrf"] += 1

        # Implementation stubs / "unfinished" signals (useful in prod hardening)
        if any(x in text for x in ("raise NotImplementedError", "IMPLEMENT", "stub", "pass  #")):
            impl += 1

    return {
        "total_files": total,
        "files_with_todo": files_with["todo"],
        "files_with_fixme": files_with["fixme"],
        "files_with_hack": files_with["hack"],
        "files_with_xxx": files_with["xxx"],
        "files_with_bug": files_with["bug"],
        "files_with_note": files_with["note"],
        "files_with_dotdotdot": files_with["dotdotdot"],
        "files_with_impl_signals": impl,
        # production-grade rollups
        "files_with_auth": files_with["auth"],
        "files_with_timeout": files_with["timeout"],
        "files_with_retry": files_with["retry"],
        "files_with_rate_limit": files_with["rate_limit"],
        "files_with_validation": files_with["validation"],
        "files_with_logging": files_with["logging"],
        "files_with_metrics": files_with["metrics"],
        "files_with_db": files_with["db"],
        "files_with_tests": files_with["tests"],
        "files_with_ci": files_with["ci"],
        "files_with_docker": files_with["docker"],
        "files_with_config": files_with["config"],
        "files_with_secrets": files_with["secrets"],
        "files_with_nplus1": files_with["nplus1"],
        "files_with_cors": files_with["cors"],
        "files_with_csrf": files_with["csrf"],
    }


# -------------------------
# Fallback helpers
# -------------------------

def _is_actionable_note(msg: str) -> bool:
    m = (msg or "").strip()
    if not m:
        return False
    if _GARBAGE_RE.match(m):
        return False
    return bool(_ACTION_WORDS_RE.search(m))


def _clean_msg(msg: str) -> str:
    m = (msg or "").strip()
    if not m:
        return "Unspecified"
    if _GARBAGE_RE.match(m):
        return "Unspecified"
    return m[:160]


def _priority_for_kind(kind: str) -> int:
    k = kind.upper()
    if k in ("BUG", "FIXME"):
        return 5
    if k in ("HACK", "XXX"):
        return 4
    if k == "TODO":
        return 3
    return 2


def _minutes_for_kind(kind: str) -> int:
    k = kind.upper()
    if k in ("BUG", "FIXME"):
        return 90
    if k in ("HACK", "XXX"):
        return 75
    if k == "TODO":
        return 60
    return 45


def _blocks_for_kind(kind: str) -> bool:
    return kind.upper() in ("BUG", "FIXME", "HACK")


# -------------------------
# Robust JSON extraction (guards against fenced/non-JSON responses)
# -------------------------

def _extract_json_object_lenient(raw: str) -> dict[str, Any]:
    """
    Local/hosted models sometimes violate “JSON only”:
      - leading text
      - wrap in ```json fences
      - trailing commentary

    Strategy:
      - strip fences
      - take substring from first '{' to last '}'
      - parse it
    """
    s = (raw or "").strip()

    # strip code fences
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError(f"No JSON object found in response. First 400 chars: {raw[:400]}")

    candidate = s[first : last + 1]
    return json.loads(candidate)


# -------------------------
# Deterministic fallback task suggestion (filtered + ranked)
# -------------------------

def _suggest_tasks_from_signals(db: Session, snapshot_id: int, project: str, limit: int = 25) -> list[SuggestedTask]:
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        return []

    files = db.scalars(
        select(RepoFile)
        .where(RepoFile.snapshot_id == snapshot_id)
        .where(RepoFile.content_kind == "text")
    ).all()

    found: list[dict[str, Any]] = []

    for f in files:
        text = (getattr(f, "content_text", None) or f.content or "")
        if not text:
            continue

        lines = text.splitlines()
        # Hard cap: scanning giant files is slow and usually not worth it for fallback.
        for i, line in enumerate(lines[:2500], start=1):
            m = _FALLBACK_RE.search(line)
            if not m:
                continue

            kind = m.group(1).upper()
            raw_msg = (m.group(2) or "").strip()

            # Prevent NOTE spam: only convert NOTE to task if actionable.
            if kind == "NOTE" and not _is_actionable_note(raw_msg):
                continue

            msg = _clean_msg(raw_msg)
            if msg == "Unspecified" and kind == "NOTE":
                continue

            found.append(
                {
                    "kind": kind,
                    "msg": msg,
                    "path": f.path,
                    "line": i,
                    "line_text": line.strip(),
                    "score": _KIND_SCORE.get(kind, 0),
                }
            )

    if not found:
        return [
            SuggestedTask(
                title="Repo scan: identify top 3 high-impact improvements",
                notes=(
                    "No actionable TODO/FIXME/HACK/XXX/BUG markers were found in the stored snapshot content.\n\n"
                    "Starter (5–10 min): Skim API entrypoints (/main.py, /routes), auth, timeouts, retries, "
                    "error handling, and tests.\n"
                    "DoD: Create 3 concrete tasks with repo links and measurable acceptance criteria."
                ),
                link=_repo_link(snap, "onehaven/"),
                tags="repo,autogen,scan",
                priority=3,
                estimated_minutes=60,
                blocks_me=False,
                starter="Skim /main.py, /routes, services, and tests; write down 3 concrete improvements.",
                dod="3 concrete tasks exist with repo links and clear DoD.",
            )
        ]

    # Rank by severity first, then stable order by path/line
    found.sort(key=lambda x: (-x["score"], x["path"], x["line"]))

    out: list[SuggestedTask] = []
    for item in found[:limit]:
        kind = item["kind"]
        msg = item["msg"]
        path = item["path"]
        line_no = item["line"]
        line_text = item["line_text"]

        out.append(
            SuggestedTask(
                title=f"{kind}: {msg}",
                notes=(
                    f"Found {kind} in `{path}` line ~{line_no}:\n\n"
                    f"{line_text}\n\n"
                    "Starter (2–5 min): Open the file at the linked line, understand the intent, and locate related callsites.\n"
                    "DoD: The marker is resolved (or converted into a real ticket/reference), behavior is correct, "
                    "and tests are updated if needed."
                ),
                link=_repo_link(snap, path, line_no),
                tags=f"repo,autogen,code-signal,{kind.lower()}",
                priority=_priority_for_kind(kind),
                estimated_minutes=_minutes_for_kind(kind),
                blocks_me=_blocks_for_kind(kind),
                starter="Open the file at the linked line; reproduce/understand in 2–5 minutes.",
                dod="Marker resolved; correct behavior; test added/updated when applicable.",
            )
        )

    return out


# -------------------------
# LLM task generation (bounded + tolerant parsing)
# -------------------------

def _signal_strength(sig: dict[str, Any]) -> int:
    """
    Heuristic ranking to avoid sending the entire repo to the LLM.
    Higher = more likely to be important / actionable.
    """
    # Classic markers (high weight)
    strength = 0
    strength += 12 * int(sig.get("fixme_count", 0))
    strength += 12 * int(sig.get("bug_count", 0))
    strength += 10 * int(sig.get("hack_count", 0))
    strength += 8 * int(sig.get("xxx_count", 0))
    strength += 6 * int(sig.get("todo_count", 0))
    strength += 2 * int(sig.get("note_count", 0))
    strength += 4 * int(sig.get("dotdotdot_count", 0))

    # Production signals (medium weight)
    strength += 8 * int(sig.get("auth_signal", 0))
    strength += 6 * int(sig.get("timeout_signal", 0))
    strength += 6 * int(sig.get("retry_signal", 0))
    strength += 6 * int(sig.get("rate_limit_signal", 0))
    strength += 6 * int(sig.get("input_validation_signal", 0))
    strength += 4 * int(sig.get("logging_signal", 0))
    strength += 4 * int(sig.get("metrics_signal", 0))
    strength += 5 * int(sig.get("db_signal", 0))
    strength += 4 * int(sig.get("tests_signal", 0))
    strength += 4 * int(sig.get("ci_signal", 0))
    strength += 3 * int(sig.get("docker_signal", 0))
    strength += 3 * int(sig.get("config_signal", 0))
    strength += 6 * int(sig.get("secrets_signal", 0))
    strength += 4 * int(sig.get("nplus1_signal", 0))
    strength += 4 * int(sig.get("cors_signal", 0))
    strength += 4 * int(sig.get("csrf_signal", 0))
    return strength


async def _llm_generate_tasks(snapshot: RepoSnapshot, files: list[RepoFile], project: str) -> list[SuggestedTask]:
    """
    Uses bounded prompt builder (top N by signal strength, capped excerpts, capped total chars).
    Adds tolerant JSON handling in case the model emits fences or leading text.
    """
    if not getattr(settings, "LLM_ENABLED", False):
        return []
    if not getattr(settings, "OPENAI_MODEL", None):
        return []

    max_files = int(getattr(settings, "REPO_PROMPT_MAX_FILES", 60))
    excerpt_chars = int(getattr(settings, "REPO_PROMPT_EXCERPT_CHARS", 1200))

    file_summaries: list[dict[str, Any]] = []
    for f in files:
        if f.content_kind != "text":
            continue
        text = (getattr(f, "content_text", None) or f.content or "")
        if not text:
            continue

        sig = count_markers_in_text(text)
        file_summaries.append(
            {
                "path": f.path,
                # keep excerpt short here to reduce upstream memory + token pressure
                "excerpt": text[:excerpt_chars],
                **sig,
                "_strength": _signal_strength(sig),
            }
        )

    # Keep only the most "signal-heavy" files; this is the biggest practical win for prompt size.
    file_summaries.sort(key=lambda d: int(d.get("_strength", 0)), reverse=True)
    file_summaries = file_summaries[:max_files]
    for d in file_summaries:
        d.pop("_strength", None)

    signal_counts = compute_signal_counts_for_files(files)

    raw = await generate_repo_tasks_json(
        repo_name=snapshot.repo,
        branch=snapshot.branch,
        commit_sha=snapshot.commit_sha,
        snapshot_id=snapshot.id,
        signal_counts=signal_counts,
        file_summaries=file_summaries,
    )

    if isinstance(raw, str):
        raw = _extract_json_object_lenient(raw)

    tasks: list[SuggestedTask] = []
    for t in (raw or {}).get("tasks", []):
        path = str(t.get("path") or "").strip() or "unknown"
        line = t.get("line")
        link = _repo_link(snapshot, path, int(line) if isinstance(line, int) else None)

        tasks.append(
            SuggestedTask(
                title=str(t.get("title", "")).strip()[:240] or "Untitled repo task",
                notes=str(t.get("notes", "")).strip(),
                link=link,
                tags=str(t.get("tags", "repo,autogen")).strip(),
                priority=int(t.get("priority", 3)) if str(t.get("priority", "3")).isdigit() else 3,
                estimated_minutes=int(t.get("estimated_minutes", 60))
                if str(t.get("estimated_minutes", "60")).isdigit()
                else 60,
                blocks_me=bool(t.get("blocks_me", False)),
                starter=str(t.get("starter", "")).strip() or None,
                dod=str(t.get("dod", "")).strip() or None,
            )
        )

    return tasks


# -------------------------
# Materialization (DB writes + dedupe)
# -------------------------

def _task_key(title: str, link: str | None) -> str:
    return (title or "").strip().lower() + "|" + (link or "")


async def generate_tasks_from_snapshot(db: Session, snapshot_id: int, project: str = "haven") -> tuple[int, int]:
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        return 0, 0

    files = db.scalars(
        select(RepoFile)
        .where(RepoFile.snapshot_id == snapshot_id)
        .where(RepoFile.content_kind == "text")
    ).all()

    # 1) LLM path (bounded + tolerant parsing)
    suggestions: list[SuggestedTask] = []
    if getattr(settings, "LLM_ENABLED", False):
        try:
            suggestions = await _llm_generate_tasks(snap, files, project=project)
        except Exception:
            suggestions = []

    # 2) deterministic fallback (filtered NOTE)
    if not suggestions:
        suggestions = _suggest_tasks_from_signals(db, snapshot_id=snapshot_id, project=project, limit=25)

    # 3) dedupe vs existing tasks in this project (title+link)
    existing = db.scalars(select(Task).where(Task.project == project)).all()
    existing_keys = {_task_key(t.title or "", t.link) for t in existing}

    created = 0
    skipped = 0

    limit = int(getattr(settings, "REPO_TASK_COUNT", 8))
    for s in suggestions[:limit]:
        key = _task_key(s.title, s.link)
        if key in existing_keys:
            skipped += 1
            continue

        db.add(
            Task(
                title=s.title,
                notes=s.notes,
                project=project,
                tags=s.tags,
                link=s.link,
                priority=s.priority,
                estimated_minutes=s.estimated_minutes,
                blocks_me=s.blocks_me,
                starter=s.starter,
                dod=s.dod,
                completed=False,
            )
        )
        existing_keys.add(key)
        created += 1

    db.commit()
    return created, skipped
