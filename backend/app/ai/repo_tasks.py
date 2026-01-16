# backend/app/ai/repo_tasks.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..config import settings
from .llm import LLMClient

SYSTEM = (
    "You are a senior software engineer generating actionable development tasks from a repo snapshot.\n"
    "Tasks must be specific, testable, and grounded in the provided file excerpts.\n"
    "Return JSON only in this shape:\n"
    "{\n"
    '  "tasks": [\n'
    "    {\n"
    '      "title": str,\n'
    '      "notes": str,\n'
    '      "tags": str,\n'
    '      "priority": int,                // 1-5\n'
    '      "estimated_minutes": int,        // 15-240\n'
    '      "blocks_me": bool,\n'
    '      "path": str,                    // one of the provided evidence_files paths\n'
    '      "line": int | null,\n'
    '      "starter": str,\n'
    '      "dod": str\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Do not invent files not provided."
)

# Signals we care about. (Case-insensitive.)
# NOTE: "..." is noisy, so we count it only on comment-ish lines elsewhere.
_MARKER_RE = re.compile(r"\b(TODO|FIXME|HACK|XXX|BUG|NOTE)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PromptBudgets:
    # “Top N files by signal strength”
    max_files: int = 18

    # “cap each excerpt to ~400–800 chars”
    max_chars_per_file: int = 800

    # “cap total prompt to ~12k chars”
    max_total_chars: int = 12_000


def _trim(s: str, n: int) -> str:
    s = s or ""
    s = s.replace("\r\n", "\n")
    return s if len(s) <= n else (s[: n - 20] + "\n...<truncated>...")


def _signal_score(s: dict[str, Any]) -> int:
    """
    Higher score => more likely to include the file in the prompt.
    We bias toward “real work surfaces”: routes, services, DB, infra, auth, tests,
    *and* explicit code markers (TODO/FIXME/XXX/etc).
    """
    path = (s.get("path") or "").lower()
    score = 0

    # explicit markers (presence-weighted)
    score += 5 * int(bool(s.get("todo_count")))
    score += 7 * int(bool(s.get("fixme_count")))
    score += 4 * int(bool(s.get("hack_count")))
    score += 4 * int(bool(s.get("xxx_count")))
    score += 4 * int(bool(s.get("bug_count")))
    score += 2 * int(bool(s.get("note_count")))
    score += 1 * int(bool(s.get("dotdotdot_count")))  # intentionally weak

    # “surface area” heuristics
    if "/routes/" in path or path.endswith("main.py"):
        score += 10
    if "/services/" in path or "/service_layer/" in path:
        score += 7
    if "docker" in path or "compose" in path or "k8" in path:
        score += 4
    if "auth" in path or "security" in path or "oauth" in path:
        score += 6
    if "/tests/" in path or path.startswith("tests/"):
        score += 4
    if "db" in path or "models" in path or "migrations" in path or "alembic" in path:
        score += 6
    if "ci" in path or ".github/workflows" in path:
        score += 5

    return score


def build_prompt(
    repo_name: str,
    branch: str,
    commit_sha: str | None,
    snapshot_id: int,
    signal_counts: dict[str, Any],
    file_summaries: list[dict[str, Any]],
) -> str:
    """
    Build an LLM prompt that is aggressively size-bounded for small-context local models.
    """
    budgets = PromptBudgets(
        max_files=int(getattr(settings, "REPO_TASKGEN_MAX_FILES", 18)),
        max_chars_per_file=int(getattr(settings, "REPO_TASKGEN_MAX_CHARS_PER_FILE", 800)),
        max_total_chars=int(getattr(settings, "REPO_TASKGEN_MAX_TOTAL_CHARS", 12_000)),
    )

    ranked = sorted(file_summaries, key=_signal_score, reverse=True)[: budgets.max_files]

    evidence: list[dict[str, Any]] = []
    running_chars = 0

    for f in ranked:
        path = f.get("path") or ""
        excerpt = _trim(f.get("excerpt") or "", budgets.max_chars_per_file)

        item = {
            "path": path,
            "signals": {
                "todo": int(f.get("todo_count") or 0),
                "fixme": int(f.get("fixme_count") or 0),
                "hack": int(f.get("hack_count") or 0),
                "xxx": int(f.get("xxx_count") or 0),
                "bug": int(f.get("bug_count") or 0),
                "note": int(f.get("note_count") or 0),
                "dotdotdot": int(f.get("dotdotdot_count") or 0),
            },
            "excerpt": excerpt,
        }

        # Compact JSON estimate (no spaces)
        item_json = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if running_chars + len(item_json) > budgets.max_total_chars:
            break

        evidence.append(item)
        running_chars += len(item_json)

    payload = {
        "repo": repo_name,
        "branch": branch,
        "commit_sha": commit_sha,
        "snapshot_id": snapshot_id,
        "signal_counts": signal_counts,
        "evidence_files": evidence,
        "instructions": [
            "Generate 5–12 tasks.",
            "Prefer tasks that improve reliability, security, DX, tests, and production-readiness.",
            "Each task must include: title, notes, tags, priority(1-5), estimated_minutes, blocks_me, path, line(optional), starter, dod.",
            "Tags must include: repo, autogen.",
            "Do not invent files not provided.",
        ],
    }

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


async def generate_repo_tasks_json(
    *,
    repo_name: str,
    branch: str,
    commit_sha: str | None,
    snapshot_id: int,
    signal_counts: dict[str, Any],
    file_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calls the local/OpenAI-compatible model and returns parsed JSON.
    """
    client = LLMClient()
    user_prompt = build_prompt(
        repo_name=repo_name,
        branch=branch,
        commit_sha=commit_sha,
        snapshot_id=snapshot_id,
        signal_counts=signal_counts,
        file_summaries=file_summaries,
    )

    raw = await client.chat(system=SYSTEM, user=user_prompt, temperature=0.2, max_tokens=1200)

    try:
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Repo task LLM returned non-JSON. First 400 chars: {raw[:400]}") from e


def count_markers_in_text(text: str) -> dict[str, int]:
    """
    Deterministic signal counter for a file content blob.
    Adds TODO/FIXME/HACK/XXX/BUG/NOTE and a conservative “...” signal.

    “...” is counted only on comment-ish lines to avoid massive noise.
    """
    t = text or ""
    todo = fixme = hack = xxx = bug = note = dotdotdot = 0

    # marker counts
    for m in _MARKER_RE.finditer(t):
        k = m.group(1).lower()
        if k == "todo":
            todo += 1
        elif k == "fixme":
            fixme += 1
        elif k == "hack":
            hack += 1
        elif k == "xxx":
            xxx += 1
        elif k == "bug":
            bug += 1
        elif k == "note":
            note += 1

    # conservative "...": only comment-ish lines
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        looks_like_comment = s.startswith(("#", "//", "/*", "*", "--")) or "  #" in line or "//" in line
        if looks_like_comment and "..." in s:
            dotdotdot += 1

    return {
        "todo_count": todo,
        "fixme_count": fixme,
        "hack_count": hack,
        "xxx_count": xxx,
        "bug_count": bug,
        "note_count": note,
        "dotdotdot_count": dotdotdot,
    }
