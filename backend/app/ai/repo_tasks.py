# backend/app/ai/repo_tasks.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..config import settings
from .llm import LLMClient


SYSTEM = """You are a senior engineer + technical program manager.
Convert repo state into a small, high-leverage task list.

Rules:
- Output must be STRICT JSON (no markdown, no code fences, no commentary).
- Output format: {"tasks":[{...}, ...]}
- Each task must include:
  - "title" (string, imperative verb start)
  - "notes" (string, includes why + what + acceptance criteria)
  - "priority" (int 1-5, 5 is highest)
  - "estimated_minutes" (int)
  - "blocks_me" (bool)
  - "tags" (string)  // must include "repo,autogen"
  - "path" (string)  // one of the provided evidence paths
  - "line" (int|null) // optional
  - "starter" (string)
  - "dod" (string)
- Keep it to 5–12 tasks max.
- Do NOT invent files that don't exist. Use provided paths only.
- Tasks must be grounded in the provided excerpts.
"""


@dataclass(frozen=True)
class PromptBudgets:
    max_files: int = 18
    max_chars_per_file: int = 800
    max_total_chars: int = 12_000  # roughly ~3k tokens-ish (rule of thumb)


def _trim(s: str, n: int) -> str:
    s = (s or "").replace("\r\n", "\n")
    if len(s) <= n:
        return s
    # leave a clear marker so the model “feels” truncation
    return s[: max(0, n - 20)] + "\n...<truncated>..."


def _signal_score(s: dict[str, Any]) -> int:
    """
    Higher score => more likely to include the file in the prompt.
    We prefer real “work surfaces”: routes, services, DB, auth, infra, tests.
    """
    path = (s.get("path") or "").lower()
    score = 0

    # Explicit markers (presence-weighted)
    score += 6 * int(bool(s.get("fixme_count")))
    score += 5 * int(bool(s.get("todo_count")))
    score += 3 * int(bool(s.get("hack_count")))
    score += 2 * int(bool(s.get("xxx_count")))
    score += 2 * int(bool(s.get("bug_count")))
    score += 1 * int(bool(s.get("note_count")))
    score += 1 * int(bool(s.get("dotdotdot_count")))

    # Surface-area heuristics
    if "/routes/" in path or path.endswith("main.py"):
        score += 10
    if "/services/" in path:
        score += 7
    if "docker" in path or "compose" in path or "github/workflows" in path or path.endswith(".yml"):
        score += 4
    if "auth" in path or "security" in path or "api_key" in path:
        score += 6
    if "/tests/" in path or path.startswith("tests/"):
        score += 4
    if "db" in path or "models" in path or "migrations" in path or "alembic" in path:
        score += 6

    return score


_MARKERS = [
    ("todo_count", re.compile(r"\bTODO\b", re.IGNORECASE)),
    ("fixme_count", re.compile(r"\bFIXME\b", re.IGNORECASE)),
    ("hack_count", re.compile(r"\bHACK\b", re.IGNORECASE)),
    ("xxx_count", re.compile(r"\bXXX\b", re.IGNORECASE)),
    ("bug_count", re.compile(r"\bBUG\b", re.IGNORECASE)),
    ("note_count", re.compile(r"\bNOTE\b", re.IGNORECASE)),
]


def count_markers_in_text(text: str) -> dict[str, int]:
    t = text or ""
    out: dict[str, int] = {k: 0 for (k, _) in _MARKERS}

    for k, rx in _MARKERS:
        out[k] = len(rx.findall(t))

    # “...” signal (useful for unfinished logic / placeholders)
    # We count literal "..." sequences, but avoid exploding on long ellipses by counting runs.
    out["dotdotdot_count"] = len(re.findall(r"\.\.\.", t))

    return out


def build_prompt(
    repo_name: str,
    branch: str,
    commit_sha: str | None,
    snapshot_id: int,
    signal_counts: dict[str, Any],
    file_summaries: list[dict[str, Any]],
) -> str:
    """
    Build a prompt that is aggressively size-bounded for small-context local models.
    """
    budgets = PromptBudgets(
        max_files=int(getattr(settings, "REPO_TASK_MAX_FILES", 18)),
        max_chars_per_file=int(getattr(settings, "REPO_TASK_EXCERPT_CHARS", 800)),
        max_total_chars=int(getattr(settings, "REPO_TASK_MAX_TOTAL_CHARS", 12_000)),
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

        # Compact estimate (no spaces)
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
            "Tags must include: repo,autogen.",
            "Do not invent files not provided.",
            "Return JSON only. No markdown. No code fences.",
        ],
    }

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _extract_json_object_lenient(raw: str) -> dict[str, Any]:
    """
    Local models often violate “JSON only”:
    - They wrap in ```json fences
    - They add a sentence before JSON
    - They may add trailing text

    We defensively:
    - strip code fences
    - take the substring from first '{' to last '}'
    - parse that
    """
    s = (raw or "").strip()

    # Remove common code fences
    s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s.strip())

    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError(f"No JSON object found. First 400 chars: {raw[:400]}")

    candidate = s[first : last + 1]
    return json.loads(candidate)


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

    raw = await client.chat(system=SYSTEM, user=user_prompt, temperature=0.2, max_tokens=900)

    try:
        return _extract_json_object_lenient(raw)
    except Exception as e:
        raise RuntimeError(
            "Repo task LLM returned non-JSON or truncated JSON. "
            f"First 500 chars:\n{raw[:500]}"
        ) from e
