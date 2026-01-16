from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..config import settings
from .llm import LLMClient


SYSTEM = """You are a senior engineer + technical program manager.
Convert repo state into a small, high-leverage task list.

Rules:
- Output must be STRICT JSON (no markdown).
- Output format: {"tasks":[{...}, ...]}
- Each task must include:
  - "title" (string, imperative verb start)
  - "notes" (string, includes why + what + acceptance criteria)
  - "priority" (int 1-5, 5 is highest)
  - "estimated_minutes" (int)
  - "blocks_me" (bool)
  - "tags" (string)  // must include "repo"
  - "link" (string)  // repo://...#path
  - "starter" (string)
  - "dod" (string)
- Keep it to 5–12 tasks max.
- Do NOT invent files that don't exist. Use provided paths only.
- Tasks must be grounded in the provided excerpts. If excerpt is missing, keep it generic and point to the file path.
"""


@dataclass(frozen=True)
class PromptBudgets:
    max_files: int = 18
    max_chars_per_file: int = 650
    max_total_chars: int = 14_000  # ~3.5k tokens-ish


def _signal_score(s: dict[str, Any]) -> int:
    """
    Higher score => more likely to include the file in the prompt.
    We bias toward "real work surfaces": API routes, services, DB, infra, auth, tests.
    """
    path = (s.get("path") or "").lower()
    score = 0

    # explicit markers
    score += 5 * int(bool(s.get("todo_count")))
    score += 6 * int(bool(s.get("fixme_count")))
    score += 3 * int(bool(s.get("hack_count")))

    # “surface area” heuristics
    if "/routes/" in path or path.endswith("main.py"):
        score += 10
    if "/services/" in path:
        score += 7
    if "docker" in path or "compose" in path:
        score += 4
    if "auth" in path or "security" in path:
        score += 6
    if "/tests/" in path or path.startswith("tests/"):
        score += 4
    if "db" in path or "models" in path or "migrations" in path:
        score += 6

    return score


def _trim(s: str, n: int) -> str:
    s = s or ""
    s = s.replace("\r\n", "\n")
    return s if len(s) <= n else (s[: n - 20] + "\n...<truncated>...")


def build_prompt(
    repo_name: str,
    branch: str,
    commit_sha: str | None,
    snapshot_id: int,
    signal_counts: dict[str, Any],
    file_summaries: list[dict[str, Any]],
) -> str:
    """
    Build an LLM prompt that CANNOT exceed a safe size for small-context local models.
    The #1 job is to avoid context overflows while preserving the highest-value evidence.
    """
    budgets = PromptBudgets(
        max_files=getattr(settings, "REPO_TASKGEN_MAX_FILES", 18),
        max_chars_per_file=getattr(settings, "REPO_TASKGEN_MAX_CHARS_PER_FILE", 650),
        max_total_chars=getattr(settings, "REPO_TASKGEN_MAX_TOTAL_CHARS", 14_000),
    )

    # Sort files by signal score (descending), then keep top N
    ranked = sorted(file_summaries, key=_signal_score, reverse=True)
    ranked = ranked[: budgets.max_files]

    # Build compact evidence list
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
            },
            "excerpt": excerpt,
        }

        item_json = json.dumps(item, ensure_ascii=False)
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
            "For each task: include a repo:// link that points to one of the evidence file paths.",
        ],
    }

    # Compact JSON to reduce tokens
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
