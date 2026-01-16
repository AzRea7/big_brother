from __future__ import annotations

import json
from typing import Any

from .llm import LLMClient


SYSTEM = """You are a senior engineer + technical program manager.
Your job: convert repo state into a small, high-leverage task list.

Rules:
- Output must be STRICT JSON (no markdown).
- Output format: {"tasks":[{...}, ...]}
- Each task must include:
  - "title" (string, imperative verb start)
  - "notes" (string, includes why + what + acceptance criteria)
  - "priority" (int 1-5, 5 is highest)
  - "estimated_minutes" (int)
  - "link" (string)  // repo://path#Lx or repo://path
  - "tags" (string)  // must include "repo"
- Keep it to 5–20 tasks max. Prefer fewer, higher leverage.
- Do NOT invent files that don't exist. Use provided paths only.
"""


def build_prompt(repo_name: str, snapshot_id: int, signals: dict[str, Any], file_summaries: list[dict[str, Any]]) -> str:
    payload = {
        "repo": repo_name,
        "snapshot_id": snapshot_id,
        "signals": signals,
        "files": file_summaries,
        "instruction": (
            "Generate tasks that advance the codebase meaningfully. "
            "Prefer tasks that unblock runtime, tests, safety, and core flows."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


async def suggest_repo_tasks(
    *,
    llm: LLMClient,
    repo_name: str,
    snapshot_id: int,
    signals: dict[str, Any],
    file_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Returns a list of task dicts:
    {title, notes, priority, estimated_minutes, link, tags}
    """
    prompt = build_prompt(repo_name, snapshot_id, signals, file_summaries)
    raw = await llm.chat(system=SYSTEM, user=prompt)

    # Parse strict JSON
    try:
        obj = json.loads(raw)
    except Exception:
        # If model wrapped in extra text, try salvage the largest JSON object
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"LLM did not return JSON. First 400 chars:\n{raw[:400]}")
        obj = json.loads(raw[start : end + 1])

    tasks = obj.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("Invalid LLM response: tasks is not a list")

    out: list[dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        title = (t.get("title") or "").strip()
        if not title:
            continue

        tags = (t.get("tags") or "").strip()
        if "repo" not in tags.split(","):
            tags = ("repo," + tags).strip(",") if tags else "repo"

        out.append(
            {
                "title": title,
                "notes": (t.get("notes") or "").strip() or None,
                "priority": int(t.get("priority") or 3),
                "estimated_minutes": int(t.get("estimated_minutes") or 30),
                "link": (t.get("link") or "").strip() or None,
                "tags": tags,
            }
        )

    # Hard cap safety
    return out[:20]
