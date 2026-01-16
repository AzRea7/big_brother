from __future__ import annotations

from typing import Any

from .llm import LLMClient

SYSTEM = """You are a senior engineer + technical program manager.
Convert repo state into a small, high-leverage task list.

Rules:
- Output must be STRICT JSON (no markdown).
- Output format must be: {"tasks":[{...}, ...]}
- Each task object MUST include:
  - "title" (string, imperative verb start)
  - "notes" (string, include why + what + acceptance criteria)
  - "priority" (int 1-5, 5 highest)
  - "estimated_minutes" (int)
  - "link" (string)  // repo://<path>#Lx-Ly  OR repo://<path>
  - "tags" (string)  // must include "repo" and 1-3 topical tags
- Keep it 5–20 tasks. Prefer fewer, higher leverage.
- DO NOT invent files. Use only the provided paths.
"""


def build_user_prompt(
    *,
    repo: str,
    branch: str,
    snapshot_id: int,
    signals: dict[str, Any],
    file_summaries: list[dict[str, Any]],
) -> str:
    """
    file_summaries items should look like:
      {"path": "...", "why_relevant": "...", "snippets":[...], "signals":[...]}
    Keep snippets short; you already store all content in DB if you need deeper.
    """
    return (
        f"Repo: {repo}\n"
        f"Branch: {branch}\n"
        f"Snapshot ID: {snapshot_id}\n\n"
        f"Signals summary: {signals}\n\n"
        "Here are relevant files (paths and short context). "
        "Generate repo tasks that directly reference these paths in link.\n\n"
        f"FILES:\n{file_summaries}\n"
    )


async def suggest_repo_tasks_llm(
    *,
    repo: str,
    branch: str,
    snapshot_id: int,
    signals: dict[str, Any],
    file_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    client = LLMClient()
    payload = await client.chat_json(
        system=SYSTEM,
        user=build_user_prompt(
            repo=repo,
            branch=branch,
            snapshot_id=snapshot_id,
            signals=signals,
            file_summaries=file_summaries,
        ),
        temperature=0.2,
        max_tokens=1400,
    )

    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError(f"LLM JSON missing 'tasks' list. Got keys: {list(payload.keys())}")

    cleaned: list[dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        title = str(t.get("title") or "").strip()
        notes = str(t.get("notes") or "").strip()
        link = str(t.get("link") or "").strip()
        tags = str(t.get("tags") or "").strip()
        try:
            priority = int(t.get("priority"))
        except Exception:
            priority = 3
        try:
            est = int(t.get("estimated_minutes"))
        except Exception:
            est = 45

        if not title or not notes or not link:
            continue
        if "repo" not in tags:
            tags = (tags + ",repo").strip(",")
        cleaned.append(
            {
                "title": title,
                "notes": notes,
                "priority": max(1, min(5, priority)),
                "estimated_minutes": max(5, est),
                "link": link,
                "tags": tags,
            }
        )

    # Hard guard: if LLM returns empty tasks, treat as error (so you notice)
    if not cleaned:
        raise RuntimeError("LLM returned 0 valid tasks after validation.")

    return cleaned
