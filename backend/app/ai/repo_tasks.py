# backend/app/ai/repo_tasks.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from ..config import settings
from .llm import LLMClient

SYSTEM_PROMPT = """You are a senior engineer and technical program manager.
You generate actionable engineering tasks from a repository snapshot.

Hard rules:
- EVERY task MUST be grounded in provided evidence excerpts.
- If you cannot cite evidence for a task, DO NOT include that task.
- Return JSON only.

Task schema (strict):
{
  "tasks": [
    {
      "title": "string (specific, actionable, NOT generic refactor)",
      "notes": "string (what + why + concrete plan)",
      "priority": 1..5,
      "estimated_minutes": 5..480,
      "blocks_me": true|false,
      "tags": "comma-separated tags like repo,autogen,security|perf|reliability,signal:todo",
      "path": "repo-relative file path",
      "line": number|null,
      "link": "repo://...#path:Lline",
      "starter": "2-5 minute first step",
      "dod": "definition of done (measurable)",
      "evidence": {
        "path": "same as path",
        "line": number|null,
        "quote": "short quote from excerpt (<=200 chars)"
      }
    }
  ]
}

Quality rules:
- Prefer tasks like: add auth to a route, add rate limiting, tighten input validation,
  fix N+1 query, add caching, improve error handling, add tests around a risky area.
- Avoid: "refactor X" unless you specify EXACTLY what and why with evidence.
- If evidence is only NOTE comments, treat it as low value (skip unless it points to real risk).
"""


@dataclass(frozen=True)
class PromptBudgets:
    max_files: int = 18
    max_chars_per_file: int = 800
    max_total_chars: int = 12_000  # rough token bound


def _trim(s: str, n: int) -> str:
    s = (s or "").replace("\r\n", "\n")
    if len(s) <= n:
        return s
    return s[: max(0, n - 20)] + "\n...<truncated>..."


def build_prompt(
    *,
    repo_name: str,
    branch: str,
    commit_sha: str | None,
    snapshot_id: int,
    signal_counts: dict[str, Any],
    file_summaries: list[dict[str, Any]],
    # Optional extra evidence (e.g., retrieved chunks)
    extra_evidence: Optional[list[dict[str, Any]]] = None,
) -> str:
    budgets = PromptBudgets(
        max_files=int(getattr(settings, "REPO_TASK_MAX_FILES", 18)),
        max_chars_per_file=int(getattr(settings, "REPO_TASK_EXCERPT_CHARS", 800)),
        max_total_chars=int(getattr(settings, "REPO_TASK_MAX_TOTAL_CHARS", 12_000)),
    )

    ranked = file_summaries[: budgets.max_files]

    evidence_files: list[dict[str, Any]] = []
    running_chars = 0

    for f in ranked:
        path = f.get("path") or ""
        excerpt = _trim(f.get("excerpt") or "", budgets.max_chars_per_file)

        item = {
            "path": path,
            "signals": {k: int(v or 0) for k, v in (f.get("signals") or {}).items()},
            "excerpt": excerpt,
        }

        item_json = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if running_chars + len(item_json) > budgets.max_total_chars:
            break

        evidence_files.append(item)
        running_chars += len(item_json)

    payload: dict[str, Any] = {
        "repo": repo_name,
        "branch": branch,
        "commit_sha": commit_sha,
        "snapshot_id": snapshot_id,
        "signal_counts": signal_counts,
        "evidence_files": evidence_files,
        "instructions": [
            "Generate 5–12 tasks.",
            "Prefer tasks that improve reliability, security, DX, tests, and production-readiness.",
            "Each task must include: title, notes, tags, priority(1-5), estimated_minutes, blocks_me, path, line(optional), starter, dod.",
            "Tags must include: repo,autogen.",
            "ALWAYS include at least one signal tag when applicable (signal:timeout, signal:auth, signal:validation, etc.).",
            "Do not invent files not provided.",
            "Return JSON only. No markdown. No code fences.",
        ],
    }

    if extra_evidence:
        payload["extra_evidence"] = extra_evidence

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _extract_json_object_lenient(raw: str) -> dict[str, Any]:
    s = (raw or "").strip()
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
    extra_evidence: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    client = LLMClient()
    user_prompt = build_prompt(
        repo_name=repo_name,
        branch=branch,
        commit_sha=commit_sha,
        snapshot_id=snapshot_id,
        signal_counts=signal_counts,
        file_summaries=file_summaries,
        extra_evidence=extra_evidence,
    )

    raw = await client.chat(system=SYSTEM_PROMPT, user=user_prompt, temperature=0.2, max_tokens=900)

    try:
        return _extract_json_object_lenient(raw)
    except Exception as e:
        raise RuntimeError(
            "Repo task LLM returned non-JSON or truncated JSON. "
            f"First 500 chars:\n{raw[:500]}"
        ) from e
