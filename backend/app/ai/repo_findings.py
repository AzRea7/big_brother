# backend/app/ai/repo_findings.py
from __future__ import annotations

import json
import re
from typing import Any

from ..config import settings
from .llm import LLMClient


def _strip_fences_and_extract_json(raw: str) -> dict[str, Any]:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError(f"No JSON object found. First 400 chars: {raw[:400]}")
    return json.loads(s[first : last + 1])


def build_findings_prompt(*, repo: str, branch: str, commit_sha: str | None, snapshot_id: int, signal_counts: dict[str, Any], file_summaries: list[dict[str, Any]]) -> str:
    rules = f"""
You are scanning a codebase snapshot and must output ONLY JSON.

Repo: {repo}
Branch: {branch}
Commit: {commit_sha or ""}
Snapshot: {snapshot_id}

Your job:
- Identify high-impact issues and missing production hardening.
- Generate "Findings" that behave like virtual TODO/FIXME markers.
- Each finding must point to a file path and (if possible) a line number.

Output JSON schema:
{{
  "findings": [
    {{
      "path": "onehaven/backend/app/main.py",
      "line": 123,
      "category": "security|reliability|db|api|tests|ops|perf|style",
      "severity": 1..5,
      "title": "short actionable title",
      "evidence": "why this is a problem + short excerpt (<=300 chars)",
      "recommendation": "what to change + acceptance criteria (clear DoD)"
    }}
  ]
}}

Hard rules:
- JSON only. No markdown, no commentary.
- Limit findings to 12 max.
- Prefer correctness over quantity.
- If you are unsure about a line number, omit it or set null.
"""

    payload = {
        "signal_counts": signal_counts,
        "files": [
            {"path": f.get("path"), "excerpt": f.get("excerpt")}
            for f in file_summaries
        ],
    }
    return rules.strip() + "\n\nCONTEXT:\n" + json.dumps(payload, indent=2)


async def generate_repo_findings_json(*, repo: str, branch: str, commit_sha: str | None, snapshot_id: int, signal_counts: dict[str, Any], file_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    client = LLMClient()
    if not client.enabled():
        raise RuntimeError("LLM not enabled (set OPENAI_BASE_URL + OPENAI_MODEL, and LLM_ENABLED=true)")

    prompt = build_findings_prompt(
        repo=repo,
        branch=branch,
        commit_sha=commit_sha,
        snapshot_id=snapshot_id,
        signal_counts=signal_counts,
        file_summaries=file_summaries,
    )

    raw = await client.chat(
        system="You are a senior backend engineer doing a production readiness review.",
        user=prompt,
        temperature=0.1,
        max_tokens=1600,
    )

    if isinstance(raw, str):
        return _strip_fences_and_extract_json(raw)
    return raw
