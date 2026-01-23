# backend/app/ai/repo_findings.py
from __future__ import annotations

import json
import re
from typing import Any

from .llm import LLMClient


def _strip_fences_and_extract_json(raw: str) -> dict[str, Any]:
    """
    Legacy helper kept for compatibility, but we now prefer LLMClient.chat_json()
    which already does robust extraction/repair.
    """
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*```$", "", s).strip()

    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError(f"No JSON object found. First 400 chars: {s[:400]}")
    return json.loads(s[first : last + 1])


def _cap(s: str, n: int) -> str:
    s2 = (s or "").strip()
    return s2 if len(s2) <= n else s2[:n]


def _compact_payload(
    *,
    signal_counts: dict[str, Any],
    file_summaries: list[dict[str, Any]],
    max_files: int,
    max_excerpt_chars: int,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for f in (file_summaries or [])[:max_files]:
        path = _cap(str(f.get("path") or ""), 220)
        excerpt = _cap(str(f.get("excerpt") or ""), max_excerpt_chars)
        if not path or not excerpt:
            continue
        files.append({"path": path, "excerpt": excerpt})
    return {"signal_counts": signal_counts or {}, "files": files}


def build_findings_prompt(
    *,
    repo: str,
    branch: str,
    commit_sha: str | None,
    snapshot_id: int,
    signal_counts: dict[str, Any],
    file_summaries: list[dict[str, Any]],
    max_files: int = 14,
    max_excerpt_chars: int = 700,
) -> str:
    rules = f"""
You are scanning a codebase snapshot and must output ONLY JSON (a single object).

Repo: {repo}
Branch: {branch}
Commit: {commit_sha or ""}
Snapshot: {snapshot_id}

PRIORITY ORDER (do not waste tokens on lint):
1) secrets exposure, auth bypass, unsafe debug endpoints
2) missing validation, unsafe defaults, broken retries/timeouts, bad error handling
3) DB integrity / migrations / data loss risk
4) observability gaps that block debugging incidents
5) tests for critical flows
Style-only findings are allowed ONLY if you found nothing else.

Output JSON schema:
{{
  "findings": [
    {{
      "path": "repo/relative/path.ext",
      "line": 123 | null,
      "category": "security|auth|secrets|reliability|timeouts|retries|validation|db|api|tests|ops|observability|perf|style",
      "severity": 1,
      "title": "short actionable title",
      "evidence": "why + short excerpt (<=300 chars)",
      "recommendation": "what to change",
      "acceptance": "how to verify (pytest/curl/etc)"
    }}
  ]
}}

Hard rules:
- JSON only. No markdown. No prose.
- Return <= 12 findings.
- Every finding MUST include a real file path from the provided files list.
- If you cannot justify a line number, use null.
"""

    payload = _compact_payload(
        signal_counts=signal_counts,
        file_summaries=file_summaries,
        max_files=max(1, min(int(max_files), 60)),
        max_excerpt_chars=max(200, min(int(max_excerpt_chars), 2000)),
    )

    return rules.strip() + "\n\nCONTEXT:\n" + json.dumps(payload, ensure_ascii=False)


async def generate_repo_findings_json(
    *,
    repo: str,
    branch: str,
    commit_sha: str | None,
    snapshot_id: int,
    signal_counts: dict[str, Any],
    file_summaries: list[dict[str, Any]],
    max_files: int = 14,
) -> dict[str, Any]:
    client = LLMClient()
    if not client.enabled():
        raise RuntimeError("LLM not enabled (set OPENAI_BASE_URL + OPENAI_MODEL, and LLM_ENABLED=true)")

    # Keep prompt under control for 4k ctx models
    prompt = build_findings_prompt(
        repo=repo,
        branch=branch,
        commit_sha=commit_sha,
        snapshot_id=snapshot_id,
        signal_counts=signal_counts,
        file_summaries=file_summaries,
        max_files=max_files,
        max_excerpt_chars=700,
    )

    # Use chat_json() so:
    # - LM Studio gets response_format "text"
    # - We parse + repair JSON on our side
    obj = await client.chat_json(
        system="You are a senior backend engineer doing a production readiness review.",
        user=prompt,
        temperature=0.1,
        max_tokens=1400,
        response_format={"type": "json_object"},  # safe: LLMClient forces text for LM Studio
        max_input_chars=int(getattr(__import__("app.config", fromlist=["settings"]).settings, "LLM_MAX_INPUT_CHARS", 12_000)),
    )

    if not isinstance(obj, dict):
        return {"findings": []}

    findings = obj.get("findings")
    if not isinstance(findings, list):
        return {"findings": []}

    # Normalize output: keep it bounded
    return {"findings": findings[:12]}
