# backend/app/ai/repo_tasks.py
# backend/app/ai/repo_tasks.py
from __future__ import annotations

import json
import re
from typing import Any, Optional


SYSTEM_PROMPT = """You are a senior engineering manager generating tasks from repository evidence.

You MUST return ONLY valid JSON (no markdown, no backticks, no commentary).
Your output MUST be a JSON OBJECT with top-level key "tasks".

Schema:
{
  "tasks": [
    {
      "title": "short, concrete",
      "notes": "Must include evidence citations like: [EVIDENCE path:Lx-Ly] ...",
      "tags": "comma,separated,tags",
      "priority": 1-5,
      "estimated_minutes": 15-240,
      "blocks_me": true|false,
      "path": "repo/relative/path.ext",
      "line": 123 | null,
      "starter": "2-5 min first action",
      "dod": "definition of done with a runnable verification command (pytest or curl)"
    }
  ]
}

Hard rules:
- Output MUST be an OBJECT. Do NOT output a bare JSON array.
- Use ONLY file paths present in evidence_files[*].path.
- Each evidence excerpt may begin with a line like: [FINDING id=123 category=auth severity=4]
  When present, you MUST create exactly ONE task per unique FINDING id, and include tag: finding:123
- Prefer security/reliability/correctness/observability/tests over style.
- For EACH task, choose ONE primary chunk from evidence_files and:
  - set task.path to that chunk path
  - set task.line to the START line shown in that chunk header if available
- notes MUST include 1-3 citations in this exact format:
  [EVIDENCE <path>:L<start>-L<end>] <1 sentence why it matters>
- dod MUST include at least one runnable verification command:
  - pytest -q ...
  - curl -i ...
- Include tags: repo,autogen and (when relevant) signal:*.
- Keep strings short to avoid truncation.
"""


def build_prompt(
    *,
    repo_name: str,
    branch: str,
    commit_sha: str | None,
    snapshot_id: int,
    signal_counts: dict[str, Any],
    file_summaries: list[dict[str, Any]],
    extra_evidence: Optional[list[dict[str, Any]]] = None,
) -> str:
    evidence_files: list[dict[str, Any]] = []
    running_chars = 0
    hard_cap_chars = 20_000

    for fs in file_summaries or []:
        item = {
            "path": str(fs.get("path") or "")[:600],
            "excerpt": str(fs.get("excerpt") or "")[:1600],
        }
        for k, v in (fs or {}).items():
            if k.startswith("signal:"):
                item[k] = v

        item_json = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if running_chars + len(item_json) > hard_cap_chars:
            break
        evidence_files.append(item)
        running_chars += len(item_json)

    for ev in extra_evidence or []:
        item = {
            "path": str(ev.get("path") or "")[:600],
            "excerpt": str(ev.get("excerpt") or "")[:1600],
        }
        item_json = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if running_chars + len(item_json) > hard_cap_chars:
            break
        evidence_files.append(item)
        running_chars += len(item_json)

    payload = {
        "repo": repo_name,
        "branch": branch,
        "commit_sha": commit_sha,
        "snapshot_id": snapshot_id,
        "signal_counts": signal_counts,
        "evidence_files": evidence_files,
    }

    return (
        SYSTEM_PROMPT
        + "\n\nINPUT_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
