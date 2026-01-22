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

    payload: dict[str, Any] = {
        "repo": repo_name,
        "branch": branch,
        "commit_sha": commit_sha,
        "snapshot_id": snapshot_id,
        "signal_counts": signal_counts,
        "evidence_files": evidence_files,
        "instructions": [
            'Return a JSON OBJECT with top-level key "tasks". Do NOT return a bare JSON array.',
            "Create at most 4 tasks.",
            "Create exactly ONE task per [FINDING] in extra_evidence if present.",
            "Each task MUST cite evidence using [EVIDENCE path:Lx-Ly].",
            "Each task MUST include a runnable verification command in DoD (pytest or curl).",
            "Prefer high-impact fixes (auth/security, secrets, timeouts/retries, validation, DB integrity, error handling, tests).",
            "Avoid pure style/lint unless there are no other issues.",
            "Do not invent files not provided.",
            "Return JSON only. No markdown. No code fences.",
            "Keep notes <= 800 chars. Keep dod <= 240 chars. Keep starter <= 140 chars.",
        ],
    }

    if extra_evidence:
        payload["extra_evidence"] = extra_evidence

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*```$", "", s).strip()
    return s


def _balanced_json_span(text: str, start_idx: int) -> tuple[int, int] | None:
    if start_idx < 0 or start_idx >= len(text):
        return None

    opener = text[start_idx]
    if opener not in "{[":
        return None

    stack = [opener]
    i = start_idx + 1
    in_str = False
    esc = False

    while i < len(text):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            i += 1
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            top = stack[-1]
            expected = "}" if top == "{" else "]"
            if ch != expected:
                return None
            stack.pop()
            if not stack:
                return (start_idx, i)
        i += 1

    return None


def _repair_truncated_json(s: str, start_idx: int) -> str:
    prefix = s[:start_idx]
    body = s[start_idx:]

    if not body or body[0] not in "{[":
        return s

    stack: list[str] = [body[0]]
    in_str = False
    esc = False
    i = 1

    while i < len(body):
        ch = body[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            i += 1
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                top = stack[-1]
                expected = "}" if top == "{" else "]"
                if ch == expected:
                    stack.pop()
                    if not stack:
                        return prefix + body
        i += 1

    repaired = body
    if in_str:
        if esc:
            repaired += "\\\\"
        repaired += '"'

    while stack:
        top = stack.pop()
        repaired += "}" if top == "{" else "]"

    return prefix + repaired


def _json_loads_best_effort(raw: str) -> Any:
    s = _strip_code_fences(str(raw or ""))

    if s.startswith("{") or s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            repaired = _repair_truncated_json(s, 0)
            return json.loads(repaired)

    first_obj = s.find("{")
    first_arr = s.find("[")
    starts = [i for i in (first_obj, first_arr) if i != -1]
    if not starts:
        raise ValueError(f"No JSON object/array found. First 400 chars:\n{s[:400]}")

    start = min(starts)

    span = _balanced_json_span(s, start)
    if span:
        cand = s[span[0] : span[1] + 1]
        return json.loads(cand)

    repaired = _repair_truncated_json(s, start)
    return json.loads(repaired[start:])


def _normalize_tasks_schema(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        if isinstance(obj.get("tasks"), list):
            return {"tasks": [x for x in obj["tasks"] if isinstance(x, dict)]}
        for alt in ("items", "suggestions", "results"):
            if isinstance(obj.get(alt), list):
                return {"tasks": [x for x in obj[alt] if isinstance(x, dict)]}
        if any(k in obj for k in ("title", "notes", "dod", "starter")):
            return {"tasks": [obj]}
        return {"tasks": []}

    if isinstance(obj, list):
        return {"tasks": [x for x in obj if isinstance(x, dict)]}

    return {"tasks": []}


def _clean_path(path_val: Any) -> str:
    if path_val is None:
        return "unknown"
    s = str(path_val).strip()
    if not s:
        return "unknown"
    low = s.lower()

    if low in ("null", "none", "false", "true"):
        return "unknown"

    if low.endswith("/main") or low.endswith("/master"):
        return "unknown"
    if low.count("/") <= 1 and "." not in low:
        return "unknown"
    if "." not in low.split("/")[-1]:
        return "unknown"

    return s[:600]


def _coerce_task_fields(t: dict[str, Any]) -> dict[str, Any]:
    title = str(t.get("title") or "").strip()[:240] or "Untitled repo task"
    notes = str(t.get("notes") or "").strip()[:4000]
    tags = str(t.get("tags") or "repo,autogen").strip()[:300]

    path = _clean_path(t.get("path"))

    starter = str(t.get("starter") or "").strip()[:1000]
    dod = str(t.get("dod") or "").strip()[:1000]

    if not starter:
        starter = "Open the linked code chunk and identify the failing behavior (5 min)."
    if not dod:
        dod = "Fix implemented and verified with pytest -q or curl repro."

    pr = t.get("priority", 3)
    try:
        pr_i = int(pr)
    except Exception:
        pr_i = 3
    pr_i = max(1, min(5, pr_i))

    em = t.get("estimated_minutes", 60)
    try:
        em_i = int(em)
    except Exception:
        em_i = 60
    em_i = max(15, min(240, em_i))

    blocks_me = bool(t.get("blocks_me", False))

    line = t.get("line")
    line_i: int | None = int(line) if isinstance(line, int) else None

    return {
        "title": title,
        "notes": notes,
        "tags": tags,
        "priority": pr_i,
        "estimated_minutes": em_i,
        "blocks_me": blocks_me,
        "path": path,
        "line": line_i,
        "starter": starter,
        "dod": dod,
    }


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
    from .llm import LLMClient

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

    raw = await client.chat(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.2,
        max_tokens=750,
        response_format={"type": "json_object"},
    )

    parsed = _json_loads_best_effort(raw)
    data = _normalize_tasks_schema(parsed)

    normalized_tasks: list[dict[str, Any]] = []
    for t in data.get("tasks", []):
        if not isinstance(t, dict):
            continue
        normalized_tasks.append(_coerce_task_fields(t))

    return {"tasks": normalized_tasks}
