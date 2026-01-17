# backend/app/ai/repo_tasks.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

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
    max_total_chars: int = 12_000  # roughly ~3k tokens-ish (rule of thumb)


def _trim(s: str, n: int) -> str:
    s = (s or "").replace("\r\n", "\n")
    if len(s) <= n:
        return s
    # leave a clear marker so the model “feels” truncation
    return s[: max(0, n - 20)] + "\n...<truncated>..."


# ------------------------------------------------------------
# Signal detection
# ------------------------------------------------------------

# Classic code markers (good for small, explicit tasks)
_MARKERS = [
    ("todo_count", re.compile(r"\bTODO\b", re.IGNORECASE)),
    ("fixme_count", re.compile(r"\bFIXME\b", re.IGNORECASE)),
    ("hack_count", re.compile(r"\bHACK\b", re.IGNORECASE)),
    ("xxx_count", re.compile(r"\bXXX\b", re.IGNORECASE)),
    ("bug_count", re.compile(r"\bBUG\b", re.IGNORECASE)),
    ("note_count", re.compile(r"\bNOTE\b", re.IGNORECASE)),
]

# Production-grade "work surfaces" signals:
# These are intentionally regex-friendly (LLM reads excerpts; we detect the patterns).
_PROD_SIGNALS: list[tuple[str, re.Pattern[str]]] = [
    # Security/auth
    ("auth_signal", re.compile(r"\b(auth|authentication|authorize|authorization|api[_-]?key|jwt|oauth|oidc)\b", re.IGNORECASE)),
    ("cors_signal", re.compile(r"\bcors\b", re.IGNORECASE)),
    ("csrf_signal", re.compile(r"\bcsrf\b", re.IGNORECASE)),
    ("secrets_signal", re.compile(r"\b(secret|secrets|token|private[_-]?key|password|apikey)\b", re.IGNORECASE)),
    ("input_validation_signal", re.compile(r"\b(validate|validation|pydantic|schema|sanitize|sanitiz|escape)\b", re.IGNORECASE)),

    # Reliability/perf
    ("timeout_signal", re.compile(r"\b(timeout|read_timeout|connect_timeout)\b", re.IGNORECASE)),
    ("retry_signal", re.compile(r"\b(retry|backoff|exponential[_-]?backoff)\b", re.IGNORECASE)),
    ("rate_limit_signal", re.compile(r"\b(rate[_-]?limit|throttl)\b", re.IGNORECASE)),
    ("idempotency_signal", re.compile(r"\b(idempotent|idempotency)\b", re.IGNORECASE)),

    # Observability
    ("logging_signal", re.compile(r"\b(logging|getLogger|logger\.|log\.)\b", re.IGNORECASE)),
    ("metrics_signal", re.compile(r"\b(metrics|prometheus|opentelemetry|otel|tracing|span)\b", re.IGNORECASE)),

    # Data/DB correctness
    ("db_signal", re.compile(r"\b(sqlalchemy|session|transaction|commit|rollback|alembic|migration|index|foreign key|unique)\b", re.IGNORECASE)),
    ("nplus1_signal", re.compile(r"\b(n\+1|eagerload|joinedload|selectinload)\b", re.IGNORECASE)),

    # Tests / CI
    ("tests_signal", re.compile(r"\b(pytest|unittest|test_)\b", re.IGNORECASE)),
    ("ci_signal", re.compile(r"\b(github actions|workflow|ci\b|pipelines?)\b", re.IGNORECASE)),

    # Deployment/config
    ("docker_signal", re.compile(r"\b(docker|dockerfile|compose)\b", re.IGNORECASE)),
    ("config_signal", re.compile(r"\b(env|dotenv|config|settings)\b", re.IGNORECASE)),
]


def count_markers_in_text(text: str) -> dict[str, int]:
    """
    Backwards-compatible function name, but now returns BOTH:
      - classic markers (todo/fixme/etc.)
      - production signals (timeout/retry/auth/validation/logging/etc.)
    """
    t = text or ""
    out: dict[str, int] = {k: 0 for (k, _) in _MARKERS}

    for k, rx in _MARKERS:
        out[k] = len(rx.findall(t))

    # “...” signal (unfinished logic / placeholders)
    out["dotdotdot_count"] = len(re.findall(r"\.\.\.", t))

    # Production signals
    for k, rx in _PROD_SIGNALS:
        out[k] = len(rx.findall(t))

    return out


def _signal_score(s: dict[str, Any]) -> int:
    """
    Higher score => more likely to include the file in the prompt.

    Scoring goals:
    - Put "production surfaces" in front of the model: routes, services, db, auth, infra, tests
    - Strongly reward production signals (timeouts/retries/auth/validation/logging/metrics/rate limit)
    - Still reward explicit TODO/FIXME because they create quick, concrete tasks
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

    # Production signals (presence-weighted; these are the ones that generate the best tasks)
    score += 7 * int(bool(s.get("auth_signal")))
    score += 6 * int(bool(s.get("timeout_signal")))
    score += 6 * int(bool(s.get("retry_signal")))
    score += 6 * int(bool(s.get("rate_limit_signal")))
    score += 5 * int(bool(s.get("input_validation_signal")))
    score += 4 * int(bool(s.get("logging_signal")))
    score += 4 * int(bool(s.get("metrics_signal")))
    score += 4 * int(bool(s.get("db_signal")))
    score += 3 * int(bool(s.get("tests_signal")))
    score += 2 * int(bool(s.get("ci_signal")))
    score += 2 * int(bool(s.get("docker_signal")))
    score += 2 * int(bool(s.get("config_signal")))

    # Surface-area heuristics
    if "/routes/" in path or path.endswith("main.py"):
        score += 10
    if "/services/" in path:
        score += 7
    if "/ai/" in path:
        score += 3
    if "docker" in path or "compose" in path or "github/workflows" in path or path.endswith(".yml"):
        score += 4
    if "auth" in path or "security" in path or "api_key" in path:
        score += 6
    if "/tests/" in path or path.startswith("tests/"):
        score += 4
    if "db" in path or "models" in path or "migrations" in path or "alembic" in path:
        score += 6

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

        # Include BOTH classic + production signals in payload. This helps the LLM
        # (a) choose good tasks, and (b) tag them with signal:timeout, signal:auth, etc.
        item = {
            "path": path,
            "signals": {
                # classic
                "todo": int(f.get("todo_count") or 0),
                "fixme": int(f.get("fixme_count") or 0),
                "hack": int(f.get("hack_count") or 0),
                "xxx": int(f.get("xxx_count") or 0),
                "bug": int(f.get("bug_count") or 0),
                "note": int(f.get("note_count") or 0),
                "dotdotdot": int(f.get("dotdotdot_count") or 0),
                # production
                "auth": int(f.get("auth_signal") or 0),
                "timeout": int(f.get("timeout_signal") or 0),
                "retry": int(f.get("retry_signal") or 0),
                "rate_limit": int(f.get("rate_limit_signal") or 0),
                "validation": int(f.get("input_validation_signal") or 0),
                "logging": int(f.get("logging_signal") or 0),
                "metrics": int(f.get("metrics_signal") or 0),
                "db": int(f.get("db_signal") or 0),
                "tests": int(f.get("tests_signal") or 0),
                "ci": int(f.get("ci_signal") or 0),
                "docker": int(f.get("docker_signal") or 0),
                "config": int(f.get("config_signal") or 0),
                "secrets": int(f.get("secrets_signal") or 0),
                "nplus1": int(f.get("nplus1_signal") or 0),
                "cors": int(f.get("cors_signal") or 0),
                "csrf": int(f.get("csrf_signal") or 0),
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
            "ALWAYS include at least one signal tag when applicable, e.g. signal:timeout, signal:auth, signal:validation, signal:rate_limit.",
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

    raw = await client.chat(system=SYSTEM_PROMPT, user=user_prompt, temperature=0.2, max_tokens=900)

    try:
        return _extract_json_object_lenient(raw)
    except Exception as e:
        raise RuntimeError(
            "Repo task LLM returned non-JSON or truncated JSON. "
            f"First 500 chars:\n{raw[:500]}"
        ) from e
