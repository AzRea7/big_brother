# backend/app/services/repo_llm_findings.py
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..ai.llm import chat_completion_json
from ..config import settings
from ..models import RepoFile, RepoFinding, Task

# Prompt import: keep stable even if your prompts module changes.
from ..ai.prompts import REPO_FINDINGS_SYSTEM  # type: ignore

try:
    from ..ai.prompts import repo_findings_user  # type: ignore
except Exception:

    def repo_findings_user(context: str) -> str:
        return (
            "Return STRICT JSON only with schema:\n"
            "{\n"
            '  "findings": [\n'
            "    {\n"
            '      "category": "security|auth|reliability|correctness|observability|performance|testing|maintainability|docs",\n'
            '      "severity": 1-5,\n'
            '      "title": "short title",\n'
            '      "path": "repo-relative file path",\n'
            '      "line": 1-or-null,\n'
            '      "evidence": "what you saw (brief)",\n'
            '      "recommendation": "what to change",\n'
            '      "acceptance": "how to verify (concrete command/check) - MUST be a sentence"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Keep findings <= 8.\n"
            "- Prefer auth/security, secrets, DB integrity, retries/timeouts, validation, error-handling, tests.\n"
            "- Avoid style-only findings unless nothing else exists.\n"
            "- acceptance MUST NOT be true/false/null.\n\n"
            "Context:\n"
            f"{context}"
        )


_SEVERITY_MAP = {
    "low": 2,
    "med": 3,
    "medium": 3,
    "high": 4,
    "critical": 5,
}

_ALLOWED_CATEGORIES = {
    "security",
    "auth",
    "reliability",
    "correctness",
    "observability",
    "performance",
    "testing",
    "maintainability",
    "docs",
}

_BAD_LITERAL_STRINGS = {"false", "true", "null", "none", "n/a", "na"}

_HIGH_SIGNAL_HINTS = [
    # auth/security
    "auth",
    "security",
    "token",
    "jwt",
    "oauth",
    "api_key",
    "x-api-key",
    "authorization",
    "bearer",
    "secret",
    "password",
    "cors",
    "csrf",
    # reliability
    "timeout",
    "retry",
    "backoff",
    "rate limit",
    "ratelimit",
    "throttle",
    "429",
    "circuit",
    # db/data
    "alembic",
    "migration",
    "transaction",
    "commit",
    "rollback",
    "constraint",
    "foreign key",
    # api correctness
    "validation",
    "pydantic",
    "fastapi",
    "status_code",
    "exception",
    # observability/tests
    "logging",
    "logger",
    "metrics",
    "prometheus",
    "trace",
    "request_id",
    "pytest",
    "test_",
]
_HINT_RE = re.compile("|".join(re.escape(x) for x in _HIGH_SIGNAL_HINTS), re.IGNORECASE)


def _fingerprint(f: dict[str, Any]) -> str:
    key = "|".join(
        [
            str(f.get("category") or ""),
            str(f.get("severity") or ""),
            str(f.get("title") or ""),
            str(f.get("file_path") or f.get("path") or ""),
            str(f.get("line_start") or f.get("line") or ""),
            str(f.get("line_end") or ""),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:64]


def _coerce_severity(x: Any) -> int:
    if x is None:
        return 3
    if isinstance(x, (int, float)):
        v = int(x)
        return max(1, min(5, v))

    s = str(x).strip().lower()
    if not s:
        return 3
    if s in _SEVERITY_MAP:
        return _SEVERITY_MAP[s]

    try:
        v = int(float(s))
        return max(1, min(5, v))
    except Exception:
        return 3


def _normalize_acceptance(a: Any) -> str:
    s = str(a or "").strip()
    if not s or s.lower() in _BAD_LITERAL_STRINGS or len(s) < 8:
        return (
            "Acceptance: fix implemented and verified with a concrete check "
            "(pytest for affected area and/or a curl reproduction for the endpoint)."
        )
    return s[:1200]


def _clean_text(v: Any, *, max_len: int) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in _BAD_LITERAL_STRINGS:
        return None
    return s[:max_len]


def _normalize_category(cat: Any) -> str:
    s = str(cat or "").strip().lower()
    if not s or s in _BAD_LITERAL_STRINGS:
        return "maintainability"

    alias_map = {
        "authentication": "auth",
        "authorization": "auth",
        "sec": "security",
        "stability": "reliability",
        "bug": "correctness",
        "logging": "observability",
        "metrics": "observability",
        "perf": "performance",
        "test": "testing",
        "tests": "testing",
        "documentation": "docs",
        "refactor": "maintainability",
        "style": "maintainability",
        "lint": "maintainability",
        "formatting": "maintainability",
    }
    s = alias_map.get(s, s)
    if s not in _ALLOWED_CATEGORIES:
        return "maintainability"
    return s


def _apply_guardrails(f: dict[str, Any]) -> dict[str, Any]:
    title = (f.get("title") or "").lower()
    evidence = (f.get("evidence") or "").lower()
    rec = (f.get("recommendation") or "").lower()
    path = (f.get("path") or "").lower()
    cat = (f.get("category") or "").lower()

    joined = " ".join([title, evidence, rec, path, cat])
    sev = int(f.get("severity") or 3)

    security_tokens = [
        "api key",
        "x-api-key",
        "authorization",
        "bearer",
        "jwt",
        "csrf",
        "sql injection",
        "injection",
        "secret",
        "password",
        "token",
        "cors",
        "auth",
        "login",
        "session",
        "oauth",
    ]
    if any(t in joined for t in security_tokens):
        sev = max(sev, 3)
        if any(p in path for p in ["auth", "security", "middleware", "deps.py", "oauth", "jwt"]):
            sev = max(sev, 4)

    style_tokens = ["trailing whitespace", "whitespace", "formatting", "line too long", "rename variable", "typo"]
    if any(t in joined for t in style_tokens) and not any(t in joined for t in security_tokens):
        sev = min(sev, 2)

    f["severity"] = max(1, min(5, sev))
    return f


def _file_signal_score(rf: RepoFile) -> int:
    score = 0
    path = (getattr(rf, "path", "") or "").lower()
    if path:
        if any(x in path for x in ["auth", "security", "middleware", "deps", "oauth", "jwt"]):
            score += 8
        if any(x in path for x in ["db", "models", "migrations", "alembic"]):
            score += 6
        if any(x in path for x in ["routes", "api", "routers", "main.py"]):
            score += 4
        if any(x in path for x in ["test", "tests"]):
            score += 3

    text = (getattr(rf, "content_text", None) or "")
    if text:
        head = text[:5000]
        score += len(_HINT_RE.findall(head))

    size = int(getattr(rf, "size", 0) or 0)
    score += min(5, size // 20_000)
    return score


def _pick_top_files(db: Session, snapshot_id: int, *, max_files: Optional[int] = None) -> list[RepoFile]:
    q = (
        db.query(RepoFile)
        .filter(RepoFile.snapshot_id == snapshot_id)
        .filter(RepoFile.content_text.isnot(None))
    )
    files = q.all()

    scored: list[tuple[int, int, RepoFile]] = []
    for rf in files:
        scored.append((_file_signal_score(rf), int(getattr(rf, "size", 0) or 0), rf))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    if max_files is None:
        max_files = int(getattr(settings, "REPO_TASK_MAX_FILES", 18) or 18)

    # Hard clamp: prompt budgets exist
    n = max(10, min(60, int(max_files)))
    return [rf for _, _, rf in scored[:n]]


def _build_prompt_context(files: list[RepoFile]) -> str:
    chunks: list[str] = []
    total = 0

    max_excerpt = int(getattr(settings, "REPO_TASK_EXCERPT_CHARS", 800) or 800)
    max_total = int(getattr(settings, "REPO_TASK_MAX_TOTAL_CHARS", 12_000) or 12_000)

    for rf in files:
        excerpt = (rf.content_text or "")[:max_excerpt]
        block = f"\n--- FILE: {rf.path}\n{excerpt}\n"
        if total + len(block) > max_total:
            break
        chunks.append(block)
        total += len(block)

    return "".join(chunks)


def _normalize_finding(f: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(f, dict):
        return None

    title = str(f.get("title") or "").strip()
    if not title:
        return None

    path = str((f.get("file_path") or f.get("path") or "")).strip()
    if not path or path.lower() == "null":
        return None

    sev = _coerce_severity(f.get("severity"))
    cat = _normalize_category(f.get("category"))

    line = None
    line_start = f.get("line_start") or f.get("line")
    if line_start is not None:
        try:
            line = int(line_start)
        except Exception:
            line = None

    out = {
        "category": cat[:48],
        "severity": int(sev),
        "title": title[:240],
        "path": path,
        "line": line,
        "evidence": _clean_text(f.get("evidence"), max_len=1600),
        "recommendation": _clean_text(f.get("recommendation"), max_len=1600),
        "acceptance": _normalize_acceptance(f.get("acceptance")),
    }
    return _apply_guardrails(out)


async def scan_snapshot_to_findings(db: Session, snapshot_id: int, *, max_files: Optional[int] = None) -> dict[str, Any]:
    files = _pick_top_files(db, snapshot_id, max_files=max_files)
    context = _build_prompt_context(files)

    policy = (
        "PRIORITY POLICY:\n"
        "- Prefer: auth/security, secrets, DB integrity, retries/timeouts, validation, error handling, tests.\n"
        "- Avoid style/lint unless there are no other issues.\n"
        "- Keep findings <= 8.\n"
        "- Keep strings short.\n"
        "- acceptance MUST be a human sentence (not true/false).\n"
        "\n"
    )

    resp = await chat_completion_json(
        system=REPO_FINDINGS_SYSTEM,
        user=repo_findings_user(policy + context),
    )

    findings = resp.get("findings") or []
    inserted = 0

    for f_raw in findings:
        if not isinstance(f_raw, dict):
            continue

        f = _normalize_finding(f_raw)
        if not f:
            continue

        fp = _fingerprint(
            {
                "category": f["category"],
                "severity": f["severity"],
                "title": f["title"],
                "file_path": f["path"],
                "line_start": f["line"] or "",
                "line_end": "",
            }
        )

        exists = (
            db.query(RepoFinding)
            .filter(RepoFinding.snapshot_id == snapshot_id)
            .filter(RepoFinding.fingerprint == fp)
            .first()
        )
        if exists:
            continue

        db.add(
            RepoFinding(
                snapshot_id=snapshot_id,
                path=f["path"],
                line=f["line"],
                category=f["category"],
                severity=int(f["severity"]),
                title=f["title"],
                evidence=f["evidence"],
                recommendation=f["recommendation"],
                acceptance=f["acceptance"],
                fingerprint=fp,
                created_at=datetime.utcnow(),
            )
        )
        inserted += 1

    db.commit()
    total = db.query(RepoFinding).filter(RepoFinding.snapshot_id == snapshot_id).count()
    return {"snapshot_id": snapshot_id, "inserted": inserted, "total_findings": int(total), "mode": "repo_llm_findings"}


async def run_llm_scan(db: Session, snapshot_id: int, *, max_files: Optional[int] = None) -> dict[str, Any]:
    return await scan_snapshot_to_findings(db, snapshot_id, max_files=max_files)


async def run_llm_repo_scan(db: Session, snapshot_id: int, *, max_files: Optional[int] = None) -> dict[str, Any]:
    return await run_llm_scan(db, snapshot_id, max_files=max_files)


def list_findings(db: Session, snapshot_id: int, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    q = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .order_by(RepoFinding.id.desc())
    )
    rows = q.offset(offset).limit(limit).all()
    total = db.query(RepoFinding).filter(RepoFinding.snapshot_id == snapshot_id).count()

    return {
        "snapshot_id": snapshot_id,
        "count": int(total),
        "findings": [
            {
                "id": r.id,
                "snapshot_id": r.snapshot_id,
                "path": r.path,
                "line": r.line,
                "category": r.category,
                "severity": _coerce_severity(r.severity),
                "title": r.title,
                "evidence": r.evidence,
                "recommendation": r.recommendation,
                "acceptance": getattr(r, "acceptance", None),
                "fingerprint": r.fingerprint,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


def tasks_from_findings(db: Session, snapshot_id: int, project: str, *, limit: int = 12) -> dict[str, Any]:
    try:
        n = int(limit)
    except Exception:
        n = 12
    n = max(1, min(200, n))

    findings = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .order_by(RepoFinding.severity.desc(), RepoFinding.id.desc())
        .limit(n)
        .all()
    )

    created = 0
    skipped = 0

    for f in findings:
        fp_tag = f"finding:{f.fingerprint}"

        existing = (
            db.query(Task)
            .filter(Task.project == project)
            .filter(Task.tags.like(f"%{fp_tag}%"))
            .first()
        )
        if existing:
            skipped += 1
            continue

        sev = _coerce_severity(f.severity)
        pri = int(max(1, min(5, sev)))

        link = f"repo://snapshot/{snapshot_id}#{f.path}" + (f":L{f.line}" if f.line else "")
        tags = ",".join(["repo", "autogen", f"category:{f.category}", f"severity:{sev}", fp_tag])

        notes_parts: list[str] = []
        if f.evidence:
            notes_parts.append(f"Evidence:\n{f.evidence}")
        if f.recommendation:
            notes_parts.append(f"Recommendation:\n{f.recommendation}")

        acceptance = _normalize_acceptance(getattr(f, "acceptance", None))
        notes_parts.append(acceptance)

        notes = "\n\n".join(notes_parts) if notes_parts else "Generated from repo findings."

        db.add(
            Task(
                title=f"[RepoFinding] {f.title}",
                notes=notes,
                priority=pri,
                estimated_minutes=90 if pri >= 4 else 45,
                blocks_me=(pri >= 5),
                project=project,
                tags=tags,
                link=link,
                starter="Open the file and reproduce/confirm the issue in 2–5 minutes.",
                dod=acceptance,
                created_at=datetime.utcnow(),
                completed=False,
            )
        )
        created += 1

    db.commit()
    return {"snapshot_id": snapshot_id, "project": project, "created": created, "skipped": skipped}
