# backend/app/services/repo_findings.py
from __future__ import annotations

import hashlib
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..ai.repo_findings import generate_repo_findings_json
from ..models import RepoFile, RepoFinding, RepoSnapshot

# IMPORTANT:
# Avoid circular imports. Do NOT import repo_taskgen at module import time.
# If you need something from repo_taskgen, import it inside a function.


# ----------------------------
# Public helper: stable repo link builder
# ----------------------------
def _repo_link(snapshot: RepoSnapshot, path: str, line: int | None = None) -> str:
    """
    Build a stable GitHub link for a file (and optional line).
    Uses snapshot.repo + snapshot.branch. commit_sha may be null; branch link is fine.
    """
    repo = (snapshot.repo or "").strip()
    branch = (snapshot.branch or "main").strip()
    p = (path or "").lstrip("/")

    base = f"https://github.com/{repo}/blob/{branch}/{p}"
    if isinstance(line, int) and line > 0:
        return f"{base}#L{line}"
    return base


# ----------------------------
# Helpers
# ----------------------------
def _fingerprint(snapshot_id: int, path: str, title: str) -> str:
    h = hashlib.sha256()
    h.update(f"{snapshot_id}|{path}|{title}".encode("utf-8", errors="ignore"))
    return h.hexdigest()[:64]


def _norm_line(v: Any) -> int | None:
    if v is None:
        return None
    try:
        n = int(v)
    except Exception:
        return None
    return n if n > 0 else None


def _norm_severity(v: Any) -> int:
    try:
        n = int(v)
    except Exception:
        n = 3
    return max(1, min(5, n))


def _norm_text(v: Any, max_len: int) -> str:
    s = str(v or "").strip()
    return s[:max_len] if s else ""


def _norm_category(v: Any) -> str:
    raw = str(v or "").strip().lower()
    mapping = {
        "security": "security",
        "auth": "auth",
        "secrets": "secrets",
        "reliability": "reliability",
        "timeout": "timeouts",
        "timeouts": "timeouts",
        "retry": "retries",
        "retries": "retries",
        "validation": "validation",
        "db": "db",
        "database": "db",
        "api": "api",
        "tests": "tests",
        "ops": "ops",
        "observability": "observability",
        "perf": "perf",
        "performance": "perf",
        "style": "style",
        "lint": "style",
        "formatting": "style",
    }
    out = mapping.get(raw, raw)
    allowed = {
        "security",
        "auth",
        "secrets",
        "reliability",
        "timeouts",
        "retries",
        "validation",
        "db",
        "api",
        "tests",
        "ops",
        "observability",
        "perf",
        "style",
    }
    return out if out in allowed else "reliability"


def _norm_acceptance(v: Any) -> str:
    s = str(v or "").strip()
    low = s.lower()
    if not s or low in {"false", "true", "null", "none"} or len(s) < 12:
        return (
            "Fix implemented and verified via a concrete command "
            "(pytest -q or curl repro) that fails before and passes after."
        )
    return s[:400]


def _clamp_limit(limit: Any, *, default: int, max_limit: int) -> int:
    """
    Defensive clamp: route layer should validate, but service layer must be safe too.
    """
    if limit is None:
        return default
    try:
        n = int(limit)
    except Exception:
        return default
    return max(1, min(max_limit, n))


def _pick_file_summaries(
    db: Session,
    snapshot_id: int,
    *,
    max_files: int = 24,
) -> list[dict[str, Any]]:
    """
    Build compact {path, excerpt} list for the LLM scan.
    """
    files = db.scalars(
        select(RepoFile)
        .where(RepoFile.snapshot_id == snapshot_id)
        .where(RepoFile.content_kind == "text")
    ).all()

    if not files:
        return []

    keywords = [
        "x-api-key",
        "authorization",
        "auth",
        "token",
        "secret",
        "password",
        "apikey",
        "timeout",
        "retry",
        "validate",
        "pydantic",
        "sql",
        "session",
        "commit",
        "rollback",
        "exception",
        "logging",
        "logger",
        "cors",
        "csrf",
        "/debug",
        "rate limit",
        "429",
    ]

    scored: list[tuple[int, RepoFile]] = []
    for f in files:
        text = (getattr(f, "content_text", None) or f.content or "")
        if not text:
            continue

        low = text.lower()
        score = 0
        for k in keywords:
            if k in low:
                score += 3

        p = (f.path or "").lower()
        if any(x in p for x in ("auth", "security", "middleware", "deps", "config", "settings", "routes", "db")):
            score += 2
        if "debug" in p:
            score += 2

        scored.append((score, f))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [f for s, f in scored[:max_files] if s > 0] or [f for _, f in scored[:max_files]]

    out: list[dict[str, Any]] = []
    for f in picked:
        text = (getattr(f, "content_text", None) or f.content or "")
        if not text:
            continue
        out.append({"path": f.path, "excerpt": text[:1400]})
    return out


# ----------------------------
# Public API used by routes/debug.py
# ----------------------------
def list_repo_findings(db: Session, snapshot_id: int, *, limit: int = 200) -> list[RepoFinding]:
    """
    List findings for a snapshot, newest/highest severity first, capped by `limit`.

    Why cap here (not only in routes):
    - prevents accidental unbounded reads
    - keeps UI endpoints fast
    - avoids memory spikes
    """
    limit_n = _clamp_limit(limit, default=200, max_limit=2000)

    return list(
        db.scalars(
            select(RepoFinding)
            .where(RepoFinding.snapshot_id == snapshot_id)
            .order_by(RepoFinding.severity.desc(), RepoFinding.id.desc())
            .limit(limit_n)
        ).all()
    )


def compute_signal_counts(db: Session, snapshot_id: int) -> dict[str, Any]:
    """
    Back-compat helper.
    If you have a canonical signals module elsewhere, you can redirect this later.
    """
    files = db.scalars(
        select(RepoFile)
        .where(RepoFile.snapshot_id == snapshot_id)
        .where(RepoFile.content_kind == "text")
    ).all()

    # Import lazily to avoid circular imports.
    try:
        from .code_signals import compute_signal_counts_for_files  # type: ignore
        return compute_signal_counts_for_files(files)
    except Exception:
        # Minimal fallback
        return {"total_files": len(files), "signals": {}}


async def run_llm_scan(db: Session, snapshot_id: int, *, max_files: int = 24) -> dict[str, Any]:
    """
    Canonical LLM findings scan:
      - builds file_summaries from stored RepoFile content
      - calls AI generator
      - inserts RepoFinding rows (deduped by fingerprint)
    """
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        raise ValueError(f"snapshot_id {snapshot_id} not found")

    signal_counts = compute_signal_counts(db, snapshot_id)
    file_summaries = _pick_file_summaries(db, snapshot_id, max_files=int(max_files))

    raw = await generate_repo_findings_json(
        repo=snap.repo,
        branch=snap.branch,
        commit_sha=snap.commit_sha,
        snapshot_id=snapshot_id,
        signal_counts=signal_counts,
        file_summaries=file_summaries,
    )

    findings = raw.get("findings") if isinstance(raw, dict) else None
    if not isinstance(findings, list):
        findings = []

    inserted = 0
    for f in findings[:12]:
        if not isinstance(f, dict):
            continue

        path = _norm_text(f.get("path"), 1024)
        if not path:
            continue

        title = _norm_text(f.get("title"), 240) or "Repo finding"
        fp = _fingerprint(snapshot_id, path, title)

        existing = db.scalar(
            select(RepoFinding.id).where(
                RepoFinding.snapshot_id == snapshot_id,
                RepoFinding.fingerprint == fp,
            )
        )
        if existing:
            continue

        db.add(
            RepoFinding(
                snapshot_id=snapshot_id,
                path=path,
                line=_norm_line(f.get("line")),
                category=_norm_category(f.get("category")),
                severity=_norm_severity(f.get("severity")),
                title=title,
                evidence=_norm_text(f.get("evidence"), 1200),
                recommendation=_norm_text(f.get("recommendation"), 1200),
                acceptance=_norm_acceptance(f.get("acceptance")),
                fingerprint=fp,
                is_resolved=False,
            )
        )
        inserted += 1

    db.commit()
    return {
        "snapshot_id": snapshot_id,
        "inserted": inserted,
        "total_findings": len(list_repo_findings(db, snapshot_id, limit=5000)),  # count-ish
    }


# --------------------------------------------------------------------
# ✅ Compatibility aliases (routes/debug.py may expect these names)
# --------------------------------------------------------------------
async def scan_repo_findings_llm(
    db: Session,
    snapshot_id: int,
    *,
    max_files: int = 24,
) -> dict[str, Any]:
    return await run_llm_scan(db, snapshot_id, max_files=max_files)


def list_findings(db: Session, snapshot_id: int, limit: int = 200) -> list[RepoFinding]:
    """
    Back-compat name used by routes/debug.py.
    IMPORTANT: must accept `limit` because debug route passes it.
    """
    return list_repo_findings(db, snapshot_id, limit=limit)


def tasks_from_findings(db: Session, snapshot_id: int, project: str, limit: int = 12) -> dict[str, Any]:
    """
    Convert findings -> tasks via repo_taskgen implementation, with optional cap.

    Why limit matters:
    - tasks endpoint should be predictable and cheap
    - avoids generating/creating too many tasks from a huge scan
    """
    limit_n = _clamp_limit(limit, default=12, max_limit=200)

    # Lazy import to avoid circular import
    from .repo_taskgen import tasks_from_findings as _impl  # type: ignore

    # Prefer passing limit through if the implementation supports it
    try:
        out = _impl(db=db, snapshot_id=snapshot_id, project=project, limit=limit_n)
    except TypeError:
        out = _impl(db=db, snapshot_id=snapshot_id, project=project)

    # If impl returns a dict that contains a list, enforce limit anyway (defensive)
    if isinstance(out, dict):
        if isinstance(out.get("findings"), list):
            out["findings"] = out["findings"][:limit_n]
            out["count"] = len(out["findings"])
        if isinstance(out.get("tasks"), list):
            out["tasks"] = out["tasks"][:limit_n]
            out["count"] = len(out["tasks"])
    return out


async def generate_tasks_from_findings_llm(db: Session, snapshot_id: int, project: str) -> dict[str, Any]:
    # Kept for compatibility; routes/debug.py also has a separate /repo/tasks_generate path.
    return tasks_from_findings(db, snapshot_id, project)
