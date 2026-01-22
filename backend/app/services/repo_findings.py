# backend/app/services/repo_findings.py
from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..ai.repo_findings import generate_repo_findings_json
from ..models import RepoFile, RepoFinding, RepoSnapshot
from .repo_taskgen import compute_signal_counts  # ✅ exists
from .repo_taskgen import tasks_from_findings as _tasks_from_findings  # ✅ exists


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


def _pick_file_summaries(db: Session, snapshot_id: int, *, max_files: int = 24) -> list[dict[str, Any]]:
    """
    Build compact {path, excerpt} list for the LLM scan.

    This replaces the missing select_repo_task_files() dependency.
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

def list_repo_findings(db: Session, snapshot_id: int) -> list[RepoFinding]:
    return list(
        db.scalars(
            select(RepoFinding)
            .where(RepoFinding.snapshot_id == snapshot_id)
            .order_by(RepoFinding.severity.desc(), RepoFinding.id.desc())
        ).all()
    )


async def run_llm_scan(db: Session, snapshot_id: int) -> dict[str, Any]:
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
    file_summaries = _pick_file_summaries(db, snapshot_id, max_files=24)

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
        "total_findings": len(list_repo_findings(db, snapshot_id)),
    }


# --------------------------------------------------------------------
# ✅ Compatibility aliases (routes/debug.py expects these names)
# --------------------------------------------------------------------

async def scan_repo_findings_llm(db: Session, snapshot_id: int) -> dict[str, Any]:
    """
    Back-compat alias for debug routes.
    """
    return await run_llm_scan(db, snapshot_id)


def list_findings(db: Session, snapshot_id: int) -> list[RepoFinding]:
    """
    Back-compat alias for debug routes.
    """
    return list_repo_findings(db, snapshot_id)


def tasks_from_findings(db: Session, snapshot_id: int, project: str) -> dict[str, Any]:
    """
    Back-compat alias for debug routes.

    routes/debug.py imports tasks_from_findings from this module.
    The real implementation lives in services/repo_taskgen.py.
    """
    return _tasks_from_findings(db=db, snapshot_id=snapshot_id, project=project)


async def generate_tasks_from_findings_llm(db: Session, snapshot_id: int, project: str) -> dict[str, Any]:
    """
    Back-compat alias for debug routes.

    Some older code imports generate_tasks_from_findings_llm; keep it stable too.
    """
    return _tasks_from_findings(db=db, snapshot_id=snapshot_id, project=project)
