# backend/app/routes/repo.py
from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import RepoFile, RepoFinding
from ..services.code_signals import compute_signal_counts_full
from ..services.github_sync import sync_repo_to_snapshot

# NOTE:
# Your repo_llm_findings file currently exposes async run_llm_scan(db, snapshot_id)
# but your routes import run_llm_repo_scan. We'll support both:
try:
    from ..services.repo_llm_findings import run_llm_repo_scan  # type: ignore
except Exception:  # pragma: no cover
    run_llm_repo_scan = None  # type: ignore

try:
    from ..services.repo_llm_findings import run_llm_scan  # type: ignore
except Exception:  # pragma: no cover
    run_llm_scan = None  # type: ignore

from ..services.repo_taskgen import tasks_from_findings
from ..services.security import debug_is_allowed, require_api_key
from ..services.static_analysis import run_static_analysis_all
from ..services.ops_gaps import run_ops_gap_scan
from ..services.metrics import JOBS_TOTAL

router = APIRouter(prefix="/debug/repo", tags=["repo-debug"])


def _guard_debug(request: Request) -> None:
    if not debug_is_allowed():
        raise HTTPException(status_code=404, detail="Not found")
    require_api_key(request)


@router.post("/sync")
def debug_repo_sync(
    request: Request,
    repo: str = Query("AzRea7/OneHaven"),
    branch: str = Query("main"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Sync repo content into RepoSnapshot/RepoFile rows.

    IMPORTANT: github_sync.sync_repo_to_snapshot returns a dict DTO:
      {snapshot_id, repo, branch, commit_sha, file_count, stored_content_files, warnings}
    So we return that directly (no .id access).
    """
    _guard_debug(request)

    JOBS_TOTAL.labels(job="repo_sync", status="start").inc()
    try:
        result = sync_repo_to_snapshot(db=db, repo=repo, branch=branch)
        JOBS_TOTAL.labels(job="repo_sync", status="ok").inc()
        return result
    except Exception:
        JOBS_TOTAL.labels(job="repo_sync", status="error").inc()
        raise


@router.post("/scan_llm")
async def debug_repo_scan_llm(
    request: Request,
    snapshot_id: int = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    LLM-driven repo scan -> RepoFinding rows (category based on prompt output).
    """
    _guard_debug(request)

    if not settings.LLM_ENABLED:
        raise HTTPException(status_code=400, detail="LLM is disabled (LLM_ENABLED=false).")

    JOBS_TOTAL.labels(job="repo_scan_llm", status="start").inc()
    try:
        # Prefer the newer async API if present
        if run_llm_scan is not None:
            out = await run_llm_scan(db=db, snapshot_id=snapshot_id)
            inserted = int(out.get("inserted", 0))
            total = int(out.get("total_findings", 0))
        elif run_llm_repo_scan is not None:
            # Older sync API style
            inserted, total = run_llm_repo_scan(db=db, snapshot_id=snapshot_id)  # type: ignore[misc]
        else:
            raise HTTPException(status_code=500, detail="LLM scan function not available.")

        JOBS_TOTAL.labels(job="repo_scan_llm", status="ok").inc()
        return {"inserted": inserted, "total_findings": total}
    except Exception:
        JOBS_TOTAL.labels(job="repo_scan_llm", status="error").inc()
        raise


@router.post("/scan_static")
def debug_repo_scan_static(
    request: Request,
    snapshot_id: int = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Deterministic static analysis (ruff+mypy+bandit) -> RepoFinding rows.
    """
    _guard_debug(request)
    JOBS_TOTAL.labels(job="repo_scan_static", status="start").inc()
    try:
        out = run_static_analysis_all(db=db, snapshot_id=snapshot_id)
        JOBS_TOTAL.labels(job="repo_scan_static", status="ok").inc()
        return out
    except Exception:
        JOBS_TOTAL.labels(job="repo_scan_static", status="error").inc()
        raise


@router.post("/scan_ops")
def debug_repo_scan_ops(
    request: Request,
    snapshot_id: int = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Deterministic ops/foot-gun checks -> RepoFinding rows.
    """
    _guard_debug(request)
    JOBS_TOTAL.labels(job="repo_scan_ops", status="start").inc()
    try:
        out = run_ops_gap_scan(db=db, snapshot_id=snapshot_id)
        JOBS_TOTAL.labels(job="repo_scan_ops", status="ok").inc()
        return out
    except Exception:
        JOBS_TOTAL.labels(job="repo_scan_ops", status="error").inc()
        raise


@router.get("/findings")
def debug_repo_findings(
    request: Request,
    snapshot_id: int = Query(...),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """
    List findings for a snapshot.
    """
    _guard_debug(request)

    rows = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .order_by(RepoFinding.severity.desc(), RepoFinding.id.desc())
        .limit(500)
        .all()
    )
    return [
        {
            "id": r.id,
            "snapshot_id": r.snapshot_id,
            "path": r.path,
            "line": r.line,
            "category": r.category,
            "severity": r.severity,
            "title": r.title,
            "evidence": r.evidence,
            "recommendation": r.recommendation,
            "fingerprint": r.fingerprint,
        }
        for r in rows
    ]


@router.post("/tasks_from_findings")
def debug_repo_tasks_from_findings(
    request: Request,
    snapshot_id: int = Query(...),
    project: str = Query("haven"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Convert findings -> tasks.
    """
    _guard_debug(request)
    JOBS_TOTAL.labels(job="repo_tasks_from_findings", status="start").inc()
    try:
        out = tasks_from_findings(db=db, snapshot_id=snapshot_id, project=project)
        JOBS_TOTAL.labels(job="repo_tasks_from_findings", status="ok").inc()
        return out
    except Exception:
        JOBS_TOTAL.labels(job="repo_tasks_from_findings", status="error").inc()
        raise


@router.get("/signal_counts_full")
def debug_repo_signal_counts_full(
    request: Request,
    snapshot_id: int = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Your existing marker signals (TODO/FIXME/etc) + per-file summary.
    """
    _guard_debug(request)

    files = db.query(RepoFile).filter(RepoFile.snapshot_id == snapshot_id).all()
    return compute_signal_counts_full(files)


@router.get("/signals_summary")
def debug_repo_signals_summary(
    request: Request,
    snapshot_id: int = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    SINGLE merged "Signals Summary" endpoint.
    """
    _guard_debug(request)

    files = db.query(RepoFile).filter(RepoFile.snapshot_id == snapshot_id).all()
    marker = compute_signal_counts_full(files)

    findings = db.query(RepoFinding).filter(RepoFinding.snapshot_id == snapshot_id).all()

    by_category: dict[str, int] = defaultdict(int)
    by_severity: dict[str, int] = defaultdict(int)

    risk_buckets: dict[str, int] = defaultdict(int)

    for f in findings:
        cat = f.category or "unknown"
        by_category[cat] += 1
        by_severity[str(f.severity)] += 1

        c = cat.lower()
        if c.startswith("security/") or "security" in c:
            risk_buckets["security"] += 1
        elif "reliability" in c:
            risk_buckets["reliability"] += 1
        elif "observability" in c:
            risk_buckets["observability"] += 1
        else:
            risk_buckets["quality"] += 1

    sources = {"llm": 0, "static": 0, "ops": 0, "other": 0}
    for cat, n in by_category.items():
        c = cat.lower()
        if c.startswith("quality/ruff") or c.startswith("typing/mypy") or c.startswith("security/bandit"):
            sources["static"] += n
        elif c.startswith("ops/"):
            sources["ops"] += n
        elif c in ("observability", "security", "typing", "quality") or c.startswith("llm/"):
            sources["llm"] += n
        else:
            sources["other"] += n

    return {
        "snapshot_id": snapshot_id,
        "marker_signals": marker.get("signals", {}),
        "marker_total_files": marker.get("total_files", 0),
        "findings_total": len(findings),
        "findings_by_category": dict(sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)),
        "findings_by_severity": dict(sorted(by_severity.items(), key=lambda kv: kv[0], reverse=True)),
        "sources": sources,
        "risk_buckets": dict(sorted(risk_buckets.items(), key=lambda kv: kv[1], reverse=True)),
    }
