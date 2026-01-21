# backend/app/routes/debug.py
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import RepoFile, RepoFinding, Task
from ..services.code_signals import compute_signal_counts_full
from ..services.github_sync import sync_repo_to_snapshot
from ..services.metrics import JOBS_TOTAL
from ..services.notifier import send_email, send_webhook
from ..services.planner import generate_daily_plan
from ..services.security import debug_is_allowed, require_api_key

from ..services.repo_chunks import (
    build_embeddings_for_snapshot,
    chunk_snapshot,
    load_chunk_text,
    search_chunks,
)
from ..services.repo_findings import (
    generate_tasks_from_findings_llm,
    list_findings,
    scan_repo_findings_llm,
    tasks_from_findings,
)

router = APIRouter(prefix="/debug", tags=["debug"])


def _guard_debug(request: Request) -> None:
    """
    Enforce production safety:
    - Optionally disable debug endpoints entirely in prod
    - Require X-API-Key for access
    """
    if not debug_is_allowed():
        raise HTTPException(status_code=404, detail="Not found")

    # require_api_key accepts a header param in FastAPI DI, but also works if you pass the string directly.
    require_api_key(request.headers.get("X-API-Key"))


# -----------------------
# Existing debug utilities
# -----------------------
@router.get("/tasks/summary")
def tasks_summary(
    request: Request,
    project: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    _guard_debug(request)

    q = select(Task)
    if project and project != "__all__":
        q = q.where(Task.project == project)

    tasks = db.scalars(q.order_by(Task.id.desc()).limit(200)).all()

    open_tasks = [t for t in tasks if not t.completed]
    done_tasks = [t for t in tasks if t.completed]

    return {
        "generated_at": None,
        "project": project,
        "open_count": len(open_tasks),
        "done_count": len(done_tasks),
        "recent_open": [
            {
                "id": t.id,
                "project": t.project,
                "title": t.title,
                "due_date": t.due_date.isoformat() if t.due_date else None,
                "estimated_minutes": t.estimated_minutes,
                "completed": t.completed,
            }
            for t in open_tasks[:80]
        ],
        "recent_completed": [
            {
                "id": t.id,
                "project": t.project,
                "title": t.title,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "estimated_minutes": t.estimated_minutes,
                "completed": t.completed,
            }
            for t in done_tasks[:50]
        ],
        "projects": sorted({(t.project or "").strip() for t in tasks if (t.project or "").strip()}),
    }


@router.get("/plan/daily")
async def plan_daily(
    request: Request,
    project: str | None = Query(default=None),
    mode: str = Query(default="single"),
    db: Session = Depends(get_db),
):
    _guard_debug(request)
    return await generate_daily_plan(db=db, project=project, mode=mode)


class WebhookBody(BaseModel):
    url: str
    payload: dict


@router.post("/notify/webhook")
def notify_webhook(request: Request, body: WebhookBody):
    _guard_debug(request)
    return send_webhook(body.url, body.payload)


class EmailBody(BaseModel):
    to: str
    subject: str
    body: str


@router.post("/notify/email")
def notify_email(request: Request, body: EmailBody):
    _guard_debug(request)
    return send_email(body.to, body.subject, body.body)


# -----------------------
# Repo: sync + signals
# -----------------------
@router.post("/repo/sync")
def repo_sync(
    request: Request,
    repo: str = Query("AzRea7/OneHaven"),
    branch: str = Query("main"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _guard_debug(request)

    JOBS_TOTAL.labels(job="repo_sync", status="start").inc()
    try:
        result = sync_repo_to_snapshot(db=db, repo=repo, branch=branch)
        JOBS_TOTAL.labels(job="repo_sync", status="ok").inc()
        return result
    except Exception:
        JOBS_TOTAL.labels(job="repo_sync", status="error").inc()
        raise


@router.get("/repo/signal_counts_full")
def repo_signal_counts_full(
    request: Request,
    snapshot_id: int = Query(..., ge=1),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _guard_debug(request)
    files = db.query(RepoFile).filter(RepoFile.snapshot_id == snapshot_id).all()
    return compute_signal_counts_full(files)


@router.get("/repo/signals_summary")
def repo_signals_summary(
    request: Request,
    snapshot_id: int = Query(..., ge=1),
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


# -----------------------
# Repo: chunks + search
# -----------------------
@router.post("/repo/chunks/build")
def repo_chunks_build(
    request: Request,
    snapshot_id: int = Query(..., ge=1),
    force: bool = Query(False),
    db: Session = Depends(get_db),
):
    _guard_debug(request)
    return chunk_snapshot(db, snapshot_id, force=force)


@router.post("/repo/chunks/embed")
async def repo_chunks_embed(
    request: Request,
    snapshot_id: int = Query(..., ge=1),
    force: bool = Query(False),
    db: Session = Depends(get_db),
):
    _guard_debug(request)
    return await build_embeddings_for_snapshot(db, snapshot_id=snapshot_id, force=force)


@router.get("/repo/chunks/search")
async def repo_chunks_search(
    request: Request,
    snapshot_id: int = Query(..., ge=1),
    q: str = Query(..., min_length=2),
    top_k: int = Query(8, ge=1, le=30),
    mode: str = Query("auto"),
    path_contains: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    _guard_debug(request)
    # Return exactly what the service returns so your CLI shows: ["fts_pg", [...]] etc.
    return await search_chunks(
        db=db,
        snapshot_id=snapshot_id,
        query=q,
        top_k=top_k,
        mode=mode,
        path_contains=path_contains,
    )


@router.get("/repo/chunks/load")
def repo_chunks_load(
    request: Request,
    chunk_id: int = Query(..., ge=1),
    db: Session = Depends(get_db),
):
    _guard_debug(request)
    return load_chunk_text(db=db, chunk_id=chunk_id)


# -----------------------
# Repo: findings
# -----------------------
@router.post("/repo/scan_llm")
async def repo_scan_llm(
    request: Request,
    snapshot_id: int = Query(..., ge=1),
    max_files: int = Query(14, ge=1, le=200),
    db: Session = Depends(get_db),
):
    _guard_debug(request)

    if not settings.LLM_ENABLED:
        raise HTTPException(status_code=400, detail="LLM is disabled (LLM_ENABLED=false).")

    JOBS_TOTAL.labels(job="repo_scan_llm", status="start").inc()
    try:
        out = await scan_repo_findings_llm(db=db, snapshot_id=snapshot_id, max_files=max_files)
        JOBS_TOTAL.labels(job="repo_scan_llm", status="ok").inc()
        return out
    except Exception:
        JOBS_TOTAL.labels(job="repo_scan_llm", status="error").inc()
        raise


@router.get("/repo/findings")
def repo_findings_list(
    request: Request,
    snapshot_id: int = Query(..., ge=1),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    _guard_debug(request)
    findings = list_findings(db=db, snapshot_id=snapshot_id, limit=limit)
    return {
        "snapshot_id": snapshot_id,
        "count": len(findings),
        "findings": [
            {
                "id": f.id,
                "snapshot_id": f.snapshot_id,
                "path": f.path,
                "line": f.line,
                "category": f.category,
                "severity": f.severity,
                "title": f.title,
                "evidence": f.evidence,
                "recommendation": f.recommendation,
                "acceptance": getattr(f, "acceptance", None),
                "fingerprint": f.fingerprint,
                "is_resolved": getattr(f, "is_resolved", False),
            }
            for f in findings
        ],
    }


# -----------------------
# Repo: findings -> tasks
# -----------------------
@router.post("/repo/tasks_from_findings")
def repo_tasks_from_findings(
    request: Request,
    snapshot_id: int = Query(..., ge=1),
    project: str = Query("haven"),
    limit: int = Query(12, ge=1, le=200),
    db: Session = Depends(get_db),
):
    _guard_debug(request)
    return tasks_from_findings(db=db, snapshot_id=snapshot_id, project=project, limit=limit)


@router.post("/repo/tasks_generate")
async def repo_tasks_generate_llm(
    request: Request,
    snapshot_id: int = Query(..., ge=1),
    project: str = Query("haven"),
    max_findings: int = Query(10, ge=1, le=100),
    chunks_per_finding: int = Query(3, ge=0, le=10),
    db: Session = Depends(get_db),
):
    _guard_debug(request)
    return await generate_tasks_from_findings_llm(
        db=db,
        snapshot_id=snapshot_id,
        project=project,
        max_findings=max_findings,
        chunks_per_finding=chunks_per_finding,
    )
