# backend/app/routes/debug.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select

from ..db import get_db
from ..models import Task
from ..services.planner import generate_daily_plan
from ..services.notifier import send_webhook, send_email
from ..services.security import require_api_key, debug_is_allowed

from ..services.repo_rag import chunk_snapshot, search_chunks, load_chunk_text
from ..services.repo_findings import (
    scan_repo_findings_llm,
    list_findings,
    tasks_from_findings,
    generate_tasks_from_findings_llm,
)

router = APIRouter(prefix="/debug", tags=["debug"])


def _guard_debug(request: Request) -> None:
    """
    Enforce production safety:
    - Optionally disable debug endpoints entirely in prod
    - Require X-API-Key for access (at least in prod; configurable in require_api_key)
    """
    if not debug_is_allowed():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")

    require_api_key(request)


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
# Repo: chunks + search
# -----------------------

@router.post("/repo/chunks/build")
def repo_chunks_build(
    request: Request,
    snapshot_id: int = Query(...),
    max_chars: int = Query(1200),
    overlap: int = Query(120),
    db: Session = Depends(get_db),
):
    _guard_debug(request)
    return chunk_snapshot(db=db, snapshot_id=snapshot_id, max_chars=max_chars, overlap=overlap)


@router.get("/repo/chunks/search")
def repo_chunks_search(
    request: Request,
    snapshot_id: int = Query(...),
    q: str = Query(...),
    top_k: int = Query(8),
    mode: str = Query("auto"),
    path_contains: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    _guard_debug(request)
    return search_chunks(
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
    chunk_id: int = Query(...),
    db: Session = Depends(get_db),
):
    _guard_debug(request)
    return load_chunk_text(db=db, chunk_id=chunk_id)


# -----------------------
# Repo: findings
# -----------------------

@router.post("/repo/scan_llm")
def repo_scan_llm(
    request: Request,
    snapshot_id: int = Query(...),
    max_files: int = Query(14),
    db: Session = Depends(get_db),
):
    _guard_debug(request)
    return scan_repo_findings_llm(db=db, snapshot_id=snapshot_id, max_files=max_files)


@router.get("/repo/findings")
def repo_findings_list(
    request: Request,
    snapshot_id: int = Query(...),
    limit: int = Query(50),
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
    snapshot_id: int = Query(...),
    project: str = Query("haven"),
    limit: int = Query(12),
    db: Session = Depends(get_db),
):
    """
    Deterministic mapping (no LLM): fast + stable.
    """
    _guard_debug(request)
    return tasks_from_findings(db=db, snapshot_id=snapshot_id, project=project, limit=limit)


@router.post("/repo/tasks_generate")
async def repo_tasks_generate_llm(
    request: Request,
    snapshot_id: int = Query(...),
    project: str = Query("haven"),
    max_findings: int = Query(10),
    chunks_per_finding: int = Query(3),
    db: Session = Depends(get_db),
):
    """
    LLM + Retrieval task generation:
      Findings -> (title/category/evidence query) -> retrieve chunks -> generate tasks -> create Task rows.

    Requirements:
      - chunks must exist (run /debug/repo/chunks/build first)
      - LLM_ENABLED + model configured
    """
    _guard_debug(request)
    return await generate_tasks_from_findings_llm(
        db=db,
        snapshot_id=snapshot_id,
        project=project,
        max_findings=max_findings,
        chunks_per_finding=chunks_per_finding,
    )
