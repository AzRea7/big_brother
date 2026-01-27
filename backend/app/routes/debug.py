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

# NEW: Level 3 diff generation + workflow
from ..services.patch_generator import generate_unified_diff, PatchGenerationError
from ..services.patch_workflow import (
    patch_workflow_enabled,
    pr_workflow_enabled,
    pr_workflow_dry_run,
    validate_unified_diff,
    apply_unified_diff_in_sandbox,
    open_pull_request_from_patch_run,
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


def _serialize_chunk_hit(h: Any) -> dict[str, Any]:
    """
    Serialize ChunkHit dataclass (or dict-ish) into a stable JSON object.
    Works whether repo_chunks.search_chunks returns dataclass instances or dicts.
    """
    if isinstance(h, dict):
        return {
            "id": h.get("id"),
            "snapshot_id": h.get("snapshot_id"),
            "path": h.get("path"),
            "start_line": h.get("start_line"),
            "end_line": h.get("end_line"),
            "symbols_json": h.get("symbols_json"),
            "score": h.get("score"),
            "chunk_text": h.get("chunk_text"),
        }

    return {
        "id": getattr(h, "id", None),
        "snapshot_id": getattr(h, "snapshot_id", None),
        "path": getattr(h, "path", None),
        "start_line": getattr(h, "start_line", None),
        "end_line": getattr(h, "end_line", None),
        "symbols_json": getattr(h, "symbols_json", None),
        "score": getattr(h, "score", None),
        "chunk_text": getattr(h, "chunk_text", None),
    }


@router.get("/repo/chunks/search")
async def repo_chunks_search(
    request: Request,
    snapshot_id: int = Query(..., ge=1),
    q: str = Query(..., min_length=2),
    top_k: int = Query(8, ge=1, le=30),
    mode: str = Query("auto"),
    path_contains: Optional[str] = Query(default=None),
    include_text: bool = Query(False),
    db: Session = Depends(get_db),
):
    _guard_debug(request)

    mode_used, hits = await search_chunks(
        db=db,
        snapshot_id=snapshot_id,
        query=q,
        top_k=top_k,
        mode=mode,
        path_contains=path_contains,
    )

    serialized = [_serialize_chunk_hit(h) for h in (hits or [])]

    if not include_text:
        for obj in serialized:
            obj.pop("chunk_text", None)

    return {
        "snapshot_id": snapshot_id,
        "query": q,
        "mode_requested": mode,
        "mode_used": mode_used,
        "top_k": top_k,
        "path_contains": path_contains,
        "count": len(serialized),
        "hits": serialized,
    }


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


# -----------------------
# Level 3: PR workflow (generate diff -> sandbox apply/test -> open PR)
# -----------------------
class PRGenerateBody(BaseModel):
    snapshot_id: int
    finding_id: Optional[int] = None
    objective: Optional[str] = None


class PRWorkflowBody(BaseModel):
    snapshot_id: int
    # You can either provide patch_text, OR ask it to generate from finding/objective.
    patch_text: Optional[str] = None
    finding_id: Optional[int] = None
    objective: Optional[str] = None
    run_tests: bool = True
    pr_title: Optional[str] = None
    pr_body: Optional[str] = None


def _serialize_patch_run(run: Any) -> dict[str, Any]:
    return {
        "run_id": getattr(run, "id", None),
        "snapshot_id": getattr(run, "snapshot_id", None),
        "valid": bool(getattr(run, "valid", False)),
        "validation_error": getattr(run, "validation_error", None),
        "files_changed": int(getattr(run, "files_changed", 0) or 0),
        "lines_changed": int(getattr(run, "lines_changed", 0) or 0),
        "file_paths_json": getattr(run, "file_paths_json", None),
        "applied": bool(getattr(run, "applied", False)),
        "apply_error": getattr(run, "apply_error", None),
        "tests_ran": bool(getattr(run, "tests_ran", False)),
        "tests_ok": bool(getattr(run, "tests_ok", False)),
        "test_output": getattr(run, "test_output", None),
        "sandbox_path": getattr(run, "sandbox_path", None),
        "created_at": getattr(run, "created_at", None).isoformat() if getattr(run, "created_at", None) else None,
    }


@router.post("/repo/pr/generate")
async def repo_pr_generate(
    request: Request,
    body: PRGenerateBody,
    db: Session = Depends(get_db),
):
    """
    Commit 1 capability:
    - Generate patch only (no apply, no PR).
    - Validate patch with the same validator used by apply workflow.
    """
    _guard_debug(request)

    if not settings.LLM_ENABLED:
        raise HTTPException(status_code=400, detail="LLM is disabled (LLM_ENABLED=false).")

    try:
        gen = await generate_unified_diff(
            db=db,
            snapshot_id=int(body.snapshot_id),
            finding_id=int(body.finding_id) if body.finding_id is not None else None,
            objective=body.objective,
        )
    except PatchGenerationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patch generation failed: {e!r}")

    # Validate (uses allowlist + caps)
    v = validate_unified_diff(db, int(body.snapshot_id), str(gen["patch_text"]))

    return {
        "snapshot_id": body.snapshot_id,
        "finding_id": body.finding_id,
        "objective": gen.get("objective"),
        "validation": v,
        "patch_text": gen["patch_text"],
        "snippets_used": gen.get("snippets_used", []),
    }


@router.post("/repo/pr/dry_run")
async def repo_pr_dry_run(
    request: Request,
    body: PRWorkflowBody,
    db: Session = Depends(get_db),
):
    """
    Commit 2 capability:
    - If patch_text missing: generate it.
    - Apply in sandbox (requires ENABLE_PATCH_WORKFLOW=true)
    - Run tests (run_tests=true by default)
    - Does NOT open PR.
    """
    _guard_debug(request)

    if not patch_workflow_enabled():
        raise HTTPException(status_code=400, detail="Patch workflow disabled (ENABLE_PATCH_WORKFLOW=false).")

    patch_text = (body.patch_text or "").strip()
    if not patch_text:
        if not settings.LLM_ENABLED:
            raise HTTPException(status_code=400, detail="LLM is disabled and patch_text not provided.")
        try:
            gen = await generate_unified_diff(
                db=db,
                snapshot_id=int(body.snapshot_id),
                finding_id=int(body.finding_id) if body.finding_id is not None else None,
                objective=body.objective,
            )
            patch_text = str(gen["patch_text"])
        except PatchGenerationError as e:
            raise HTTPException(status_code=400, detail=str(e))

    run = await apply_unified_diff_in_sandbox(
        db=db,
        snapshot_id=int(body.snapshot_id),
        patch_text=patch_text,
        run_tests=bool(body.run_tests),
    )

    return {
        "snapshot_id": body.snapshot_id,
        "mode": "dry_run",
        "patch_text": patch_text,
        "run": _serialize_patch_run(run),
    }


@router.post("/repo/pr/run")
async def repo_pr_run(
    request: Request,
    body: PRWorkflowBody,
    db: Session = Depends(get_db),
):
    """
    Full Level 3:
    - generate patch (if needed)
    - validate/apply/tests
    - open PR (requires ENABLE_PR_WORKFLOW=true and PR_WORKFLOW_DRY_RUN=false)
    """
    _guard_debug(request)

    if not patch_workflow_enabled():
        raise HTTPException(status_code=400, detail="Patch workflow disabled (ENABLE_PATCH_WORKFLOW=false).")

    if not pr_workflow_enabled():
        raise HTTPException(status_code=400, detail="PR workflow disabled (ENABLE_PR_WORKFLOW=false).")

    if pr_workflow_dry_run():
        raise HTTPException(
            status_code=400,
            detail="PR workflow is in DRY RUN mode (PR_WORKFLOW_DRY_RUN=true). Set it to false to actually open PRs.",
        )

    patch_text = (body.patch_text or "").strip()
    if not patch_text:
        if not settings.LLM_ENABLED:
            raise HTTPException(status_code=400, detail="LLM is disabled and patch_text not provided.")
        try:
            gen = await generate_unified_diff(
                db=db,
                snapshot_id=int(body.snapshot_id),
                finding_id=int(body.finding_id) if body.finding_id is not None else None,
                objective=body.objective,
            )
            patch_text = str(gen["patch_text"])
        except PatchGenerationError as e:
            raise HTTPException(status_code=400, detail=str(e))

    run = await apply_unified_diff_in_sandbox(
        db=db,
        snapshot_id=int(body.snapshot_id),
        patch_text=patch_text,
        run_tests=bool(body.run_tests),
    )

    if not getattr(run, "valid", False):
        return {"snapshot_id": body.snapshot_id, "mode": "run", "patch_text": patch_text, "run": _serialize_patch_run(run), "pr": None}

    if not getattr(run, "applied", False):
        return {"snapshot_id": body.snapshot_id, "mode": "run", "patch_text": patch_text, "run": _serialize_patch_run(run), "pr": None}

    if bool(body.run_tests) and not getattr(run, "tests_ok", False):
        return {"snapshot_id": body.snapshot_id, "mode": "run", "patch_text": patch_text, "run": _serialize_patch_run(run), "pr": None}

    title = (body.pr_title or "").strip()
    if not title:
        # reasonable default
        title = "Automated fix from Goal Autopilot"

    pr = await open_pull_request_from_patch_run(
        db=db,
        snapshot_id=int(body.snapshot_id),
        patch_run_id=int(getattr(run, "id")),
        title=title,
        body=(body.pr_body or None),
    )

    return {
        "snapshot_id": body.snapshot_id,
        "mode": "run",
        "patch_text": patch_text,
        "run": _serialize_patch_run(run),
        "pr": {
            "id": getattr(pr, "id", None),
            "pr_number": getattr(pr, "pr_number", None),
            "pr_url": getattr(pr, "pr_url", None),
            "branch": getattr(pr, "branch", None),
            "title": getattr(pr, "title", None),
        },
    }
