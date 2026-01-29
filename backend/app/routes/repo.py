from __future__ import annotations

from collections import defaultdict
from typing import Any
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import RepoFile, RepoFinding
from ..services.code_signals import compute_signal_counts_full
from ..services.github_sync import sync_repo_to_snapshot
from ..services.metrics import JOBS_TOTAL
from ..services.ops_gaps import run_ops_gap_scan
from ..services.security import debug_is_allowed, require_api_key
from ..services.static_analysis import run_static_analysis_all

# Task conversion (findings -> tasks). This wrapper now prefers "finding-driven retrieval" if available.
from ..services.repo_taskgen import tasks_from_findings

# NOTE:
# Your repo_llm_findings file currently exposes async run_llm_scan(db, snapshot_id)
# but some older routes import run_llm_repo_scan. We'll support both:
try:
    from ..services.repo_llm_findings import run_llm_repo_scan  # type: ignore
except Exception:  # pragma: no cover
    run_llm_repo_scan = None  # type: ignore

try:
    from ..services.repo_llm_findings import run_llm_scan  # type: ignore
except Exception:  # pragma: no cover
    run_llm_scan = None  # type: ignore

# --- Level 2 RAG (chunks + retrieval) ---
try:
    from ..services.repo_chunks import (  # type: ignore
        chunk_snapshot as build_chunks_for_snapshot,
        build_embeddings_for_snapshot,
        search_chunks,
        load_chunk_text,
    )
except Exception:  # pragma: no cover
    build_chunks_for_snapshot = None  # type: ignore
    build_embeddings_for_snapshot = None  # type: ignore
    search_chunks = None  # type: ignore
    load_chunk_text = None  # type: ignore

# --- Level 3 PR workflow (optional, gated) ---
# If you haven't added services/patch_workflow.py yet, these endpoints will return 501-ish errors.
try:
    from ..services.patch_workflow import (  # type: ignore
        validate_unified_diff,
        apply_unified_diff_in_sandbox,
        open_pull_request_from_patch_run,
    )
except Exception:  # pragma: no cover
    validate_unified_diff = None  # type: ignore
    apply_unified_diff_in_sandbox = None  # type: ignore
    open_pull_request_from_patch_run = None  # type: ignore

router = APIRouter(prefix="/debug/repo", tags=["repo-debug"])


def _guard_debug(request: Request) -> None:
    if not debug_is_allowed():
        raise HTTPException(status_code=404, detail="Not found")
    require_api_key(request.headers.get("X-API-Key"))


@router.post("/sync")
def debug_repo_sync(
    request: Request,
    repo: str = Query("AzRea7/OneHaven"),
    branch: str = Query("main"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Sync repo content into RepoSnapshot/RepoFile rows.

    github_sync.sync_repo_to_snapshot returns a dict DTO:
      {snapshot_id, repo, branch, commit_sha, file_count, stored_content_files, warnings}
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
            "acceptance": getattr(r, "acceptance", None),
            "fingerprint": r.fingerprint,
            "is_resolved": getattr(r, "is_resolved", False),
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

    Implementation detail:
      - The wrapper in services/repo_taskgen.py prefers "finding-driven retrieval" when chunks exist,
        and falls back to deterministic conversion if anything is missing.
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
    Marker signals (TODO/FIXME/etc) + per-file summary.
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


# -------------------------------------------------------------------
# ✅ Level 2 RAG endpoints: chunking + retrieval
# -------------------------------------------------------------------

@router.post("/chunks/build")
def debug_repo_chunks_build(
    request: Request,
    snapshot_id: int = Query(...),
    max_lines: int = Query(220, ge=40, le=800),
    overlap: int = Query(40, ge=0, le=200),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Build RepoChunk rows from RepoFile text content for a snapshot.

    Run this once after /sync, then retrieval + finding-focused taskgen becomes powerful.
    """
    _guard_debug(request)

    if build_chunks_for_snapshot is None:
        raise HTTPException(status_code=500, detail="Chunk builder not available (services.repo_chunks missing).")

    JOBS_TOTAL.labels(job="repo_chunks_build", status="start").inc()
    try:
        out = build_chunks_for_snapshot(db=db, snapshot_id=snapshot_id, max_lines=max_lines, overlap=overlap)
        JOBS_TOTAL.labels(job="repo_chunks_build", status="ok").inc()
        return out
    except Exception:
        JOBS_TOTAL.labels(job="repo_chunks_build", status="error").inc()
        raise


@router.post("/chunks/embed")
async def debug_repo_chunks_embed(
    request: Request,
    snapshot_id: int = Query(..., ge=1),
    force: bool = Query(False),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _guard_debug(request)
    if build_embeddings_for_snapshot is None:
        raise HTTPException(status_code=501, detail="Embeddings not available (missing services.repo_chunks).")

    JOBS_TOTAL.labels(job="repo_chunks_embed", status="start").inc()
    try:
        out = await build_embeddings_for_snapshot(db, snapshot_id, force=force)
        JOBS_TOTAL.labels(job="repo_chunks_embed", status="ok").inc()
        return out
    except Exception:
        JOBS_TOTAL.labels(job="repo_chunks_embed", status="error").inc()
        raise


@router.get("/chunks/search")
async def debug_repo_chunks_search(
    request: Request,
    snapshot_id: int = Query(..., ge=1),
    q: str = Query(..., min_length=2),
    top_k: int = Query(16, ge=1, le=100),
    mode: str = Query("auto", max_length=20),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Search chunks using the retrieval provider interface.

    IMPORTANT:
      services.repo_chunks.search_chunks returns: (mode_used, hits[list[ChunkHit]])
      We return hits as dicts to keep JSON stable.
    """
    _guard_debug(request)
    if search_chunks is None:
        raise HTTPException(status_code=501, detail="Chunk search not available")

    JOBS_TOTAL.labels(job="repo_chunks_search", status="start").inc()
    try:
        mode_used, hits = await search_chunks(
            db,
            snapshot_id=snapshot_id,
            query=q,
            top_k=int(top_k),
            mode=str(mode or "auto"),
        )
        JOBS_TOTAL.labels(job="repo_chunks_search", status="ok").inc()
        return {
            "snapshot_id": snapshot_id,
            "query": q,
            "mode_used": mode_used,
            "hits": [h.__dict__ for h in hits],
        }
    except Exception:
        JOBS_TOTAL.labels(job="repo_chunks_search", status="error").inc()
        raise


@router.get("/chunks/get")
def debug_repo_chunk_get(
    request: Request,
    chunk_id: int = Query(..., ge=1),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Load a single chunk body for UI/debug.
    """
    _guard_debug(request)
    if load_chunk_text is None:
        raise HTTPException(status_code=501, detail="Chunk loader not available")
    return load_chunk_text(db=db, chunk_id=chunk_id)


# -------------------------------------------------------------------
# ✅ Level 3 PR workflow endpoints (optional, gated)
# -------------------------------------------------------------------

@router.post("/patch/validate")
def debug_repo_patch_validate(
    request: Request,
    snapshot_id: int = Query(...),
    db: Session = Depends(get_db),
    patch_text: str = "",
) -> dict[str, Any]:
    """
    Validate unified diff patch against safety rules.

    NOTE: This is intentionally gated behind debug + API key + (optional) service.
    """
    _guard_debug(request)

    if validate_unified_diff is None:
        raise HTTPException(status_code=501, detail="Patch workflow not wired yet (services/patch_workflow.py missing).")

    JOBS_TOTAL.labels(job="repo_patch_validate", status="start").inc()
    try:
        out = validate_unified_diff(db=db, snapshot_id=snapshot_id, patch_text=patch_text)  # type: ignore[misc]
        JOBS_TOTAL.labels(job="repo_patch_validate", status="ok").inc()
        return out
    except Exception:
        JOBS_TOTAL.labels(job="repo_patch_validate", status="error").inc()
        raise


@router.post("/patch/apply")
async def debug_repo_patch_apply(
    request: Request,
    snapshot_id: int = Query(...),
    run_tests: bool = Query(True),
    db: Session = Depends(get_db),
    patch_text: str = "",
) -> dict[str, Any]:
    """
    Apply patch in sandbox + optionally run tests.

    IMPORTANT:
      services.patch_workflow.apply_unified_diff_in_sandbox is async → route must await.
    """
    _guard_debug(request)

    if apply_unified_diff_in_sandbox is None:
        raise HTTPException(status_code=501, detail="Patch workflow not wired yet (services/patch_workflow.py missing).")

    JOBS_TOTAL.labels(job="repo_patch_apply", status="start").inc()
    try:
        out = await apply_unified_diff_in_sandbox(
            db=db,
            snapshot_id=snapshot_id,
            patch_text=patch_text,
            run_tests=run_tests,
        )  # type: ignore[misc]
        JOBS_TOTAL.labels(job="repo_patch_apply", status="ok").inc()
        return out
    except Exception:
        JOBS_TOTAL.labels(job="repo_patch_apply", status="error").inc()
        raise


@router.post("/pr/create")
def debug_repo_pr_create(
    request: Request,
    snapshot_id: int = Query(...),
    patch_run_id: int = Query(...),
    title: str = Query(..., min_length=4, max_length=300),
    body: str = Query("", max_length=20_000),
    base_branch: str = Query("main", max_length=120),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Create a PR from a validated/applied patch run.

    Requires services/patch_workflow.py (and GitHub credentials in settings).

    HARD GATE is enforced in the service: valid + applied + tests_ran + tests_ok.
    """
    _guard_debug(request)

    if open_pull_request_from_patch_run is None:
        raise HTTPException(status_code=501, detail="PR workflow not wired yet (services/patch_workflow.py missing).")

    JOBS_TOTAL.labels(job="repo_pr_create", status="start").inc()
    try:
        out = open_pull_request_from_patch_run(
            db=db,
            snapshot_id=snapshot_id,
            patch_run_id=patch_run_id,
            title=title,
            body=body,
            base_branch=base_branch,
        )  # type: ignore[misc]
        JOBS_TOTAL.labels(job="repo_pr_create", status="ok").inc()
        return out
    except Exception:
        JOBS_TOTAL.labels(job="repo_pr_create", status="error").inc()
        raise

# -------------------------------------------------------------------
# ✅ Level 3 PR workflow convenience endpoints (dry_run + run)
# -------------------------------------------------------------------

class PRDryRunRequest(BaseModel):
    snapshot_id: int
    patch_text: str
    run_tests: bool = True


class PRRunRequest(BaseModel):
    snapshot_id: int
    patch_text: str
    title: str
    body: str | None = None
    base_branch: str = "main"
    run_tests: bool = True


@router.post("/pr/dry_run")
async def debug_repo_pr_dry_run(
    request: Request,
    payload: PRDryRunRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Dry-run the Level 3 pipeline:
      1) validate unified diff
      2) apply in sandbox
      3) optionally run tests
    Returns validate + apply output. Never opens a PR.
    """
    _guard_debug(request)

    if validate_unified_diff is None or apply_unified_diff_in_sandbox is None:
        raise HTTPException(status_code=501, detail="Patch workflow not wired yet (services/patch_workflow.py missing).")

    # 1) validate
    JOBS_TOTAL.labels(job="repo_pr_dry_run_validate", status="start").inc()
    try:
        v = validate_unified_diff(db=db, snapshot_id=payload.snapshot_id, patch_text=payload.patch_text)  # type: ignore[misc]
        JOBS_TOTAL.labels(job="repo_pr_dry_run_validate", status="ok").inc()
    except Exception:
        JOBS_TOTAL.labels(job="repo_pr_dry_run_validate", status="error").inc()
        raise

    # If your validator returns {"ok": False, ...}, hard stop
    if isinstance(v, dict) and v.get("ok") is False:
        return {"ok": False, "stage": "validate", "validate": v, "apply": None}

    # 2) apply (+ optional tests)
    JOBS_TOTAL.labels(job="repo_pr_dry_run_apply", status="start").inc()
    try:
        a = await apply_unified_diff_in_sandbox(
            db=db,
            snapshot_id=payload.snapshot_id,
            patch_text=payload.patch_text,
            run_tests=bool(payload.run_tests),
        )  # type: ignore[misc]
        JOBS_TOTAL.labels(job="repo_pr_dry_run_apply", status="ok").inc()
    except Exception:
        JOBS_TOTAL.labels(job="repo_pr_dry_run_apply", status="error").inc()
        raise

    return {"ok": True, "validate": v, "apply": a}


@router.post("/pr/run")
async def debug_repo_pr_run(
    request: Request,
    payload: PRRunRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Full Level 3 pipeline:
      1) validate unified diff
      2) apply in sandbox
      3) run tests (recommended: keep run_tests=True)
      4) open PR (requires ENABLE_PR_WORKFLOW=true and PR_WORKFLOW_DRY_RUN=false)
    """
    _guard_debug(request)

    if validate_unified_diff is None or apply_unified_diff_in_sandbox is None or open_pull_request_from_patch_run is None:
        raise HTTPException(status_code=501, detail="PR workflow not wired yet (services/patch_workflow.py missing).")

    # 1) validate
    JOBS_TOTAL.labels(job="repo_pr_run_validate", status="start").inc()
    try:
        v = validate_unified_diff(db=db, snapshot_id=payload.snapshot_id, patch_text=payload.patch_text)  # type: ignore[misc]
        JOBS_TOTAL.labels(job="repo_pr_run_validate", status="ok").inc()
    except Exception:
        JOBS_TOTAL.labels(job="repo_pr_run_validate", status="error").inc()
        raise

    if isinstance(v, dict) and v.get("ok") is False:
        return {"ok": False, "stage": "validate", "validate": v, "apply": None, "pr": None}

    # 2) apply (+ tests)
    JOBS_TOTAL.labels(job="repo_pr_run_apply", status="start").inc()
    try:
        a = await apply_unified_diff_in_sandbox(
            db=db,
            snapshot_id=payload.snapshot_id,
            patch_text=payload.patch_text,
            run_tests=bool(payload.run_tests),
        )  # type: ignore[misc]
        JOBS_TOTAL.labels(job="repo_pr_run_apply", status="ok").inc()
    except Exception:
        JOBS_TOTAL.labels(job="repo_pr_run_apply", status="error").inc()
        raise

    # pull patch_run_id out of apply response (supports a few common shapes)
    patch_run_id = None
    if isinstance(a, dict):
        patch_run_id = a.get("patch_run_id") or a.get("run_id") or a.get("id")
    if patch_run_id is None:
        raise HTTPException(status_code=500, detail=f"apply_unified_diff_in_sandbox did not return patch_run_id. Got: {a}")

    # 3) open PR (service itself enforces hard gates)
    JOBS_TOTAL.labels(job="repo_pr_run_open_pr", status="start").inc()
    try:
        pr = open_pull_request_from_patch_run(
            db=db,
            snapshot_id=payload.snapshot_id,
            patch_run_id=int(patch_run_id),
            title=payload.title,
            body=payload.body,
            base_branch=payload.base_branch,
        )  # type: ignore[misc]
        JOBS_TOTAL.labels(job="repo_pr_run_open_pr", status="ok").inc()
    except Exception:
        JOBS_TOTAL.labels(job="repo_pr_run_open_pr", status="error").inc()
        raise

    return {"ok": True, "validate": v, "apply": a, "pr": pr}
