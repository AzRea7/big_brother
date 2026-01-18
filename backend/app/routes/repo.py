# backend/app/routes/repo.py
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import RepoFile, RepoFinding, RepoSnapshot
from ..services.repo_llm_findings import run_llm_scan, tasks_from_findings
from ..services.repo_taskgen import compute_signal_counts

router = APIRouter(prefix="/debug/repo", tags=["repo"])
status_router = APIRouter(tags=["repo-ui"])


def _snapshot_or_404(db: Session, snapshot_id: int) -> RepoSnapshot:
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        raise HTTPException(status_code=404, detail=f"snapshot_id={snapshot_id} not found")
    return snap


def _get_sync_callable():
    """
    Prefer sync_repo_snapshot if present (compat wrappers),
    otherwise fall back to create_repo_snapshot.
    Both are async in your code.
    """
    from ..services import github_sync as gs

    if hasattr(gs, "sync_repo_snapshot"):
        return getattr(gs, "sync_repo_snapshot")
    if hasattr(gs, "create_repo_snapshot"):
        return getattr(gs, "create_repo_snapshot")

    raise HTTPException(
        status_code=500,
        detail="github_sync does not export sync_repo_snapshot or create_repo_snapshot.",
    )


@router.post("/sync", response_class=JSONResponse)
async def repo_sync(
    repo: str | None = Query(default=None),
    branch: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    repo = repo or settings.GITHUB_REPO
    branch = branch or settings.GITHUB_BRANCH

    sync_fn = _get_sync_callable()
    out = await sync_fn(db=db, repo=repo, branch=branch)

    if isinstance(out, dict):
        return out
    return getattr(out, "__dict__", {"result": str(out)})


@router.post("/scan_llm", response_class=JSONResponse)
async def scan_llm(
    snapshot_id: int = Query(..., ge=1),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _snapshot_or_404(db, snapshot_id)
    return await run_llm_scan(db, snapshot_id=snapshot_id)


@router.get("/findings", response_class=JSONResponse)
def findings(
    snapshot_id: int = Query(..., ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _snapshot_or_404(db, snapshot_id)

    rows = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .order_by(RepoFinding.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    out = []
    for r in rows:
        out.append(
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
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
        )

    total = db.query(RepoFinding).filter(RepoFinding.snapshot_id == snapshot_id).count()
    return {"snapshot_id": snapshot_id, "count": int(total), "findings": out}


@router.post("/tasks_from_findings", response_class=JSONResponse)
def make_tasks_from_findings(
    snapshot_id: int = Query(..., ge=1),
    project: str = Query(default="haven"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _snapshot_or_404(db, snapshot_id)
    return tasks_from_findings(db, snapshot_id=snapshot_id, project=project)


@router.get("/signal_counts_full", response_class=JSONResponse)
def signal_counts_full(
    snapshot_id: int = Query(..., ge=1),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _snapshot_or_404(db, snapshot_id)
    return {"snapshot_id": snapshot_id, "signals": compute_signal_counts(db, snapshot_id)}


@router.get("/search", response_class=JSONResponse)
def search_snapshot(
    snapshot_id: int = Query(..., ge=1),
    q: str = Query(..., min_length=1, max_length=120),
    limit: int = Query(default=25, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _snapshot_or_404(db, snapshot_id)

    needle = q.lower()
    files = db.scalars(select(RepoFile).where(RepoFile.snapshot_id == snapshot_id)).all()

    hits: list[dict[str, Any]] = []
    for rf in files:
        if rf.content_kind != "text":
            continue

        text = (getattr(rf, "content_text", None) or rf.content or "")
        if not text:
            continue

        idx = text.lower().find(needle)
        if idx == -1:
            continue

        start = max(0, idx - 140)
        end = min(len(text), idx + 240)
        snippet = text[start:end].replace("\r\n", "\n")

        hits.append({"path": rf.path, "content_kind": rf.content_kind, "size": rf.size, "snippet": snippet})
        if len(hits) >= limit:
            break

    return {"snapshot_id": snapshot_id, "q": q, "hit_count": len(hits), "hits": hits}


@status_router.get("/api/repo/status", response_class=JSONResponse)
def repo_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    latest = db.scalars(select(RepoSnapshot).order_by(RepoSnapshot.id.desc()).limit(1)).first()
    if not latest:
        return {"latest_snapshot": None}

    total_files = db.scalar(select(func.count(RepoFile.id)).where(RepoFile.snapshot_id == latest.id)) or 0
    stored_text = db.scalar(
        select(func.count(RepoFile.id))
        .where(RepoFile.snapshot_id == latest.id)
        .where(RepoFile.content_kind == "text")
    ) or 0

    return {
        "latest_snapshot": {
            "id": latest.id,
            "repo": latest.repo,
            "branch": latest.branch,
            "commit_sha": latest.commit_sha,
            "file_count": latest.file_count,
            "stored_content_files": latest.stored_content_files,
        },
        "counts": {"total_files": int(total_files), "stored_text_files": int(stored_text)},
    }


@status_router.get("/ui/repo", response_class=HTMLResponse)
def repo_ui() -> HTMLResponse:
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Repo Status</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
    .card { border: 1px solid #e6e6e6; border-radius: 12px; padding: 14px; max-width: 980px; }
    pre { background: #fafafa; padding: 12px; border-radius: 12px; overflow:auto; }
    input, button { padding: 8px 10px; border-radius: 10px; border: 1px solid #ddd; background: white; }
    button { cursor:pointer; }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    .muted { color:#666; font-size:13px; }
  </style>
</head>
<body>
  <h2>Repo status</h2>
  <div class="card">
    <div class="row">
      <button onclick="loadStatus()">Refresh status</button>
      <span class="muted">Try: /debug/repo/search?snapshot_id=...&q=TODO</span>
    </div>
    <pre id="status">Loading...</pre>

    <h3>Search snapshot content</h3>
    <div class="row">
      <input id="snapshotId" placeholder="snapshot_id" />
      <input id="query" placeholder="TODO / FIXME / rate_limit / etc" />
      <button onclick="runSearch()">Search</button>
    </div>
    <pre id="search"></pre>
  </div>

<script>
async function loadStatus() {
  const r = await fetch('/api/repo/status');
  const j = await r.json();
  document.getElementById('status').textContent = JSON.stringify(j, null, 2);
  if (j.latest_snapshot && j.latest_snapshot.id) {
    document.getElementById('snapshotId').value = j.latest_snapshot.id;
  }
}
async function runSearch() {
  const sid = document.getElementById('snapshotId').value;
  const q = document.getElementById('query').value;
  const r = await fetch(`/debug/repo/search?snapshot_id=${encodeURIComponent(sid)}&q=${encodeURIComponent(q)}&limit=25`);
  const j = await r.json();
  document.getElementById('search').textContent = JSON.stringify(j, null, 2);
}
loadStatus();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)
