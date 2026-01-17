# backend/app/routes/repo.py
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import RepoFile, RepoSnapshot
from ..services.repo_taskgen import compute_signal_counts, generate_tasks_from_snapshot
from ..services.repo_findings import run_llm_scan, list_findings, tasks_from_findings

# ---- github_sync import: support multiple names defensively ----
try:
    from ..services.github_sync import sync_repo_snapshot  # preferred
except Exception:
    sync_repo_snapshot = None  # type: ignore

try:
    from ..services.github_sync import sync_snapshot  # fallback
except Exception:
    sync_snapshot = None  # type: ignore

try:
    from ..services.github_sync import sync_repo  # fallback
except Exception:
    sync_repo = None  # type: ignore


router = APIRouter(prefix="/debug/repo", tags=["repo"])
status_router = APIRouter(tags=["repo-ui"])


def _snapshot_or_404(db: Session, snapshot_id: int) -> RepoSnapshot:
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        raise HTTPException(status_code=404, detail=f"snapshot_id={snapshot_id} not found")
    return snap


def _sync(db: Session, repo: str, branch: str) -> dict[str, Any]:
    """
    Some earlier versions exported different function names.
    Try a few in order.
    """
    if sync_repo_snapshot:
        return sync_repo_snapshot(db=db, repo=repo, branch=branch)  # type: ignore[misc]
    if sync_snapshot:
        return sync_snapshot(db=db, repo=repo, branch=branch)  # type: ignore[misc]
    if sync_repo:
        return sync_repo(db=db, repo=repo, branch=branch)  # type: ignore[misc]
    raise HTTPException(
        status_code=500,
        detail=(
            "github_sync does not export a supported sync function. Tried: "
            "sync_repo_snapshot, sync_snapshot, sync_repo."
        ),
    )


@router.post("/sync", response_class=JSONResponse)
def repo_sync(
    repo: str | None = Query(default=None),
    branch: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    repo = repo or settings.GITHUB_REPO
    branch = branch or settings.GITHUB_BRANCH
    return _sync(db=db, repo=repo, branch=branch)


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

        hits.append(
            {
                "path": rf.path,
                "content_kind": rf.content_kind,
                "size": rf.size,
                "snippet": snippet,
            }
        )
        if len(hits) >= limit:
            break

    return {"snapshot_id": snapshot_id, "q": q, "hit_count": len(hits), "hits": hits}


@router.post("/tasks_from_snapshot", response_class=JSONResponse)
async def tasks_from_snapshot(
    snapshot_id: int = Query(..., ge=1),
    project: str = Query(default="haven"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _snapshot_or_404(db, snapshot_id)
    created, skipped = await generate_tasks_from_snapshot(db=db, snapshot_id=snapshot_id, project=project)
    return {"snapshot_id": snapshot_id, "project": project, "created": created, "skipped": skipped}


# --------------------------
# NEW: LLM scan -> findings
# --------------------------

@router.post("/scan_llm", response_class=JSONResponse)
async def scan_llm(
    snapshot_id: int = Query(..., ge=1),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Run a bounded LLM scan and store RepoFinding rows.
    """
    _snapshot_or_404(db, snapshot_id)
    out = await run_llm_scan(db=db, snapshot_id=snapshot_id)
    return out


@router.get("/findings", response_class=JSONResponse)
def findings(
    snapshot_id: int = Query(..., ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _snapshot_or_404(db, snapshot_id)
    rows = list_findings(db=db, snapshot_id=snapshot_id, limit=limit)

    return {
        "snapshot_id": snapshot_id,
        "count": len(rows),
        "findings": [
            {
                "id": r.id,
                "path": r.path,
                "line": r.line,
                "category": r.category,
                "severity": r.severity,
                "title": r.title,
                "evidence": r.evidence,
                "recommendation": r.recommendation,
                "fingerprint": r.fingerprint,
                "created_at": str(r.created_at) if getattr(r, "created_at", None) else None,
            }
            for r in rows
        ],
    }


@router.post("/tasks_from_findings", response_class=JSONResponse)
def tasks_from_findings_route(
    snapshot_id: int = Query(..., ge=1),
    project: str = Query(default="haven"),
    limit: int = Query(default=12, ge=1, le=50),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _snapshot_or_404(db, snapshot_id)
    out = tasks_from_findings(db=db, snapshot_id=snapshot_id, project=project, limit=limit)
    return out


# --------------------------
# UI helpers
# --------------------------

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
      <span class="muted">Also try: /debug/repo/search?snapshot_id=...&q=TODO</span>
    </div>
    <pre id="status">Loading...</pre>

    <h3>Search snapshot content</h3>
    <div class="row">
      <input id="snapshotId" placeholder="snapshot_id" />
      <input id="query" placeholder="TODO / FIXME / rate_limit / etc" />
      <button onclick="runSearch()">Search</button>
    </div>
    <pre id="search"></pre>

    <h3>LLM scan</h3>
    <div class="row">
      <button onclick="runScan()">Run /scan_llm</button>
      <button onclick="loadFindings()">Load /findings</button>
      <button onclick="makeTasks()">Make tasks from findings</button>
    </div>
    <pre id="findings"></pre>
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
async function runScan() {
  const sid = document.getElementById('snapshotId').value;
  const r = await fetch(`/debug/repo/scan_llm?snapshot_id=${encodeURIComponent(sid)}`, { method: "POST" });
  const j = await r.json();
  document.getElementById('findings').textContent = JSON.stringify(j, null, 2);
}
async function loadFindings() {
  const sid = document.getElementById('snapshotId').value;
  const r = await fetch(`/debug/repo/findings?snapshot_id=${encodeURIComponent(sid)}&limit=50`);
  const j = await r.json();
  document.getElementById('findings').textContent = JSON.stringify(j, null, 2);
}
async function makeTasks() {
  const sid = document.getElementById('snapshotId').value;
  const r = await fetch(`/debug/repo/tasks_from_findings?snapshot_id=${encodeURIComponent(sid)}&project=haven&limit=12`, { method: "POST" });
  const j = await r.json();
  document.getElementById('findings').textContent = JSON.stringify(j, null, 2);
}
loadStatus();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)
