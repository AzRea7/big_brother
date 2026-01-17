# backend/app/routes/repo.py
from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import RepoSnapshot
from ..schemas import RepoSignalCountsOut, RepoSyncOut, RepoTaskGenOut
from ..services.github_sync import (
    create_repo_snapshot,
    latest_snapshot as latest_repo_snapshot,
    snapshot_file_stats,
)
from ..services.repo_taskgen import compute_signal_counts, generate_tasks_from_snapshot

# --------------------------
# Debug router (repo sync + task gen)
# --------------------------
router = APIRouter(prefix="/debug/repo", tags=["repo"])


@router.post("/sync", response_model=RepoSyncOut)
async def sync_repo(
    repo: str | None = Query(default=None),
    branch: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    res = await create_repo_snapshot(db=db, repo=repo, branch=branch)
    return {
        "snapshot_id": res.snapshot_id,
        "repo": res.repo,
        "branch": res.branch,
        "commit_sha": res.commit_sha,
        "file_count": res.file_count,
        "stored_content_files": res.stored_content_files,
        "warnings": res.warnings,
    }


@router.get("/latest_snapshot")
def latest_snapshot_debug(db: Session = Depends(get_db)):
    snap = db.execute(select(RepoSnapshot).order_by(RepoSnapshot.id.desc())).scalars().first()
    if not snap:
        return {"snapshot_id": None}
    return {"snapshot_id": snap.id, "repo": snap.repo, "branch": snap.branch, "commit_sha": snap.commit_sha}


@router.get("/signal_counts", response_model=RepoSignalCountsOut)
def signal_counts(snapshot_id: int, db: Session = Depends(get_db)):
    """
    Backwards-compatible signal endpoint: only returns fields your schema expects.
    """
    s = compute_signal_counts(db, snapshot_id)
    return {
        "snapshot_id": snapshot_id,
        "total_files": s["total_files"],
        "files_with_todo": s["files_with_todo"],
        "files_with_fixme": s["files_with_fixme"],
        "files_with_impl_signals": s["files_with_impl_signals"],
    }


@router.get("/signal_counts_full", response_class=JSONResponse)
def signal_counts_full(snapshot_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    Full production-grade signal breakdown.

    This endpoint avoids breaking RepoSignalCountsOut while letting the UI (and you)
    see what the snapshot actually contains: auth/timeout/retry/rate-limit/validation/etc.
    """
    s = compute_signal_counts(db, snapshot_id)
    return {"snapshot_id": snapshot_id, "signals": s}


@router.post("/generate_tasks", response_model=RepoTaskGenOut)
async def generate_tasks(
    snapshot_id: int,
    project: str = Query(default="haven"),
    db: Session = Depends(get_db),
):
    created, skipped = await generate_tasks_from_snapshot(db=db, snapshot_id=snapshot_id, project=project)
    return {"snapshot_id": snapshot_id, "created_tasks": created, "skipped_duplicates": skipped}


@router.post("/sync_and_generate", response_model=RepoTaskGenOut)
async def sync_and_generate(
    project: str = Query(default="haven"),
    repo: str | None = Query(default=None),
    branch: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    res = await create_repo_snapshot(db=db, repo=repo, branch=branch)
    created, skipped = await generate_tasks_from_snapshot(db=db, snapshot_id=res.snapshot_id, project=project)
    return {"snapshot_id": res.snapshot_id, "created_tasks": created, "skipped_duplicates": skipped}


# --------------------------
# Status endpoint used by the Repo UI (/ui/repo)
# --------------------------
status_router = APIRouter(tags=["repo"])


@status_router.get("/api/repo/status", response_class=JSONResponse)
def repo_status(
    db: Session = Depends(get_db),
    repo: Optional[str] = Query(default=None),
    branch: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    snap = latest_repo_snapshot(db, repo=repo, branch=branch)
    if not snap:
        return {"has_snapshot": False}

    stats = snapshot_file_stats(db, snapshot_id=snap.id)
    warnings: list[str] = []
    if snap.warnings_json:
        try:
            warnings = json.loads(snap.warnings_json)
        except Exception:
            warnings = ["(failed to parse warnings_json)"]

    return {
        "has_snapshot": True,
        "snapshot": {
            "id": snap.id,
            "repo": snap.repo,
            "branch": snap.branch,
            "commit_sha": snap.commit_sha,
            "file_count": snap.file_count,
            "stored_content_files": snap.stored_content_files,
            "created_at": str(snap.created_at),
            "warnings": warnings,
        },
        "stats": stats,
    }


@status_router.get("/ui/repo", response_class=HTMLResponse)
def repo_page() -> HTMLResponse:
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Goal Autopilot — Repo</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
    .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    .card { border: 1px solid #e6e6e6; border-radius: 12px; padding: 14px; min-width: 280px; }
    .title { font-size: 20px; font-weight: 800; margin: 0 0 10px 0; }
    .muted { color:#666; font-size: 13px; }
    button, input { padding: 8px 10px; border-radius: 10px; border: 1px solid #ddd; background: white; }
    button { cursor:pointer; }
    pre { white-space: pre-wrap; word-break: break-word; background:#fafafa; padding:12px; border-radius:12px; border:1px solid #eee; }
    a { color: inherit; }
    .pill { display:inline-block; padding:4px 10px; border:1px solid #eee; border-radius:999px; margin:4px 6px 0 0; font-size:12px; background:#fafafa; }
  </style>
</head>
<body>
  <div class="row" style="justify-content:space-between; align-items:flex-end;">
    <div>
      <div class="title">Repo → Tasks</div>
      <div class="muted">Sync repo snapshot, generate tasks, and view them in the dashboard.</div>
    </div>
    <div class="muted"><a href="/ui/dashboard" style="text-decoration:none;">Dashboard</a> · <a href="/docs" style="text-decoration:none;">API Docs</a></div>
  </div>

  <div class="row" style="margin: 14px 0;">
    <input id="repo" style="min-width:320px" placeholder="repo (owner/name) e.g. AzRea7/OneHaven" />
    <input id="branch" style="min-width:160px" placeholder="branch e.g. main" />
    <input id="project" style="min-width:160px" value="haven" />
    <button id="syncBtn">Sync</button>
    <button id="syncGenBtn">Sync + Generate</button>
    <button id="refreshBtn">Refresh Status</button>
  </div>

  <div id="status" class="card"></div>

  <div class="card" style="margin-top:12px;">
    <div class="muted">Signals (production-oriented)</div>
    <div id="signals"></div>
  </div>

  <div class="card" style="margin-top:12px;">
    <div class="muted">Output</div>
    <pre id="out">{}</pre>
  </div>

<script>
function $(id){ return document.getElementById(id); }

async function apiGet(url) {
  const r = await fetch(url);
  const j = await r.json();
  if (!r.ok) throw new Error(JSON.stringify(j));
  return j;
}
async function apiPost(url) {
  const r = await fetch(url, { method: "POST" });
  const j = await r.json();
  if (!r.ok) throw new Error(JSON.stringify(j));
  return j;
}

function renderSignals(signals) {
  const keys = [
    "files_with_auth","files_with_timeout","files_with_retry","files_with_rate_limit",
    "files_with_validation","files_with_logging","files_with_metrics","files_with_db",
    "files_with_tests","files_with_ci","files_with_docker","files_with_config",
    "files_with_secrets","files_with_nplus1","files_with_cors","files_with_csrf",
    "files_with_todo","files_with_fixme","files_with_impl_signals"
  ];
  const el = $("signals");
  if (!signals) { el.innerHTML = `<span class="muted">No signals loaded.</span>`; return; }
  el.innerHTML = keys.map(k => `<span class="pill">${k}: <b>${signals[k] ?? 0}</b></span>`).join("");
}

async function refreshStatus() {
  const repo = $("repo").value.trim();
  const branch = $("branch").value.trim();
  const url = repo || branch
    ? `/api/repo/status?repo=${encodeURIComponent(repo)}&branch=${encodeURIComponent(branch || "main")}`
    : `/api/repo/status`;
  const data = await apiGet(url);

  if (!data.has_snapshot) {
    $("status").innerHTML = `<b>No snapshot yet.</b><div class="muted">Click Sync.</div>`;
    renderSignals(null);
    return;
  }

  const s = data.snapshot;
  const stats = data.stats || {};
  $("status").innerHTML = `
    <div><b>Snapshot #${s.id}</b> — ${s.repo}@${s.branch}</div>
    <div class="muted">files=${s.file_count} stored_text=${s.stored_content_files}</div>
    <div class="muted">top_folders=${(stats.top_folders || []).map(x => x.folder+":"+x.count).join(", ")}</div>
  `;

  // Load full signals
  try {
    const sig = await apiGet(`/debug/repo/signal_counts_full?snapshot_id=${encodeURIComponent(s.id)}`);
    renderSignals(sig.signals || {});
  } catch(e) {
    renderSignals(null);
  }
}

$("syncBtn").onclick = async () => {
  const repo = $("repo").value.trim();
  const branch = $("branch").value.trim();
  const url = `/debug/repo/sync?repo=${encodeURIComponent(repo)}&branch=${encodeURIComponent(branch || "main")}`;
  const out = await apiPost(url);
  $("out").textContent = JSON.stringify(out, null, 2);
  await refreshStatus();
};

$("syncGenBtn").onclick = async () => {
  const repo = $("repo").value.trim();
  const branch = $("branch").value.trim();
  const project = $("project").value.trim() || "haven";
  const url = `/debug/repo/sync_and_generate?project=${encodeURIComponent(project)}&repo=${encodeURIComponent(repo)}&branch=${encodeURIComponent(branch || "main")}`;
  const out = await apiPost(url);
  $("out").textContent = JSON.stringify(out, null, 2);
  await refreshStatus();
};

$("refreshBtn").onclick = refreshStatus;

refreshStatus().catch(e => {
  document.body.innerHTML = `<pre style="color:#b00;">Repo UI failed: ${e}</pre>`;
});
</script>
</body>
</html>
"""
    return HTMLResponse(content=html, status_code=200)
