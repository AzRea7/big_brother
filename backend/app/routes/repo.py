# backend/app/routes/repo.py
from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..db import get_db
from ..services.github_sync import create_repo_snapshot, latest_snapshot, snapshot_file_stats
from ..services.code_signals import materialize_suggestions_as_tasks
from ..config import settings

router = APIRouter()

@router.get("/debug/repo/signal_counts", response_class=JSONResponse)
def repo_signal_counts(
    db: Session = Depends(get_db),
    snapshot_id: int = Query(..., ge=1),
) -> dict[str, Any]:
    # crude but effective: count TODO/FIXME occurrences in stored content
    row = db.execute(
        text("""
        SELECT
          SUM(CASE WHEN content LIKE '%TODO%'  THEN 1 ELSE 0 END) AS files_with_todo,
          SUM(CASE WHEN content LIKE '%FIXME%' THEN 1 ELSE 0 END) AS files_with_fixme,
          COUNT(*) AS total_files
        FROM repo_files
        WHERE snapshot_id = :sid
          AND content_kind = 'text'
          AND content IS NOT NULL
        """),
        {"sid": snapshot_id},
    ).mappings().first()

    return dict(row) if row else {"total_files": 0}


@router.get("/debug/repo/github_auth", response_class=JSONResponse)
def github_auth_debug():
    token = settings.GITHUB_TOKEN or ""
    return {
        "has_token": bool(token),
        "token_prefix": token[:6] + "..." if token else None,
        "repo": settings.GITHUB_REPO,
        "branch": settings.GITHUB_BRANCH,
    }


@router.get("/debug/repo/db_counts", response_class=JSONResponse)
def repo_db_counts(db: Session = Depends(get_db)):
    # raw SQL so we see if tables exist + have rows
    try:
        snap = db.execute(text("SELECT COUNT(*) FROM repo_snapshots")).scalar_one()
    except Exception as e:
        snap = f"ERR: {e}"

    try:
        files = db.execute(text("SELECT COUNT(*) FROM repo_files")).scalar_one()
    except Exception as e:
        files = f"ERR: {e}"

    return {"repo_snapshots": snap, "repo_files": files}


@router.get("/debug/repo/db_url", response_class=JSONResponse)
def repo_db_url():
    return {"DB_URL": settings.DB_URL}


@router.get("/api/repo/status", response_class=JSONResponse)
def repo_status(
    db: Session = Depends(get_db),
    repo: Optional[str] = Query(default=None),
    branch: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    snap = latest_snapshot(db, repo=repo, branch=branch)
    if not snap:
        return {"has_snapshot": False}

    stats = snapshot_file_stats(db, snapshot_id=snap.id)
    warnings = []
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
            "created_at": snap.created_at.isoformat(),
            "commit_sha": snap.commit_sha,
            "file_count": snap.file_count,
            "stored_content_files": snap.stored_content_files,
            "warnings": warnings,
        },
        "stats": stats,
    }


@router.post("/debug/repo/sync", response_class=JSONResponse)
async def repo_sync(
    db: Session = Depends(get_db),
    repo: Optional[str] = Query(default=None),
    branch: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    result = await create_repo_snapshot(db, repo=repo, branch=branch)
    return {
        "snapshot_id": result.snapshot_id,
        "repo": result.repo,
        "branch": result.branch,
        "file_count": result.file_count,
        "stored_content_files": result.stored_content_files,
        "warnings": result.warnings,
    }


@router.post("/debug/repo/suggest_tasks", response_class=JSONResponse)
def repo_suggest_tasks(
    db: Session = Depends(get_db),
    snapshot_id: int = Query(..., ge=1),
    project: str = Query(default="haven"),
    limit: int = Query(default=30, ge=1, le=200),
) -> dict[str, Any]:
    return materialize_suggestions_as_tasks(db, snapshot_id=snapshot_id, project=project, limit=limit)


@router.get("/ui/repo", response_class=HTMLResponse)
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
  </style>
</head>
<body>
  <div class="row" style="justify-content:space-between; align-items:flex-end;">
    <div>
      <div class="title">Repo Context (Read-only)</div>
      <div class="muted">Sync GitHub → snapshot DB → suggest tasks from code signals.</div>
    </div>
    <div class="muted"><a href="/ui/dashboard" style="text-decoration:none;">Dashboard</a> · <a href="/docs" style="text-decoration:none;">API Docs</a></div>
  </div>

  <div class="row" style="margin: 14px 0;">
    <input id="repo" style="min-width:320px" placeholder="repo (owner/name) e.g. AzRea7/OneHaven" />
    <input id="branch" style="min-width:160px" placeholder="branch e.g. main" />
    <button id="syncBtn">Sync</button>
    <button id="refreshBtn">Refresh Status</button>
  </div>

  <div id="status" class="card"></div>

  <div class="row" style="margin-top:12px;">
    <input id="project" style="min-width:160px" value="haven" />
    <input id="limit" style="min-width:120px" value="30" />
    <button id="suggestBtn" disabled>Suggest Tasks (TODO/FIXME)</button>
  </div>

  <div class="card" style="margin-top:12px;">
    <div class="muted">Output</div>
    <pre id="out">{}</pre>
  </div>

<script>
let latestSnapshotId = null;

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

function renderStatus(data) {
  const el = $("status");
  if (!data.has_snapshot) {
    el.innerHTML = `<div><b>No snapshot yet.</b></div><div class="muted">Click Sync to create one.</div>`;
    $("suggestBtn").disabled = true;
    latestSnapshotId = null;
    return;
  }

  const s = data.snapshot;
  latestSnapshotId = s.id;
  $("suggestBtn").disabled = false;

  const stats = data.stats || {};
  const warnings = (s.warnings || []).slice(0, 8).map(w => `<li>${w}</li>`).join("");
  el.innerHTML = `
    <div><b>Latest snapshot:</b> #${s.id}</div>
    <div class="muted">${s.repo}@${s.branch} · ${s.created_at}</div>
    <div style="margin-top:8px;">
      <div><b>Files:</b> ${stats.total_files || s.file_count} (text: ${stats.text_files || 0}, binary: ${stats.binary_files || 0}, skipped: ${stats.skipped_files || 0})</div>
      <div><b>Stored content:</b> ${s.stored_content_files}</div>
    </div>
    <div style="margin-top:8px;">
      <div class="muted"><b>Top folders</b></div>
      <div class="muted">${(stats.top_folders||[]).map(x => `${x.folder}(${x.count})`).join(" · ")}</div>
    </div>
    <div style="margin-top:8px;">
      <div class="muted"><b>Warnings</b></div>
      <ul class="muted">${warnings || "<li>none</li>"}</ul>
    </div>
  `;
}

async function refresh() {
  const repo = $("repo").value.trim();
  const branch = $("branch").value.trim();
  let url = "/api/repo/status";
  const qs = [];
  if (repo) qs.push("repo="+encodeURIComponent(repo));
  if (branch) qs.push("branch="+encodeURIComponent(branch));
  if (qs.length) url += "?" + qs.join("&");
  const data = await apiGet(url);
  renderStatus(data);
  $("out").textContent = JSON.stringify(data, null, 2);
}

$("refreshBtn").onclick = async () => {
  try { await refresh(); } catch(e) { $("out").textContent = String(e); }
};

$("syncBtn").onclick = async () => {
  try {
    const repo = $("repo").value.trim();
    const branch = $("branch").value.trim();
    let url = "/debug/repo/sync";
    const qs = [];
    if (repo) qs.push("repo="+encodeURIComponent(repo));
    if (branch) qs.push("branch="+encodeURIComponent(branch));
    if (qs.length) url += "?" + qs.join("&");

    const data = await apiPost(url);
    $("out").textContent = JSON.stringify(data, null, 2);
    await refresh();
  } catch(e) {
    $("out").textContent = String(e);
  }
};

$("suggestBtn").onclick = async () => {
  try {
    if (!latestSnapshotId) throw new Error("No snapshot id");
    const project = $("project").value.trim() || "haven";
    const limit = parseInt($("limit").value || "30", 10);

    const url = `/debug/repo/suggest_tasks?snapshot_id=${latestSnapshotId}&project=${encodeURIComponent(project)}&limit=${limit}`;
    const data = await apiPost(url);
    $("out").textContent = JSON.stringify(data, null, 2);
  } catch(e) {
    $("out").textContent = String(e);
  }
};

refresh().catch(()=>{});
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)
