# backend/app/routes/dashboard.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db  # NOTE: if your get_db is named differently, see note below.
from ..models import Goal, Task

router = APIRouter()


def _project_list(db: Session) -> list[str]:
    # distinct projects from tasks + goals
    proj_tasks = db.scalars(select(Task.project).distinct()).all()
    proj_goals = db.scalars(select(Goal.project).distinct()).all()
    projects = sorted({*(p for p in proj_tasks if p), *(p for p in proj_goals if p)})
    return projects or ["onestream", "haven"]


@router.get("/api/dashboard", response_class=JSONResponse)
def dashboard_data(
    db: Session = Depends(get_db),
    project: Optional[str] = Query(default=None, description="Filter to a single project"),
    limit_open: int = Query(default=25, ge=1, le=200),
    limit_done: int = Query(default=25, ge=1, le=200),
) -> dict[str, Any]:
    """
    JSON backing for the dashboard.

    Returns:
      - per_project summary
      - open tasks (optionally filtered)
      - recently completed tasks (optionally filtered)
    """
    projects = _project_list(db)

    # ---- per-project summary counts ----
    per_project: dict[str, dict[str, Any]] = {}
    for p in projects:
        total = db.scalar(select(func.count(Task.id)).where(Task.project == p)) or 0
        done = db.scalar(
            select(func.count(Task.id)).where(Task.project == p).where(Task.completed == True)  # noqa: E712
        ) or 0
        open_ = total - done

        remaining_minutes = (
            db.scalar(
                select(func.coalesce(func.sum(Task.estimated_minutes), 0))
                .where(Task.project == p)
                .where(Task.completed == False)  # noqa: E712
            )
            or 0
        )

        per_project[p] = {
            "total": int(total),
            "open": int(open_),
            "completed": int(done),
            "completion_rate": (float(done) / float(total)) if total else 0.0,
            "remaining_minutes_est": int(remaining_minutes),
        }

    # ---- tasks list queries ----
    task_filter = []
    if project:
        task_filter.append(Task.project == project)

    # open tasks
    open_tasks = db.scalars(
        select(Task)
        .where(*task_filter)
        .where(Task.completed == False)  # noqa: E712
        .order_by(Task.priority.desc(), Task.due_date.asc().nullslast(), Task.created_at.desc())
        .limit(limit_open)
    ).all()

    # recent completed
    done_tasks = db.scalars(
        select(Task)
        .where(*task_filter)
        .where(Task.completed == True)  # noqa: E712
        .order_by(Task.completed_at.desc().nullslast(), Task.created_at.desc())
        .limit(limit_done)
    ).all()

    def task_row(t: Task) -> dict[str, Any]:
        return {
            "id": t.id,
            "project": t.project,
            "goal_id": t.goal_id,
            "title": t.title,
            "priority": t.priority,
            "estimated_minutes": t.estimated_minutes,
            "due_date": t.due_date.isoformat() if t.due_date else None,
            "blocks_me": bool(t.blocks_me),
            "completed": bool(t.completed),
            "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            "parent_task_id": t.parent_task_id,
        }

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "projects": projects,
        "filter_project": project,
        "per_project": per_project,
        "open_tasks": [task_row(t) for t in open_tasks],
        "recent_completed": [task_row(t) for t in done_tasks],
    }


@router.get("/ui/dashboard", response_class=HTMLResponse)
def dashboard_page() -> HTMLResponse:
    """
    Simple dashboard UI (no template engine needed).
    Fetches JSON from /api/dashboard and renders tables client-side.
    """
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Goal Autopilot — Dashboard</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    .card { border: 1px solid #e6e6e6; border-radius: 12px; padding: 14px; min-width: 240px; }
    .title { font-size: 20px; font-weight: 700; margin: 0 0 10px 0; }
    .muted { color: #666; font-size: 13px; }
    .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; border: 1px solid #ddd; }
    .pill.red { border-color: #f2b8b8; background: #fff6f6; }
    .pill.green { border-color: #b8f2c3; background: #f6fff8; }
    .pill.gray { border-color: #ddd; background: #fafafa; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #eee; padding: 10px; text-align: left; font-size: 14px; vertical-align: top; }
    th { color: #444; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .controls { display:flex; gap:10px; align-items:center; margin: 12px 0 18px 0; flex-wrap: wrap; }
    select, button { padding: 8px 10px; border-radius: 10px; border: 1px solid #ddd; background: white; }
    button { cursor: pointer; }
    .progress { height: 10px; width: 100%; background:#f1f1f1; border-radius: 999px; overflow:hidden; }
    .bar { height: 10px; background: #222; width: 0%; }
    .task-title { font-weight: 600; }
    .right { text-align:right; }
  </style>
</head>
<body>
  <div class="row" style="justify-content:space-between; align-items:flex-end;">
    <div>
      <div class="title">Dashboard</div>
      <div class="muted">Track open + completed tasks per project.</div>
    </div>
    <div class="muted" id="generatedAt"></div>
  </div>

  <div class="controls">
    <label class="muted">Project</label>
    <select id="projectSelect"></select>
    <button id="refreshBtn">Refresh</button>
    <a class="muted" href="/docs" target="_blank" style="text-decoration:none;">API Docs</a>
  </div>

  <div id="summaryRow" class="row"></div>

  <h3 style="margin-top:22px;">Open tasks</h3>
  <div class="card" style="padding:0;">
    <table>
      <thead>
        <tr>
          <th>ID</th><th>Project</th><th>Title</th><th>Due</th><th class="right">Est</th><th>Flags</th><th></th>
        </tr>
      </thead>
      <tbody id="openBody"></tbody>
    </table>
  </div>

  <h3 style="margin-top:22px;">Recently completed</h3>
  <div class="card" style="padding:0;">
    <table>
      <thead>
        <tr>
          <th>ID</th><th>Project</th><th>Title</th><th>Completed</th><th class="right">Est</th>
        </tr>
      </thead>
      <tbody id="doneBody"></tbody>
    </table>
  </div>

<script>
async function fetchJSON(project) {
  const url = project && project !== "__all__"
    ? `/api/dashboard?project=${encodeURIComponent(project)}`
    : `/api/dashboard`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Dashboard fetch failed: ${r.status}`);
  return await r.json();
}

function fmtDate(s) {
  if (!s) return "";
  try { return new Date(s).toLocaleString(); } catch { return s; }
}

function pill(text, cls) {
  return `<span class="pill ${cls||"gray"}">${text}</span>`;
}

function renderSummary(data) {
  const row = document.getElementById("summaryRow");
  row.innerHTML = "";

  const per = data.per_project || {};
  const projects = data.projects || [];

  for (const p of projects) {
    const s = per[p] || { total:0, open:0, completed:0, completion_rate:0, remaining_minutes_est:0 };
    const pct = Math.round((s.completion_rate || 0) * 100);
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div class="task-title">${p}</div>
        ${pill(`${pct}%`, pct >= 60 ? "green" : (pct >= 30 ? "gray" : "red"))}
      </div>
      <div class="muted" style="margin:8px 0 8px 0;">Open ${s.open} • Done ${s.completed} • Total ${s.total}</div>
      <div class="progress"><div class="bar" style="width:${pct}%"></div></div>
      <div class="muted" style="margin-top:8px;">Remaining est: ${s.remaining_minutes_est} min</div>
    `;
    row.appendChild(card);
  }
}

function renderTables(data) {
  const openBody = document.getElementById("openBody");
  const doneBody = document.getElementById("doneBody");
  openBody.innerHTML = "";
  doneBody.innerHTML = "";

  for (const t of (data.open_tasks || [])) {
    const flags = [
      t.blocks_me ? pill("blocks_me", "red") : "",
      (t.parent_task_id ? pill("micro/child", "gray") : ""),
    ].filter(Boolean).join(" ");

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${t.id}</td>
      <td>${t.project || ""}</td>
      <td>
        <div class="task-title">${t.title}</div>
        <div class="muted">goal_id=${t.goal_id ?? ""}</div>
      </td>
      <td>${t.due_date || ""}</td>
      <td class="right">${t.estimated_minutes || 0}m</td>
      <td>${flags || ""}</td>
      <td class="right">
        <button onclick="completeTask(${t.id})">Complete</button>
      </td>
    `;
    openBody.appendChild(tr);
  }

  for (const t of (data.recent_completed || [])) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${t.id}</td>
      <td>${t.project || ""}</td>
      <td><div class="task-title">${t.title}</div></td>
      <td>${fmtDate(t.completed_at)}</td>
      <td class="right">${t.estimated_minutes || 0}m</td>
    `;
    doneBody.appendChild(tr);
  }
}

async function completeTask(id) {
  // assumes your API has POST /tasks/{id}/complete
  const r = await fetch(`/tasks/${id}/complete`, { method: "POST" });
  if (!r.ok) {
    const txt = await r.text();
    alert(`Complete failed: ${r.status}\\n${txt}`);
    return;
  }
  await load();
}

async function load() {
  const sel = document.getElementById("projectSelect");
  const project = sel.value;
  const data = await fetchJSON(project);
  document.getElementById("generatedAt").textContent = `Updated: ${fmtDate(data.generated_at)}`;
  renderSummary(data);
  renderTables(data);
}

async function init() {
  const data = await fetchJSON(null);
  const sel = document.getElementById("projectSelect");
  sel.innerHTML = "";
  sel.insertAdjacentHTML("beforeend", `<option value="__all__">All projects</option>`);
  for (const p of (data.projects || [])) {
    sel.insertAdjacentHTML("beforeend", `<option value="${p}">${p}</option>`);
  }
  document.getElementById("refreshBtn").addEventListener("click", load);
  await load();
}

init().catch(e => {
  document.body.innerHTML = `<pre style="color:#b00;">Dashboard failed: ${e}</pre>`;
});
</script>
</body>
</html>
"""
    return HTMLResponse(content=html, status_code=200)
