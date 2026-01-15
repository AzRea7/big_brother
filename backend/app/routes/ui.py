# backend/app/routes/ui.py
from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, Depends, Form, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Task
from ..services.planner import generate_daily_plan

router = APIRouter(tags=["ui"])


@router.get("/ui/today", response_class=HTMLResponse)
async def ui_today(
    project: str = Query(default="onestream"),
    db: Session = Depends(get_db),
):
    # Pull tasks for lane
    tasks = list(
        db.scalars(
            select(Task)
            .where(Task.project == project)
            .where(Task.completed == False)  # noqa: E712
            .order_by(Task.priority.desc(), Task.created_at.desc())
        ).all()
    )

    # Generate plan live when you open the page
    plan = await generate_daily_plan(db=db, focus_project=project, mode="single")

    rows_html = ""
    for t in tasks[:80]:
        starter = (t.starter or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        dod = (t.dod or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        notes = (t.notes or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        rows_html += f"""
        <div class="task">
          <div class="taskTop">
            <div class="title"><b>[{t.id}]</b> {t.title}</div>
            <div class="meta">p={t.priority} • est={t.estimated_minutes}m • due={t.due_date or "—"} • blocks_me={str(t.blocks_me).lower()}</div>
          </div>

          <div class="actions">
            <form method="post" action="/ui/task/{t.id}/complete">
              <input type="hidden" name="project" value="{project}">
              <button class="btn2" type="submit">Complete</button>
            </form>

            <form method="post" action="/ui/task/{t.id}/reopen">
              <input type="hidden" name="project" value="{project}">
              <button class="btn3" type="submit">Reopen</button>
            </form>
          </div>

          <form class="edit" method="post" action="/ui/task/{t.id}/edit">
            <input type="hidden" name="project" value="{project}">
            <label>Starter</label>
            <textarea name="starter" placeholder="Starter (2 min)">{starter}</textarea>
            <label>DoD</label>
            <textarea name="dod" placeholder="Definition of Done">{dod}</textarea>
            <label>Notes</label>
            <textarea name="notes" placeholder="Notes / context">{notes}</textarea>
            <button class="btn" type="submit">Save</button>
          </form>
        </div>
        """

    plan_safe = (
        plan.content.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    now = datetime.now().strftime("%b %d, %Y • %I:%M %p")

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Today — {project}</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system; background: #0b1220; color: #e5e7eb; margin: 0; }}
    .wrap {{ max-width: 980px; margin: 0 auto; padding: 18px; }}
    .card {{ background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 16px; }}
    .row {{ display:flex; gap:12px; flex-wrap: wrap; align-items:center; justify-content: space-between; }}
    .left {{ display:flex; gap:12px; flex-wrap: wrap; align-items:center; }}
    button {{ cursor:pointer; border:0; border-radius: 12px; padding: 10px 12px; font-weight: 800; }}
    .btn {{ background:#38bdf8; color:#06202a; }}
    .btn2 {{ background:#22c55e; color:#06220f; }}
    .btn3 {{ background:#f59e0b; color:#241400; }}
    pre {{ white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular; font-size: 13px; line-height: 1.55; margin:0; }}
    .muted {{ color:#94a3b8; font-size: 12px; }}
    .grid {{ display:grid; grid-template-columns: 1fr; gap: 12px; margin-top: 12px; }}
    .task {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 12px; }}
    .taskTop {{ display:flex; flex-direction:column; gap:6px; }}
    .title {{ font-size: 14px; }}
    .meta {{ font-size: 12px; color:#94a3b8; }}
    .actions {{ display:flex; gap:10px; margin-top: 10px; }}
    .edit {{ margin-top: 10px; display:flex; flex-direction:column; gap:8px; }}
    label {{ font-size: 12px; color:#94a3b8; }}
    textarea {{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.12);
      color:#e5e7eb;
      border-radius: 12px;
      padding: 10px 12px;
      width: 100%;
      min-height: 56px;
      font-family: ui-monospace, SFMono-Regular;
      font-size: 12px;
      line-height: 1.4;
    }}
    .topbar {{ margin-bottom: 10px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card topbar">
      <div class="row">
        <div class="left">
          <a class="muted" href="/docs" target="_blank">API docs</a>
          <span class="muted">Project: <b>{project}</b></span>
          <span class="muted">Generated: {now}</span>
        </div>
        <div class="left">
          <form method="post" action="/ui/refresh">
            <input type="hidden" name="project" value="{project}">
            <button class="btn" type="submit">Refresh Plan</button>
          </form>
          <form method="post" action="/ui/cleanup_microtasks">
            <input type="hidden" name="project" value="{project}">
            <button class="btn3" type="submit">Cleanup microtask junk</button>
          </form>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="muted" style="margin-bottom:10px;">Today’s Plan</div>
      <pre>{plan_safe}</pre>
    </div>

    <div class="grid">
      {rows_html if rows_html else '<div class="card"><div class="muted">No tasks available.</div></div>'}
    </div>
  </div>
</body>
</html>
"""
    return HTMLResponse(html)


@router.post("/ui/refresh")
async def ui_refresh(
    project: str = Form(default="onestream"),
):
    return RedirectResponse(url=f"/ui/today?project={project}", status_code=303)


@router.post("/ui/task/{task_id}/complete")
def ui_complete(
    task_id: int,
    project: str = Form(default="onestream"),
    db: Session = Depends(get_db),
):
    t = db.get(Task, task_id)
    if t and not t.completed:
        t.completed = True
        t.completed_at = datetime.utcnow()
        db.commit()
    return RedirectResponse(url=f"/ui/today?project={project}", status_code=303)


@router.post("/ui/task/{task_id}/reopen")
def ui_reopen(
    task_id: int,
    project: str = Form(default="onestream"),
    db: Session = Depends(get_db),
):
    t = db.get(Task, task_id)
    if t and t.completed:
        t.completed = False
        t.completed_at = None
        db.commit()
    return RedirectResponse(url=f"/ui/today?project={project}", status_code=303)


@router.post("/ui/task/{task_id}/edit")
def ui_edit(
    task_id: int,
    project: str = Form(default="onestream"),
    starter: str = Form(default=""),
    dod: str = Form(default=""),
    notes: str = Form(default=""),
    db: Session = Depends(get_db),
):
    t = db.get(Task, task_id)
    if t:
        t.starter = starter.strip() or None
        t.dod = dod.strip() or None
        t.notes = notes.strip() or None
        db.commit()
    return RedirectResponse(url=f"/ui/today?project={project}", status_code=303)
