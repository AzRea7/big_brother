# backend/app/routes/ui.py
from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])


@router.get("/ui/today", response_class=HTMLResponse)
def ui_today(project: str = Query(default="onestream")):
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Today — {project}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system; background: #0b1220; color: #e5e7eb; margin: 0; }}
    .wrap {{ max-width: 980px; margin: 0 auto; padding: 22px; }}
    .card {{ background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 18px; }}
    .row {{ display:flex; gap:12px; flex-wrap: wrap; align-items:center; }}
    button {{ cursor:pointer; border:0; border-radius: 12px; padding: 10px 12px; font-weight: 700; }}
    .btn {{ background:#38bdf8; color:#06202a; }}
    .btn2 {{ background:#22c55e; color:#06220f; }}
    .btn3 {{ background:#f59e0b; color:#241400; }}
    pre {{ white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular; font-size: 13px; line-height: 1.55; }}
    input {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); color:#e5e7eb;
             border-radius: 12px; padding: 10px 12px; width: 160px; }}
    .muted {{ color:#94a3b8; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="row">
        <button class="btn" onclick="refreshPlan()">Refresh plan</button>
        <input id="taskId" placeholder="task id (e.g. 12)" />
        <button class="btn2" onclick="completeAndRefresh()">Complete + Refresh</button>
        <a class="muted" href="/docs" target="_blank">API docs</a>
      </div>
      <div style="margin-top:14px;">
        <div class="muted">Project: <b>{project}</b></div>
        <pre id="plan">Loading...</pre>
      </div>
    </div>
  </div>

<script>
async function refreshPlan() {{
  const r = await fetch(`/debug/run/daily?project={project}&mode=single`, {{ method: "POST" }});
  const text = await r.text();
  try {{
    const j = JSON.parse(text);
    document.getElementById("plan").textContent = j.content || text;
  }} catch (e) {{
    document.getElementById("plan").textContent = text;
  }}
}}

async function completeAndRefresh() {{
  const id = document.getElementById("taskId").value.trim();
  if (!id) return alert("Enter a task id.");
  const r = await fetch(`/tasks/${{id}}/complete_and_refresh?project={project}&mode=single`, {{ method: "POST" }});
  const text = await r.text();
  try {{
    const j = JSON.parse(text);
    document.getElementById("plan").textContent = j.new_plan?.content || text;
  }} catch (e) {{
    document.getElementById("plan").textContent = text;
  }}
}}

refreshPlan();
</script>
</body>
</html>
"""
    return HTMLResponse(html)
