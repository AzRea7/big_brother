# backend/app/services/notifier.py
from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

import httpx

from ..config import settings


async def send_webhook(message: str) -> None:
    if not settings.WEBHOOK_URL:
        return
    payload = {"content": message}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(settings.WEBHOOK_URL, json=payload)
        r.raise_for_status()


def _render_html_email(subject: str, content: str, project: str) -> str:
    safe = (
        content.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )

    now = datetime.now().strftime("%b %d, %Y • %I:%M %p")

    base = settings.PUBLIC_BASE_URL.rstrip("/")
    ui_today = f"{base}/ui/today?project={project}"
    docs = f"{base}/docs"
    regen = f"{base}/debug/run/daily?project={project}&mode=single"

    return f"""\
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{subject}</title>
</head>
<body style="margin:0;padding:0;background:#0b1220;font-family:Arial,Helvetica,sans-serif;">
  <div style="max-width:720px;margin:0 auto;padding:20px;">
    <div style="background:linear-gradient(135deg,#2563eb,#7c3aed);border-radius:16px;padding:18px 18px 14px 18px;color:#fff;">
      <div style="font-size:12px;opacity:0.9;">Goal Autopilot</div>
      <div style="font-size:22px;font-weight:700;line-height:1.2;margin-top:4px;">{subject}</div>
      <div style="font-size:12px;opacity:0.9;margin-top:6px;">Generated {now}</div>
    </div>

    <div style="margin-top:14px;display:flex;gap:10px;flex-wrap:wrap;">
      <a href="{ui_today}"
         style="text-decoration:none;background:#22c55e;color:#06220f;padding:10px 12px;border-radius:12px;font-weight:700;font-size:13px;">
         Open Today UI (complete/edit tasks)
      </a>
      <a href="{regen}"
         style="text-decoration:none;background:#38bdf8;color:#06202a;padding:10px 12px;border-radius:12px;font-weight:700;font-size:13px;">
         Regenerate Plan
      </a>
      <a href="{docs}"
         style="text-decoration:none;background:#f59e0b;color:#241400;padding:10px 12px;border-radius:12px;font-weight:700;font-size:13px;">
         API Docs
      </a>
    </div>

    <div style="background:#0f172a;border-radius:16px;padding:18px;margin-top:14px;color:#e5e7eb;border:1px solid rgba(255,255,255,0.08);">
      <div style="font-size:14px;line-height:1.65;color:#e5e7eb;">
        {safe}
      </div>
    </div>

    <div style="margin-top:14px;font-size:12px;color:#94a3b8;line-height:1.5;">
      Tip: If Top 3 are mostly [MICRO] tasks, fill Starter/DoD on the parent tasks.
      Next day the plan gets sharper and stops nagging.
    </div>
  </div>
</body>
</html>
"""


def send_email(subject: str, message: str, project: str = "onestream") -> None:
    if not (
        settings.SMTP_HOST
        and settings.SMTP_USER
        and settings.SMTP_PASS
        and settings.EMAIL_FROM
        and settings.EMAIL_TO
    ):
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.EMAIL_FROM
    msg["To"] = settings.EMAIL_TO

    text_part = MIMEText(message, "plain", "utf-8")
    html_part = MIMEText(_render_html_email(subject, message, project=project), "html", "utf-8")

    msg.attach(text_part)
    msg.attach(html_part)

    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
        server.starttls()
        server.login(settings.SMTP_USER, settings.SMTP_PASS)
        server.sendmail(settings.EMAIL_FROM, [settings.EMAIL_TO], msg.as_string())
