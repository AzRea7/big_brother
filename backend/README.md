# Goal Autopilot (AI + Automation)

## Run
1) Copy env:
   cp .env.example .env

2) Start:
   docker compose up --build

3) Health:
   curl http://127.0.0.1:8000/health

## Add a goal
curl -X POST http://127.0.0.1:8000/goals \
  -H "Content-Type: application/json" \
  -d '{"title":"Ship my Section 8 lead engine MVP","why":"I want a real business","target_date":"2026-02-28"}'

## Add tasks
curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"title":"Implement MLS ingestion stub","goal_id":1,"due_date":"2026-01-15","priority":4,"estimated_minutes":90}'

## Trigger daily plan immediately (debug)
curl -X POST http://127.0.0.1:8000/debug/run/daily

## Webhook
Set WEBHOOK_URL in .env (Discord/Slack/etc).
The app POSTs a JSON payload:
{ "content": "<message>" }

## Email
Set SMTP_* and EMAIL_* in .env
