# backend/app/services/metrics.py
from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "goal_autopilot_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_LATENCY_SECONDS = Histogram(
    "goal_autopilot_request_latency_seconds",
    "Request latency in seconds",
    ["path"],
)

JOBS_TOTAL = Counter(
    "goal_autopilot_jobs_total",
    "Background/debug job runs",
    ["job", "status"],
)
