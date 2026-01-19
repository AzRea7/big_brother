# backend/app/main.py
from __future__ import annotations

import time
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import settings
from .db import init_db

from .routes.health import router as health_router
from .routes.goals import router as goals_router
from .routes.tasks import router as tasks_router
from .routes.debug import router as debug_router
from .routes.repo import router as repo_router
from .routes.ui import router as ui_router
from .routes.dashboard import router as dashboard_router
from .routes.metrics import router as metrics_router

from .services.request_id import request_id_middleware
from .services.metrics import REQUESTS_TOTAL, REQUEST_LATENCY_SECONDS


app = FastAPI(title="Goal Autopilot")


@app.on_event("startup")
def _startup() -> None:
    init_db()


# --- request id middleware (first) ---
@app.middleware("http")
async def _rid(request: Request, call_next):
    return await request_id_middleware(request, call_next)


# --- timing + metrics ---
@app.middleware("http")
async def timing_and_metrics_middleware(request: Request, call_next):
    start = time.time()
    resp = await call_next(request)
    dur = time.time() - start

    path = request.url.path
    resp.headers["X-Response-Time-ms"] = str(int(dur * 1000))

    # Prometheus
    try:
        REQUESTS_TOTAL.labels(method=request.method, path=path, status=str(resp.status_code)).inc()
        REQUEST_LATENCY_SECONDS.labels(path=path).observe(dur)
    except Exception:
        # Never let metrics break requests
        pass

    return resp


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # In dev, return a useful stack trace; in prod, be minimal.
    if settings.ENV != "prod":
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "request_id": getattr(request.state, "request_id", None),
            },
        )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "request_id": getattr(request.state, "request_id", None)},
    )


# Routers
app.include_router(health_router)
app.include_router(goals_router)
app.include_router(tasks_router)
app.include_router(ui_router)
app.include_router(dashboard_router)

# Debug / repo debugging
app.include_router(debug_router)
app.include_router(repo_router)

# Metrics
app.include_router(metrics_router)
