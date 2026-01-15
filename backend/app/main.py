# backend/app/main.py
from __future__ import annotations

import os
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
from .routes.ui import router as ui_router
from .services.scheduler import start_scheduler, shutdown_scheduler
from .routes.dashboard import router as dashboard_router    

app = FastAPI(title="Goal Autopilot")


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    resp = await call_next(request)
    resp.headers["X-Response-Time-ms"] = str(int((time.time() - start) * 1000))
    return resp


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # In dev, return a useful payload instead of mystery "Internal Server Error"
    if settings.APP_ENV == "dev":
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type": exc.__class__.__name__,
                "traceback": traceback.format_exc(),
            },
        )
    return JSONResponse(status_code=500, content={"error": "Internal Server Error"})


@app.on_event("startup")
def on_startup():
    init_db()
    # Scheduler is optional; it can stay off in dev if you want.
    start_scheduler()


@app.on_event("shutdown")
def on_shutdown():
    shutdown_scheduler()


app.include_router(health_router)
app.include_router(goals_router)
app.include_router(tasks_router)
app.include_router(debug_router)
app.include_router(ui_router)
app.include_router(dashboard_router)
