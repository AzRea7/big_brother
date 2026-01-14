from __future__ import annotations

import os
import time
from fastapi import FastAPI

from .config import settings
from .db import init_db
from .routes.health import router as health_router
from .routes.goals import router as goals_router
from .routes.tasks import router as tasks_router
from .routes.debug import router as debug_router
from .services.scheduler import start_scheduler, shutdown_scheduler

# Ensure TZ is honored in container
os.environ["TZ"] = settings.TZ
try:
    time.tzset()  # type: ignore[attr-defined]
except Exception:
    pass

app = FastAPI(title="Goal Autopilot", version="1.0.0")

app.include_router(health_router)
app.include_router(goals_router)
app.include_router(tasks_router)
app.include_router(debug_router)


@app.on_event("startup")
def on_startup():
    init_db()
    start_scheduler()


@app.on_event("shutdown")
def on_shutdown():
    shutdown_scheduler()
