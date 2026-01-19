# backend/app/routes/metrics.py
from __future__ import annotations

from fastapi import APIRouter, Response

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..config import settings

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def metrics() -> Response:
    if not settings.ENABLE_METRICS:
        return Response(status_code=404, content="metrics disabled")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
