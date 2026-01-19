# backend/app/services/security.py
from __future__ import annotations

from fastapi import HTTPException, Request

from ..config import settings


def require_api_key(request: Request) -> None:
    """
    Protect sensitive debug endpoints.

    Client must send header:
      X-API-Key: <settings.API_KEY>
    """
    expected = settings.API_KEY or ""
    provided = request.headers.get("X-API-Key", "")

    if not expected or expected == "change-me":
        # In dev, we still allow, but you should set a real key for prod.
        # If you want to hard-fail even in dev, delete this special-case.
        if settings.ENV != "prod":
            return

    if provided != expected:
        raise HTTPException(status_code=401, detail="Missing/invalid X-API-Key")


def debug_is_allowed() -> bool:
    """
    If DISABLE_DEBUG_IN_PROD is enabled, block /debug/* when ENV=prod.
    """
    if settings.ENV == "prod" and settings.DISABLE_DEBUG_IN_PROD:
        return False
    return True
