# backend/app/services/security.py
from __future__ import annotations

from typing import Any

from fastapi import Header, HTTPException, Request

from ..config import settings


def require_api_key(
    arg: Any = None,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """
    Supports three calling styles:

    1) FastAPI dependency injection:
         require_api_key()  # FastAPI provides x_api_key

    2) Manual with Request:
         require_api_key(request)

    3) Manual with raw string:
         require_api_key("...") or require_api_key(request.headers.get("X-API-Key"))
    """
    # Production safety gate
    if settings.ENV.lower() == "prod" and settings.DISABLE_DEBUG_IN_PROD:
        raise HTTPException(status_code=403, detail="Debug endpoints disabled in prod")

    key: str | None = None

    # Called like require_api_key(request)
    if isinstance(arg, Request):
        key = arg.headers.get("X-API-Key") or x_api_key

    # Called like require_api_key("abc123")
    elif isinstance(arg, str):
        key = arg or x_api_key

    # Called as dependency (or weird call)
    else:
        key = x_api_key

    if not key or key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def debug_is_allowed() -> bool:
    if settings.ENV == "prod" and settings.DISABLE_DEBUG_IN_PROD:
        return False
    return True
