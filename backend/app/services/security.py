# backend/app/services/security.py
from __future__ import annotations

from fastapi import Header, HTTPException

from ..config import settings


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if settings.ENV.lower() == "prod" and settings.DISABLE_DEBUG_IN_PROD:
        # In prod, you can still allow debug by flipping DISABLE_DEBUG_IN_PROD=false
        # but default is "off".
        raise HTTPException(status_code=403, detail="Debug endpoints disabled in prod")

    if not x_api_key or x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")



def debug_is_allowed() -> bool:
    """
    If DISABLE_DEBUG_IN_PROD is enabled, block /debug/* when ENV=prod.
    """
    if settings.ENV == "prod" and settings.DISABLE_DEBUG_IN_PROD:
        return False
    return True
