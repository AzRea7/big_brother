# backend/app/services/request_id.py
from __future__ import annotations

import uuid
from typing import Callable

from fastapi import Request, Response


REQUEST_ID_HEADER = "X-Request-Id"


def get_or_create_request_id(request: Request) -> str:
    rid = request.headers.get(REQUEST_ID_HEADER)
    if rid:
        return rid
    return str(uuid.uuid4())


async def request_id_middleware(request: Request, call_next: Callable) -> Response:
    rid = get_or_create_request_id(request)
    request.state.request_id = rid
    response: Response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = rid
    return response
