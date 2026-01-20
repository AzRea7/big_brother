# backend/app/services/repo_embeddings.py
from __future__ import annotations

import json
import math
from typing import List, Optional

import httpx

from ..config import settings


def _l2_norm(vec: List[float]) -> float:
    s = 0.0
    for x in vec:
        s += x * x
    return math.sqrt(s)


def cosine_similarity(a: List[float], b: List[float], a_norm: Optional[float] = None, b_norm: Optional[float] = None) -> float:
    if not a or not b:
        return -1.0
    if len(a) != len(b):
        return -1.0
    dot = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
    na = a_norm if a_norm is not None else _l2_norm(a)
    nb = b_norm if b_norm is not None else _l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (na * nb)


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Returns embeddings for texts using configured provider.

    Provider "off" -> raises RuntimeError (callers should gate).
    Provider "openai" -> uses OpenAI embeddings endpoint.
    """
    provider = (settings.EMBEDDINGS_PROVIDER or "off").lower().strip()
    if provider == "off":
        raise RuntimeError("Embeddings provider is off")

    if provider != "openai":
        raise RuntimeError(f"Unsupported EMBEDDINGS_PROVIDER={provider}")

    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = f"{settings.OPENAI_BASE_URL.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}

    payload = {
        "model": settings.EMBEDDINGS_MODEL,
        "input": texts,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # Expected: data["data"][i]["embedding"]
    out: List[List[float]] = []
    for item in data.get("data", []):
        out.append(item.get("embedding", []))
    if len(out) != len(texts):
        raise RuntimeError("Unexpected embeddings response length mismatch")
    return out


def dumps_embedding(vec: List[float]) -> str:
    return json.dumps(vec, separators=(",", ":"))


def loads_embedding(s: str) -> List[float]:
    return json.loads(s)
