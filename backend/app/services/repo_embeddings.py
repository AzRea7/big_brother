# backend/app/services/repo_embeddings.py
from __future__ import annotations

import math
from typing import List, Optional

import httpx

from ..config import settings


def _l2_norm(vec: List[float]) -> float:
    s = 0.0
    for x in vec:
        s += x * x
    return math.sqrt(s)


def cosine_similarity(
    a: List[float],
    b: List[float],
    a_norm: Optional[float] = None,
    b_norm: Optional[float] = None,
) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
    na = a_norm if a_norm is not None else _l2_norm(a)
    nb = b_norm if b_norm is not None else _l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (na * nb)


def embeddings_enabled() -> bool:
    return (settings.EMBEDDINGS_PROVIDER or "off").lower().strip() != "off"


def embedding_model_name() -> str:
    return str(settings.EMBEDDINGS_MODEL or "text-embedding-3-small")


def _openai_base() -> str:
    # IMPORTANT: OpenAI-compatible base WITHOUT /v1 at the end
    # e.g. https://api.openai.com
    base = str(settings.OPENAI_BASE_URL or "https://api.openai.com").rstrip("/")
    return base


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Canonical embeddings call used by chunk embedding + retrieval.
    OpenAI-compatible:
      POST {base}/v1/embeddings
      { "model": "...", "input": [...] }
    """
    provider = (settings.EMBEDDINGS_PROVIDER or "off").lower().strip()
    if provider == "off":
        raise RuntimeError("Embeddings provider is off")
    if provider != "openai":
        raise RuntimeError(f"Unsupported EMBEDDINGS_PROVIDER={provider}")

    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    if not texts:
        return []

    url = f"{_openai_base()}/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": embedding_model_name(), "input": texts}

    timeout_s = float(getattr(settings, "OPENAI_TIMEOUT_S", 60.0) or 60.0)

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"Embeddings failed {resp.status_code}: {resp.text}")

        data = resp.json()
        items = data.get("data") or []
        items = sorted(items, key=lambda x: int(x.get("index", 0)))

        out: List[List[float]] = []
        for it in items:
            emb = it.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError("Embeddings response missing vector list.")
            out.append([float(v) for v in emb])
        return out


def loads_embedding(s: str | None) -> List[float]:
    """
    Backward-compatible loader for embeddings stored as JSON strings.
    """
    if not s:
        return []
    try:
        import json

        x = json.loads(s)
        if isinstance(x, list):
            return [float(v) for v in x]
        return []
    except Exception:
        return []
