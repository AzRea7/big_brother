# backend/app/ai/embeddings.py
from __future__ import annotations

from typing import List

import httpx

from ..config import settings


class EmbeddingsError(RuntimeError):
    pass


def _base_url() -> str:
    # Many repos already use OPENAI_BASE_URL in settings.
    return str(getattr(settings, "OPENAI_BASE_URL", "") or "https://api.openai.com")


def _api_key() -> str:
    key = str(getattr(settings, "OPENAI_API_KEY", "") or "")
    if not key:
        raise EmbeddingsError("OPENAI_API_KEY is required for embeddings.")
    return key


def _model() -> str:
    return str(getattr(settings, "OPENAI_EMBEDDING_MODEL", "") or "text-embedding-3-small")


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    OpenAI-compatible embeddings call:
      POST {base}/v1/embeddings
      { "model": "...", "input": ["text1", "text2"] }
    """
    if not texts:
        return []

    base = _base_url().rstrip("/")
    url = f"{base}/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }

    payload = {"model": _model(), "input": texts}

    timeout_s = float(getattr(settings, "OPENAI_TIMEOUT_S", 60.0) or 60.0)

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise EmbeddingsError(f"Embeddings failed {r.status_code}: {r.text}")

        data = r.json()
        items = data.get("data") or []
        # each: {"index":0,"embedding":[...],"object":"embedding"}
        # preserve order by index
        items = sorted(items, key=lambda x: int(x.get("index", 0)))
        out: List[List[float]] = []
        for it in items:
            emb = it.get("embedding")
            if not isinstance(emb, list):
                raise EmbeddingsError("Embeddings response missing vector list.")
            out.append([float(x) for x in emb])
        return out
