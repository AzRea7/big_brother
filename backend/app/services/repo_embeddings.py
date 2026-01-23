from __future__ import annotations

import math
from typing import List, Optional

from ..config import settings


def _l2_norm(vec: List[float]) -> float:
    s = 0.0
    for x in vec:
        fx = float(x)
        s += fx * fx
    return math.sqrt(s)


def cosine_similarity(
    a: List[float],
    b: List[float],
    *,
    a_norm: Optional[float] = None,
    b_norm: Optional[float] = None,
) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if norms are ~0 or vectors are empty."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0

    dot = 0.0
    if a_norm is None:
        na = 0.0
    else:
        na = float(a_norm) * float(a_norm)

    if b_norm is None:
        nb = 0.0
    else:
        nb = float(b_norm) * float(b_norm)

    for i in range(n):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        if a_norm is None:
            na += av * av
        if b_norm is None:
            nb += bv * bv

    denom = math.sqrt(na) * math.sqrt(nb)
    if denom <= 1e-12:
        return 0.0
    return dot / denom


def embeddings_enabled() -> bool:
    provider = str(getattr(settings, "EMBEDDINGS_PROVIDER", "off") or "off").lower().strip()
    enabled = bool(getattr(settings, "EMBEDDINGS_ENABLED", False))
    return enabled and provider != "off"


def embedding_model_name() -> str:
    return str(getattr(settings, "EMBEDDINGS_MODEL", "") or "text-embedding-3-small")


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Delegates to app.ai.embeddings to avoid duplicating provider logic."""
    if not embeddings_enabled():
        raise RuntimeError("Embeddings are disabled (EMBEDDINGS_ENABLED=false or EMBEDDINGS_PROVIDER=off).")

    provider = str(getattr(settings, "EMBEDDINGS_PROVIDER", "off") or "off").lower().strip()
    if provider != "openai":
        raise RuntimeError(f"Unsupported EMBEDDINGS_PROVIDER={provider}")

    from ..ai.embeddings import embed_texts as _embed  # local import avoids cycles

    return await _embed(texts)


def embedding_norm(vec: List[float]) -> float:
    return _l2_norm(vec)
