from __future__ import annotations

import json
from typing import Any

import httpx

from ..config import settings
from .prompts import SYSTEM_PROMPT


class LLMClient:
    """
    OpenAI-compatible chat-completions client that works with:
    - LM Studio local OpenAI-compatible server
    - Any OpenAI-compatible endpoint

    Key fixes:
    - Doesn't require a "real" API key (dummy works)
    - Uses long read timeout so local generation won't disconnect
    - Forces non-streaming responses for simpler clients
    - Limits output length for speed
    """

    def __init__(self) -> None:
        self.base_url = settings.OPENAI_BASE_URL.rstrip("/")
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL

    def enabled(self) -> bool:
        # For LM Studio, API key can be 'dummy' (still counts as enabled),
        # and model/base_url must exist.
        return bool(self.base_url) and bool(self.model)

    async def generate(self, user_prompt: str) -> str:
        if not self.enabled():
            raise RuntimeError("LLM not enabled (missing OPENAI_BASE_URL or OPENAI_MODEL)")

        url = f"{self.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        # LM Studio usually ignores auth, but keep for compatibility
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,      # IMPORTANT: avoid streaming to prevent client issues
            "temperature": 0.25,
            "max_tokens": 350,    # IMPORTANT: keep it short so it finishes fast
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }

        timeout = httpx.Timeout(
            connect=10.0,
            read=240.0,   # IMPORTANT: >30s so your local model can finish
            write=30.0,
            pool=10.0,
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, headers=headers, content=json.dumps(payload))
            r.raise_for_status()
            data = r.json()

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected LLM response format: {data}") from e
