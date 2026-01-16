from __future__ import annotations

import json
from typing import Any

import httpx

from ..config import settings
from .prompts import SYSTEM_PROMPT


class LLMClient:
    """
    OpenAI-compatible chat-completions client.

    Works with:
    - OpenAI
    - LM Studio / local OpenAI-compatible servers

    Key behavior:
    - LLM must be intentionally enabled by configuration (if LLM_ENABLED exists)
    - Uses a minimal payload for compatibility
    - On HTTP error, includes response body to diagnose 400s
    """

    def __init__(self) -> None:
        self.base_url = (getattr(settings, "OPENAI_BASE_URL", "") or "").rstrip("/")
        self.api_key = getattr(settings, "OPENAI_API_KEY", None)
        self.model = (getattr(settings, "OPENAI_MODEL", "") or "").strip()
        self.llm_enabled_flag = getattr(settings, "LLM_ENABLED", None)

    def enabled(self) -> bool:
        # If the app has an explicit flag, respect it
        if self.llm_enabled_flag is not None and not bool(self.llm_enabled_flag):
            return False

        # Otherwise: enabled only if base_url + model are present
        return bool(self.base_url) and bool(self.model)
    
    async def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 600,
    ) -> str:
        """
        Generic chat call used by repo task generation (and anything else).
        """
        if not self.enabled():
            raise RuntimeError(
                "LLM not enabled (set LLM_ENABLED=true and/or OPENAI_BASE_URL + OPENAI_MODEL)"
            )

        url = f"{self.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        timeout = httpx.Timeout(connect=10.0, read=240.0, write=30.0, pool=10.0)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, headers=headers, content=json.dumps(payload))
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = e.response.text
            except Exception:
                body = ""
            raise RuntimeError(
                f"LLM HTTP error {e.response.status_code} from {url}. "
                f"Response body: {body[:500]}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"LLM request failed to {url}: {e}") from e

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected LLM response format: {data}") from e

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        """
        Calls chat(), then parses JSON with hard failure if it's not JSON.
        This prevents silent 'seed' fallbacks and makes testing deterministic.
        """
        text = await self.chat(system=system, user=user, temperature=temperature, max_tokens=max_tokens)
        try:
            return json.loads(text)
        except Exception as e:
            raise RuntimeError(f"LLM did not return valid JSON. Got: {text[:800]}") from e

    async def generate(self, user_prompt: str) -> str:
        if not self.enabled():
            raise RuntimeError("LLM not enabled (set LLM_ENABLED=true and/or OPENAI_BASE_URL + OPENAI_MODEL)")

        url = f"{self.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Minimal, widely compatible payload for LM Studio / OpenAI-compatible servers.
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 350,
        }

        timeout = httpx.Timeout(connect=10.0, read=240.0, write=30.0, pool=10.0)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, headers=headers, content=json.dumps(payload))
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = e.response.text
            except Exception:
                body = ""
            raise RuntimeError(
                f"LLM HTTP error {e.response.status_code} from {url}. "
                f"Response body: {body[:500]}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"LLM request failed to {url}: {e}") from e

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected LLM response format: {data}") from e

     