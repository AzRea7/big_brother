# backend/app/ai/llm.py
from __future__ import annotations

import json
import re
from typing import Any, Optional

import httpx

from ..config import settings


class LLMError(RuntimeError):
    pass


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_json_object_lenient(raw: str) -> dict[str, Any]:
    """
    Local models sometimes wrap JSON in text or markdown fences.
    We take the first {...} blob and parse it.
    """
    s = _strip_code_fences(raw)
    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError(f"No JSON object found. First 400 chars: {raw[:400]}")
    return json.loads(s[first : last + 1])


class LLMClient:
    """
    OpenAI-compatible chat-completions client.

    Works with:
    - OpenAI
    - LM Studio / local OpenAI-compatible servers
    - Any server exposing POST {base_url}/chat/completions
      (so base_url should usually end with /v1)

    Env/Settings expected:
      LLM_ENABLED (bool-ish)
      OPENAI_BASE_URL (e.g. http://host.docker.internal:1234/v1)
      OPENAI_MODEL (e.g. local-model-name)
      OPENAI_API_KEY (optional for local servers)
      LLM_READ_TIMEOUT_S (optional)
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
        max_tokens: int = 2000,
    ) -> str:
        """
        Returns the assistant message content as plain text.
        """
        if not self.enabled():
            raise LLMError(
                "LLM not enabled. Set LLM_ENABLED=true and OPENAI_BASE_URL + OPENAI_MODEL."
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

        # Optional: OpenAI-compatible JSON-only mode if available
        # (LM Studio supports it for some models; harmless if ignored by server)
        # payload["response_format"] = {"type": "json_object"}

        timeout = httpx.Timeout(
            connect=10.0,
            read=float(getattr(settings, "LLM_READ_TIMEOUT_S", 600.0)),
            write=60.0,
            pool=10.0,
        )

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
            raise LLMError(
                f"LLM HTTP error {e.response.status_code} from {url}. Response body: {body[:800]}"
            ) from e
        except Exception as e:
            raise LLMError(f"LLM request failed to {url}: {e}") from e

        try:
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        except Exception:
            return str(data)

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """
        Calls chat() and parses a JSON object from the response,
        with a lenient fallback for local models.
        """
        raw = await self.chat(
            system=system, user=user, temperature=temperature, max_tokens=max_tokens
        )
        try:
            return json.loads(_strip_code_fences(raw))
        except Exception:
            return _extract_json_object_lenient(raw)


async def chat_completion_json(system: str, user: str) -> dict[str, Any]:
    """
    Backwards-compatible helper used by repo_llm_findings.py.
    """
    llm = LLMClient()
    if not llm.enabled():
        raise LLMError("LLM is disabled (LLM_ENABLED=false or missing OPENAI_BASE_URL/OPENAI_MODEL).")
    return await llm.chat_json(system=system, user=user, temperature=0.2, max_tokens=2000)
