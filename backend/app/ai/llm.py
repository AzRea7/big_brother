# backend/app/ai/llm.py
from __future__ import annotations

import json
import re
from typing import Any

import httpx

from ..config import settings


class LLMError(RuntimeError):
    pass


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    m = _JSON_FENCE_RE.search(s)
    if m:
        return (m.group(1) or "").strip()
    # also handle raw triple-fence without regex capture (rare)
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_balanced_json_prefix(s: str) -> str | None:
    """
    Extract earliest balanced JSON object/array starting at first '{' or '['.
    Returns None if never balanced (truncated) or malformed.
    """
    s = (s or "").strip()
    first_obj = s.find("{")
    first_arr = s.find("[")
    if first_obj == -1 and first_arr == -1:
        return None

    start = first_obj if (first_arr == -1 or (first_obj != -1 and first_obj < first_arr)) else first_arr

    depth_obj = 0
    depth_arr = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth_obj += 1
        elif ch == "}":
            depth_obj -= 1
        elif ch == "[":
            depth_arr += 1
        elif ch == "]":
            depth_arr -= 1

        if depth_obj == 0 and depth_arr == 0 and i >= start:
            return s[start : i + 1]

        if depth_obj < 0 or depth_arr < 0:
            return None

    return None


def _repair_truncated_json(s: str) -> str:
    """
    Best-effort close missing ']' / '}' at end. Also trims obvious extra closers.
    Conservative: only adjusts the tail.
    """
    s = (s or "").strip()

    # Trim trailing junk after last JSON-ish closer if any
    last_curly = s.rfind("}")
    last_brack = s.rfind("]")
    last = max(last_curly, last_brack)
    if last != -1:
        s = s[: last + 1]

    # Count braces/brackets outside strings
    in_str = False
    esc = False
    opens_curly = closes_curly = 0
    opens_brack = closes_brack = 0

    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            opens_curly += 1
        elif ch == "}":
            closes_curly += 1
        elif ch == "[":
            opens_brack += 1
        elif ch == "]":
            closes_brack += 1

    # If we have too many closing braces, trim extras from the end only
    extra_curly = max(0, closes_curly - opens_curly)
    extra_brack = max(0, closes_brack - opens_brack)
    while extra_curly > 0 and s.endswith("}"):
        s = s[:-1].rstrip()
        extra_curly -= 1
    while extra_brack > 0 and s.endswith("]"):
        s = s[:-1].rstrip()
        extra_brack -= 1

    # Append missing closers (arrays then objects)
    missing_brack = max(0, opens_brack - closes_brack)
    missing_curly = max(0, opens_curly - closes_curly)
    if missing_brack:
        s += "]" * missing_brack
    if missing_curly:
        s += "}" * missing_curly

    return s


def _extract_json_object_lenient(raw: str) -> Any:
    """
    Robust JSON extraction for LLM outputs.
    Strategy:
      1) strip code fences
      2) extract balanced JSON prefix and parse
      3) if parse fails, repair + retry
      4) fallback: first '{'..last '}' then parse/repair
      5) fallback: first '['..last ']' then parse/repair
    """
    s = _strip_code_fences(raw)

    cand = _extract_balanced_json_prefix(s)
    if cand is not None:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            return json.loads(_repair_truncated_json(cand))

    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        cand2 = s[first : last + 1]
        try:
            return json.loads(cand2)
        except json.JSONDecodeError:
            return json.loads(_repair_truncated_json(cand2))

    first = s.find("[")
    last = s.rfind("]")
    if first != -1 and last != -1 and last > first:
        cand3 = s[first : last + 1]
        try:
            return json.loads(cand3)
        except json.JSONDecodeError:
            return json.loads(_repair_truncated_json(cand3))

    raise ValueError("No JSON object or array found in LLM output.")


class LLMClient:
    """
    OpenAI-compatible chat-completions client.

    Works with:
    - OpenAI
    - LM Studio / local OpenAI-compatible servers
    - Any server exposing POST {base_url}/chat/completions
      (base_url should usually end with /v1)

    Settings expected:
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
        if self.llm_enabled_flag is not None and not bool(self.llm_enabled_flag):
            return False
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
        Returns assistant message content as plain text.
        """
        if not self.enabled():
            raise LLMError("LLM not enabled. Set LLM_ENABLED=true and OPENAI_BASE_URL + OPENAI_MODEL.")

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

        # If your server supports it, this can help. Safe if ignored.
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
            return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
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
        Calls chat() and parses JSON from the response with lenient fallback.
        """
        raw = await self.chat(system=system, user=user, temperature=temperature, max_tokens=max_tokens)

        # strict attempt first (fast)
        try:
            obj = json.loads(_strip_code_fences(raw))
            if isinstance(obj, dict):
                return obj
            raise ValueError("Top-level JSON is not an object")
        except Exception:
            obj2 = _extract_json_object_lenient(raw)
            if not isinstance(obj2, dict):
                raise ValueError("Top-level JSON is not an object after lenient parse")
            return obj2


async def chat_completion_json(system: str, user: str) -> dict[str, Any]:
    """
    Backwards-compatible helper used by repo_llm_findings.py.
    """
    llm = LLMClient()
    if not llm.enabled():
        raise LLMError("LLM is disabled (LLM_ENABLED=false or missing OPENAI_BASE_URL/OPENAI_MODEL).")
    return await llm.chat_json(system=system, user=user, temperature=0.2, max_tokens=2000)
