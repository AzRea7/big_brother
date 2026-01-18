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
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")  # ", }" or ", ]" -> "}" / "]"


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    m = _JSON_FENCE_RE.search(s)
    if m:
        return (m.group(1) or "").strip()
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


def _count_brackets_outside_strings(s: str) -> tuple[int, int, int, int]:
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

    return opens_curly, closes_curly, opens_brack, closes_brack


def _repair_truncated_json(s: str) -> str:
    """
    Best-effort repair for truncated LLM JSON.

    Handles the common failure mode you hit:
      - JSON ends with a '}' but is missing the closing ']' for an array inside it.
      - naive repair appends ']' AFTER '}', creating invalid nesting and the
        "Expecting ',' delimiter" JSONDecodeError.

    Strategy:
      1) Trim to last '}' or ']'
      2) Remove trailing commas before '}' or ']'
      3) Count braces/brackets outside strings
      4) If missing ']' and string ends with '}', INSERT ']' before final '}' (correct nesting)
      5) Append any missing '}' at the end
    """
    s = (s or "").strip()

    # 1) Trim trailing junk after last JSON-ish closer if any
    last_curly = s.rfind("}")
    last_brack = s.rfind("]")
    last = max(last_curly, last_brack)
    if last != -1:
        s = s[: last + 1]

    # 2) Remove trailing commas before closing tokens
    s = _TRAILING_COMMA_RE.sub(r"\1", s)

    # 3) Count open/close tokens outside strings
    opens_curly, closes_curly, opens_brack, closes_brack = _count_brackets_outside_strings(s)

    # If we have too many closing braces/brackets, trim extras from the end only
    extra_curly = max(0, closes_curly - opens_curly)
    extra_brack = max(0, closes_brack - opens_brack)
    while extra_curly > 0 and s.rstrip().endswith("}"):
        s = s.rstrip()
        s = s[:-1].rstrip()
        extra_curly -= 1
    while extra_brack > 0 and s.rstrip().endswith("]"):
        s = s.rstrip()
        s = s[:-1].rstrip()
        extra_brack -= 1

    # Re-count after trimming
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    opens_curly, closes_curly, opens_brack, closes_brack = _count_brackets_outside_strings(s)

    missing_brack = max(0, opens_brack - closes_brack)
    missing_curly = max(0, opens_curly - closes_curly)

    # 4) Insert missing ']' BEFORE the final '}' when we end with an object close.
    # This is the key fix for your error.
    if missing_brack > 0:
        stripped = s.rstrip()
        if stripped.endswith("}"):
            insert_at = stripped.rfind("}")
            # Insert all missing brackets before the last '}' (keeps nesting valid)
            s = stripped[:insert_at] + ("]" * missing_brack) + stripped[insert_at:]
            missing_brack = 0
        else:
            s = stripped + ("]" * missing_brack)
            missing_brack = 0

    # 5) Append missing '}' at the end
    if missing_curly > 0:
        s = s.rstrip() + ("}" * missing_curly)

    # Final trailing-comma cleanup
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
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
        raw = await self.chat(system=system, user=user, temperature=temperature, max_tokens=max_tokens)

        # strict attempt first
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
    llm = LLMClient()
    if not llm.enabled():
        raise LLMError("LLM is disabled (LLM_ENABLED=false or missing OPENAI_BASE_URL/OPENAI_MODEL).")
    return await llm.chat_json(system=system, user=user, temperature=0.2, max_tokens=2000)
