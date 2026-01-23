# backend/app/ai/llm.py
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import httpx

from ..config import settings

logger = logging.getLogger(__name__)


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

    Strategy:
      1) Trim to last '}' or ']'
      2) Remove trailing commas before '}' or ']'
      3) Count braces/brackets outside strings
      4) If missing ']' and string ends with '}', INSERT ']' before final '}' (correct nesting)
      5) Append any missing '}' at the end
    """
    s = (s or "").strip()

    last_curly = s.rfind("}")
    last_brack = s.rfind("]")
    last = max(last_curly, last_brack)
    if last != -1:
        s = s[: last + 1]

    s = _TRAILING_COMMA_RE.sub(r"\1", s)

    opens_curly, closes_curly, opens_brack, closes_brack = _count_brackets_outside_strings(s)

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

    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    opens_curly, closes_curly, opens_brack, closes_brack = _count_brackets_outside_strings(s)

    missing_brack = max(0, opens_brack - closes_brack)
    missing_curly = max(0, opens_curly - closes_curly)

    if missing_brack > 0:
        stripped = s.rstrip()
        if stripped.endswith("}"):
            insert_at = stripped.rfind("}")
            s = stripped[:insert_at] + ("]" * missing_brack) + stripped[insert_at:]
        else:
            s = stripped + ("]" * missing_brack)

    if missing_curly > 0:
        s = s.rstrip() + ("}" * missing_curly)

    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s


def _insert_missing_commas_contextual(s: str) -> str:
    """
    Context-aware comma inserter.

    Fixes BOTH:
      - missing commas between object members: {"a":1 "b":2}
      - missing commas between array elements: [{"a":1} {"b":2}] or [{...}{...}]

    Works by tracking container stack and whether we are "after a value"
    inside an object or array.
    """
    s = s or ""
    out: list[str] = []

    in_str = False
    esc = False

    # stack items: ("obj"|"arr", state)
    # obj states: "need_key", "need_colon", "need_value", "after_value"
    # arr states: "need_value", "after_value"
    stack: list[tuple[str, str]] = []

    def top() -> tuple[str, str] | None:
        return stack[-1] if stack else None

    def set_state(kind: str, state: str) -> None:
        if stack and stack[-1][0] == kind:
            stack[-1] = (kind, state)

    def next_non_ws(i: int) -> int:
        j = i
        while j < len(s) and s[j].isspace():
            j += 1
        return j

    def starts_value_at(i: int) -> bool:
        if i >= len(s):
            return False
        ch = s[i]
        if ch in "{[\"":
            return True
        if ch == "-" or ch.isdigit():
            return True
        # true/false/null
        if s.startswith("true", i) or s.startswith("false", i) or s.startswith("null", i):
            return True
        return False

    def is_key_at(i: int) -> bool:
        # key is a string followed by optional ws then colon
        if i >= len(s) or s[i] != '"':
            return False
        j = i + 1
        esc2 = False
        while j < len(s):
            ch = s[j]
            if esc2:
                esc2 = False
            elif ch == "\\":
                esc2 = True
            elif ch == '"':
                break
            j += 1
        if j >= len(s) or s[j] != '"':
            return False
        k = next_non_ws(j + 1)
        return k < len(s) and s[k] == ":"

    i = 0
    while i < len(s):
        ch = s[i]

        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                # string ended: could be key or value depending on state; state transitions happen on ':' / ',' / closers
            i += 1
            continue

        # Outside string: maybe insert comma if we are after_value and next token begins a new value/key
        t = top()
        if t is not None:
            kind, state = t
            if state == "after_value":
                j = i
                if s[j].isspace():
                    # do not decide until we hit a non-ws char
                    pass
                else:
                    if kind == "obj":
                        # valid next tokens: ',' or '}' ; if we see a key, insert comma
                        if ch not in {",", "}"} and is_key_at(i):
                            out.append(",")
                            set_state("obj", "need_key")
                    elif kind == "arr":
                        # valid next tokens: ',' or ']' ; if we see a value, insert comma
                        if ch not in {",", "]"} and starts_value_at(i):
                            out.append(",")
                            set_state("arr", "need_value")

        # Now process the actual character and update state machine
        if ch == '"':
            # entering string
            out.append(ch)
            in_str = True
            esc = False
            # If we're in obj and need_key, this is a key
            t2 = top()
            if t2 is not None and t2[0] == "obj" and t2[1] == "need_key":
                set_state("obj", "need_colon")
            # If we're in arr and need_value, a string value is starting
            if t2 is not None and t2[0] == "arr" and t2[1] == "need_value":
                set_state("arr", "after_value")
            i += 1
            continue

        if ch == "{":
            out.append(ch)
            stack.append(("obj", "need_key"))
            # If we were in an array needing a value, an object counts as a value
            if len(stack) >= 2 and stack[-2][0] == "arr" and stack[-2][1] == "need_value":
                stack[-2] = ("arr", "after_value")
            # If we were in an object needing a value, an object counts as a value
            if len(stack) >= 2 and stack[-2][0] == "obj" and stack[-2][1] == "need_value":
                stack[-2] = ("obj", "after_value")
            i += 1
            continue

        if ch == "[":
            out.append(ch)
            stack.append(("arr", "need_value"))
            # If we were in an array needing a value, an array counts as a value
            if len(stack) >= 2 and stack[-2][0] == "arr" and stack[-2][1] == "need_value":
                stack[-2] = ("arr", "after_value")
            # If we were in an object needing a value, an array counts as a value
            if len(stack) >= 2 and stack[-2][0] == "obj" and stack[-2][1] == "need_value":
                stack[-2] = ("obj", "after_value")
            i += 1
            continue

        if ch == ":":
            out.append(ch)
            # obj: key -> value
            if stack and stack[-1][0] == "obj" and stack[-1][1] == "need_colon":
                set_state("obj", "need_value")
            i += 1
            continue

        if ch == ",":
            out.append(ch)
            # comma resets state in current container
            if stack:
                kind, _state = stack[-1]
                if kind == "obj":
                    stack[-1] = ("obj", "need_key")
                else:
                    stack[-1] = ("arr", "need_value")
            i += 1
            continue

        if ch == "}":
            out.append(ch)
            if stack and stack[-1][0] == "obj":
                stack.pop()
            # closing object completes a value in parent
            if stack:
                pk, ps = stack[-1]
                if pk == "arr" and ps == "need_value":
                    stack[-1] = ("arr", "after_value")
                if pk == "obj" and ps == "need_value":
                    stack[-1] = ("obj", "after_value")
            i += 1
            continue

        if ch == "]":
            out.append(ch)
            if stack and stack[-1][0] == "arr":
                stack.pop()
            # closing array completes a value in parent
            if stack:
                pk, ps = stack[-1]
                if pk == "arr" and ps == "need_value":
                    stack[-1] = ("arr", "after_value")
                if pk == "obj" and ps == "need_value":
                    stack[-1] = ("obj", "after_value")
            i += 1
            continue

        # primitives: true/false/null/number
        if stack:
            kind, state = stack[-1]
            if state == "need_value":
                # If this position starts a primitive, mark after_value.
                if starts_value_at(i) and ch not in "{[\"":
                    if kind == "obj":
                        stack[-1] = ("obj", "after_value")
                    else:
                        stack[-1] = ("arr", "after_value")

        out.append(ch)
        i += 1

    return "".join(out)


def _extract_json_object_lenient(raw: str) -> Any:
    """
    Robust JSON extraction for LLM outputs.
    """
    s = _strip_code_fences(raw)

    def _loads_with_repairs(c: str) -> Any:
        # Attempt 1: raw
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            pass

        # Attempt 2: truncation repair
        c2 = _repair_truncated_json(c)
        try:
            return json.loads(c2)
        except json.JSONDecodeError:
            pass

        # Attempt 3: contextual missing-comma repair (handles arrays + objects)
        c3 = _insert_missing_commas_contextual(c)
        try:
            return json.loads(c3)
        except json.JSONDecodeError:
            pass

        # Attempt 4: truncation + contextual commas
        c4 = _insert_missing_commas_contextual(c2)
        return json.loads(c4)

    cand = _extract_balanced_json_prefix(s)
    if cand is not None:
        return _loads_with_repairs(cand)

    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        cand2 = s[first : last + 1]
        return _loads_with_repairs(cand2)

    first = s.find("[")
    last = s.rfind("]")
    if first != -1 and last != -1 and last > first:
        cand3 = s[first : last + 1]
        return _loads_with_repairs(cand3)

    raise ValueError("No JSON object or array found in LLM output.")


def _truncate_chars(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    s = s or ""
    return s if len(s) <= max_chars else s[:max_chars]


class LLMClient:
    """
    OpenAI-compatible client.

    - Uses /v1/chat/completions (LM Studio expects this)
    - Forces response_format to "text" when caller asks for json_object
      (LM Studio rejects json_object unless using json_schema)
    - Budgets prompt size for 4k-context local models
    """

    def __init__(self) -> None:
        self.base_url = (getattr(settings, "OPENAI_BASE_URL", "") or "").rstrip("/")
        self.api_key = getattr(settings, "OPENAI_API_KEY", None)
        self.model = (getattr(settings, "OPENAI_MODEL", "") or "").strip()
        self.llm_enabled_flag = getattr(settings, "LLM_ENABLED", None)

        self.max_input_chars = int(getattr(settings, "LLM_MAX_INPUT_CHARS", 12_000))
        self.read_timeout_s = float(getattr(settings, "LLM_READ_TIMEOUT_S", 600.0))

    def enabled(self) -> bool:
        if self.llm_enabled_flag is not None and not bool(self.llm_enabled_flag):
            return False
        return bool(self.base_url) and bool(self.model)

    def _endpoint(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    async def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        response_format: Optional[dict[str, Any]] = None,
        extra_payload: Optional[dict[str, Any]] = None,
        force_text_response_format: bool = False,
        max_input_chars: Optional[int] = None,
    ) -> str:
        if not self.enabled():
            raise LLMError("LLM not enabled. Set LLM_ENABLED=true and OPENAI_BASE_URL + OPENAI_MODEL.")

        url = self._endpoint()

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        budget = self.max_input_chars if max_input_chars is None else int(max_input_chars)
        safe_user = _truncate_chars(user, max_chars=budget)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": safe_user},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        # LM Studio: response_format.type must be 'json_schema' or 'text'
        if force_text_response_format:
            payload["response_format"] = {"type": "text"}
        elif response_format is not None:
            rf_type = str(response_format.get("type", "")).strip().lower()
            if rf_type == "json_object":
                payload["response_format"] = {"type": "text"}
            else:
                payload["response_format"] = response_format

        if extra_payload:
            payload.update(extra_payload)

        timeout = httpx.Timeout(
            connect=10.0,
            read=self.read_timeout_s,
            write=60.0,
            pool=10.0,
        )

        async def _post(p: dict[str, Any]) -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, headers=headers, content=json.dumps(p))
                r.raise_for_status()
                return r.json()

        try:
            data = await _post(payload)
        except httpx.HTTPStatusError as e:
            body = ""
            status = e.response.status_code
            try:
                body = e.response.text
            except Exception:
                body = ""

            if "response_format" in payload and status in (400, 404, 422):
                payload.pop("response_format", None)
                try:
                    data = await _post(payload)
                except Exception as e2:
                    raise LLMError(f"LLM HTTP error {status} from {url}. Response body: {body[:800]}") from e2
            else:
                raise LLMError(f"LLM HTTP error {status} from {url}. Response body: {body[:800]}") from e
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
        response_format: Optional[dict[str, Any]] = None,
        max_input_chars: Optional[int] = None,
    ) -> dict[str, Any]:
        raw = await self.chat(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            force_text_response_format=True,  # critical for LM Studio
            max_input_chars=max_input_chars,
        )

        # strict attempt first
        try:
            obj = json.loads(_strip_code_fences(raw))
            if isinstance(obj, dict):
                return obj
            raise ValueError("Top-level JSON is not an object")
        except Exception as e:
            # helpful debug breadcrumb (doesn't spam too hard)
            logger.warning("LLM JSON parse failed, attempting lenient repair: %s", str(e))
            obj2 = _extract_json_object_lenient(raw)
            if not isinstance(obj2, dict):
                raise ValueError("Top-level JSON is not an object after lenient parse")
            return obj2


async def chat_completion_json(system: str, user: str) -> dict[str, Any]:
    llm = LLMClient()
    if not llm.enabled():
        raise LLMError("LLM is disabled (LLM_ENABLED=false or missing OPENAI_BASE_URL/OPENAI_MODEL).")
    return await llm.chat_json(
        system=system,
        user=user,
        temperature=0.2,
        max_tokens=2000,
        response_format={"type": "json_object"},  # safe: will be forced to text
    )
