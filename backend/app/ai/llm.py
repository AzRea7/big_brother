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

# Unquoted key like: { foo: 1 } or , bar : 2  ->  { "foo": 1 }, "bar": 2
_UNQUOTED_KEY_RE = re.compile(r'(^|[{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)')

# If the model outputs python-ish dicts with single quotes, this helps.
# We apply this only in lenient repair mode, not as a first pass.
_SINGLE_QUOTED_STR_RE = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'")  # '...'(with escapes)


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

    def starts_value_at(i: int) -> bool:
        if i >= len(s):
            return False
        ch = s[i]
        if ch in "{[\"":
            return True
        if ch == "-" or ch.isdigit():
            return True
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
            ch2 = s[j]
            if esc2:
                esc2 = False
            elif ch2 == "\\":
                esc2 = True
            elif ch2 == '"':
                break
            j += 1
        if j >= len(s) or s[j] != '"':
            return False
        k = j + 1
        while k < len(s) and s[k].isspace():
            k += 1
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
            i += 1
            continue

        # Outside string: maybe insert comma if we are after_value and next token begins a new value/key
        t = top()
        if t is not None:
            kind, state = t
            if state == "after_value" and not ch.isspace():
                if kind == "obj":
                    if ch not in {",", "}"} and is_key_at(i):
                        out.append(",")
                        set_state("obj", "need_key")
                elif kind == "arr":
                    if ch not in {",", "]"} and starts_value_at(i):
                        out.append(",")
                        set_state("arr", "need_value")

        if ch == '"':
            out.append(ch)
            in_str = True
            esc = False
            t2 = top()
            if t2 is not None and t2[0] == "obj" and t2[1] == "need_key":
                set_state("obj", "need_colon")
            if t2 is not None and t2[0] == "arr" and t2[1] == "need_value":
                set_state("arr", "after_value")
            i += 1
            continue

        if ch == "{":
            out.append(ch)
            stack.append(("obj", "need_key"))
            if len(stack) >= 2 and stack[-2][0] == "arr" and stack[-2][1] == "need_value":
                stack[-2] = ("arr", "after_value")
            if len(stack) >= 2 and stack[-2][0] == "obj" and stack[-2][1] == "need_value":
                stack[-2] = ("obj", "after_value")
            i += 1
            continue

        if ch == "[":
            out.append(ch)
            stack.append(("arr", "need_value"))
            if len(stack) >= 2 and stack[-2][0] == "arr" and stack[-2][1] == "need_value":
                stack[-2] = ("arr", "after_value")
            if len(stack) >= 2 and stack[-2][0] == "obj" and stack[-2][1] == "need_value":
                stack[-2] = ("obj", "after_value")
            i += 1
            continue

        if ch == ":":
            out.append(ch)
            if stack and stack[-1][0] == "obj" and stack[-1][1] == "need_colon":
                set_state("obj", "need_value")
            i += 1
            continue

        if ch == ",":
            out.append(ch)
            if stack:
                kind, _state = stack[-1]
                stack[-1] = (kind, "need_key" if kind == "obj" else "need_value")
            i += 1
            continue

        if ch == "}":
            out.append(ch)
            if stack and stack[-1][0] == "obj":
                stack.pop()
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
                if starts_value_at(i) and ch not in "{[\"":
                    stack[-1] = (kind, "after_value")

        out.append(ch)
        i += 1

    return "".join(out)


def _quote_unquoted_keys_outside_strings(s: str) -> str:
    """
    Convert { foo: 1, bar: 2 } into { "foo": 1, "bar": 2 }.

    IMPORTANT: we only do this outside of double-quoted JSON strings.
    """
    s = s or ""
    out: list[str] = []

    in_str = False
    esc = False
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
            i += 1
            continue

        if ch == '"':
            in_str = True
            esc = False
            out.append(ch)
            i += 1
            continue

        # Try match at this position for unquoted key.
        # We need to run the regex on the remaining substring but preserve prefix group.
        m = _UNQUOTED_KEY_RE.match(s, i)
        if m:
            # m.group(1) includes "{" or "," plus whitespace
            out.append(m.group(1))
            out.append(f'"{m.group(2)}"')
            out.append(m.group(3))
            i = m.end()
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _replace_single_quoted_strings_lenient(s: str) -> str:
    """
    Convert single-quoted strings to double-quoted strings:
      {'a': 'b'} -> {"a": "b"}

    This is a heuristic. We apply it late, only if other repairs failed.
    """
    s = s or ""

    # We must avoid touching apostrophes inside already-double-quoted strings.
    out: list[str] = []
    in_dq = False
    esc = False
    i = 0

    while i < len(s):
        ch = s[i]

        if in_dq:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_dq = False
            i += 1
            continue

        if ch == '"':
            in_dq = True
            esc = False
            out.append(ch)
            i += 1
            continue

        if ch == "'":
            # parse a single-quoted string token
            j = i + 1
            esc2 = False
            buf: list[str] = []
            while j < len(s):
                ch2 = s[j]
                if esc2:
                    buf.append(ch2)
                    esc2 = False
                elif ch2 == "\\":
                    esc2 = True
                elif ch2 == "'":
                    break
                else:
                    buf.append(ch2)
                j += 1

            if j < len(s) and s[j] == "'":
                # emit as JSON double-quoted string with proper escaping
                inner = "".join(buf)
                inner = inner.replace("\\", "\\\\").replace('"', '\\"')
                out.append('"')
                out.append(inner)
                out.append('"')
                i = j + 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def _insert_comma_at_error_pos(s: str, pos: int) -> str:
    """
    If json.loads says 'Expecting , delimiter' at position pos,
    we can often fix by inserting a comma at/near that position.
    """
    if not s:
        return s
    pos = max(0, min(int(pos), len(s)))

    j = pos
    while j < len(s) and s[j].isspace():
        j += 1

    if j < len(s) and s[j] == ",":
        return s

    return s[:j] + "," + s[j:]


def _loads_with_repairs(c: str) -> Any:
    """
    Parse JSON with multiple increasingly aggressive repair passes.

    This is designed for local / small models that occasionally emit
    almost-JSON (missing commas, trailing commas, truncated output, python-ish dicts).
    """
    # Attempt 1: raw
    try:
        return json.loads(c)
    except json.JSONDecodeError:
        pass

    # Attempt 2: truncation repair + trailing comma cleanup
    c2 = _repair_truncated_json(c)
    c2 = _TRAILING_COMMA_RE.sub(r"\1", c2)
    try:
        return json.loads(c2)
    except json.JSONDecodeError:
        pass

    # Attempt 3: contextual commas
    c3 = _insert_missing_commas_contextual(c2)
    c3 = _TRAILING_COMMA_RE.sub(r"\1", c3)
    try:
        return json.loads(c3)
    except json.JSONDecodeError:
        pass

    # Attempt 4: quote unquoted keys
    c4 = _quote_unquoted_keys_outside_strings(c3)
    c4 = _TRAILING_COMMA_RE.sub(r"\1", c4)
    try:
        return json.loads(c4)
    except json.JSONDecodeError as e4:
        last = e4

    # Attempt 5: single-quote repair (python-ish)
    c5 = _replace_single_quoted_strings_lenient(c4)
    c5 = _TRAILING_COMMA_RE.sub(r"\1", c5)
    try:
        return json.loads(c5)
    except json.JSONDecodeError as e5:
        last = e5

    # Attempt 6: iterative insert-comma at decoder position (only for missing comma)
    c6 = c5
    for _ in range(12):
        try:
            return json.loads(c6)
        except json.JSONDecodeError as e:
            last = e
            if "Expecting ',' delimiter" not in str(e):
                break
            nxt = _insert_comma_at_error_pos(c6, e.pos)
            if nxt == c6:
                break
            c6 = nxt

    raise last


def _extract_json_object_lenient(raw: str) -> Any:
    """
    Robust JSON extraction for LLM outputs.
    """
    s = _strip_code_fences(raw)

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


def _best_json_candidate_from_raw(raw: str) -> str:
    """
    For the self-heal pass: feed the model only the most-likely JSON-ish chunk,
    not the whole rambling completion (which wastes tokens and increases truncation).
    """
    s = _strip_code_fences(raw)

    cand = _extract_balanced_json_prefix(s)
    if cand is not None:
        return cand

    # fallback to widest object span
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        return s[first : last + 1]

    # fallback array span
    first = s.find("[")
    last = s.rfind("]")
    if first != -1 and last != -1 and last > first:
        return s[first : last + 1]

    return s


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
        self.enable_self_heal = bool(getattr(settings, "LLM_SELF_HEAL_JSON", True))

        # If your model keeps truncating the fixer response, bump this.
        self.self_heal_max_tokens = int(getattr(settings, "LLM_SELF_HEAL_MAX_TOKENS", 2200))

        # Hard safety cap so we don't explode the 4k context with the fixer prompt.
        self.self_heal_max_input_chars = int(getattr(settings, "LLM_SELF_HEAL_MAX_INPUT_CHARS", 18_000))

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

    async def _self_heal_json(self, raw: str) -> dict[str, Any]:
        """
        If the model produced almost-JSON, ask it to output strictly valid JSON.

        Key fixes vs your current behavior:
        - Feed the model only the JSON-ish candidate (not the whole completion).
        - Allow more tokens so the fixer doesn't truncate mid-object.
        - Still parse with our lenient extractor as a final safety net.
        """
        candidate = _best_json_candidate_from_raw(raw)
        candidate = _truncate_chars(candidate, self.self_heal_max_input_chars)

        fixer_system = "You are a strict JSON formatter. Output ONLY valid JSON. No markdown, no code fences, no commentary."
        fixer_user = (
            "Convert the following into valid JSON.\n"
            "- Output must be a single JSON object.\n"
            "- Preserve the same structure and keys.\n"
            "- If input is truncated, return the longest valid JSON object you can complete.\n"
            "- Do not add any extra text.\n\n"
            f"RAW:\n{candidate}"
        )

        fixed = await self.chat(
            system=fixer_system,
            user=fixer_user,
            temperature=0.0,
            max_tokens=self.self_heal_max_tokens,
            response_format={"type": "json_object"},
            force_text_response_format=True,
            max_input_chars=self.self_heal_max_input_chars,
        )

        obj = _extract_json_object_lenient(fixed)
        if not isinstance(obj, dict):
            raise ValueError("Self-heal JSON result is not a JSON object")
        return obj

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
            logger.warning("LLM JSON parse failed, attempting lenient repair: %s", str(e))

        # lenient local repair
        try:
            obj2 = _extract_json_object_lenient(raw)
            if not isinstance(obj2, dict):
                raise ValueError("Top-level JSON is not an object after lenient parse")
            return obj2
        except Exception as e2:
            logger.warning("Lenient JSON repair failed: %s", str(e2))

        # last resort: ask the model to reformat its own output
        if self.enable_self_heal:
            logger.warning("Attempting JSON self-heal pass via LLM")
            return await self._self_heal_json(raw)

        raise ValueError("Unable to parse JSON from LLM output (strict + lenient failed).")


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
