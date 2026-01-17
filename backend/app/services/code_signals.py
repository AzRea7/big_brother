# backend/app/services/code_signals.py
from __future__ import annotations

import re
from typing import Any


_MARKERS = {
    "todo": re.compile(r"\bTODO\b", re.IGNORECASE),
    "fixme": re.compile(r"\bFIXME\b", re.IGNORECASE),
    "hack": re.compile(r"\bHACK\b", re.IGNORECASE),
    "xxx": re.compile(r"\bXXX\b", re.IGNORECASE),
    "bug": re.compile(r"\bBUG\b", re.IGNORECASE),
    "note": re.compile(r"\bNOTE\b", re.IGNORECASE),
    "dotdotdot": re.compile(r"\.\.\.", re.IGNORECASE),
}


def count_markers_in_text(text: str) -> dict[str, int]:
    """
    Counts signal markers in a file's content.

    NOTE:
    - We still count NOTE, but repo_taskgen should treat NOTE as low-priority noise
      unless you explicitly allow it.
    """
    out: dict[str, int] = {}
    for k, rx in _MARKERS.items():
        out[f"{k}_count"] = len(rx.findall(text or ""))
    return out


def pick_best_evidence_snippets(text: str, max_snippets: int = 3, radius: int = 120) -> list[str]:
    """
    Extract a few short snippets around the highest-value markers. These snippets go to the LLM.
    """
    if not text:
        return []

    hits: list[tuple[int, str]] = []
    for k, rx in _MARKERS.items():
        for m in rx.finditer(text):
            hits.append((m.start(), k))

    # Prioritize high-value markers, de-prioritize NOTE
    weight = {"fixme": 5, "todo": 4, "bug": 4, "hack": 3, "xxx": 3, "dotdotdot": 2, "note": 1}
    hits.sort(key=lambda t: weight.get(t[1], 0), reverse=True)

    snippets: list[str] = []
    used_ranges: list[tuple[int, int]] = []
    for pos, k in hits:
        start = max(0, pos - radius)
        end = min(len(text), pos + radius)

        # Avoid overlapping snippets
        if any(not (end < a or start > b) for a, b in used_ranges):
            continue

        chunk = text[start:end].replace("\r\n", "\n").replace("\r", "\n")
        chunk = chunk.strip()

        # Hard cap snippet length
        if len(chunk) > 500:
            chunk = chunk[:500] + "…"

        snippets.append(f"[{k.upper()}] {chunk}")
        used_ranges.append((start, end))

        if len(snippets) >= max_snippets:
            break

    return snippets


def looks_like_impl_stub(text: str) -> bool:
    if not text:
        return False
    needles = ("raise NotImplementedError", "IMPLEMENT", "stub", "pass  #", "pass #")
    return any(n in text for n in needles)
