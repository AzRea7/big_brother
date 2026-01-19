# backend/app/services/code_signals.py
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import RepoFile

# -------------------------
# Marker signals (cheap heuristics)
# -------------------------

# Classic marker detection (counts files-with)
_MARKERS = {
    "todo": re.compile(r"\bTODO\b", re.IGNORECASE),
    "fixme": re.compile(r"\bFIXME\b", re.IGNORECASE),
    "hack": re.compile(r"\bHACK\b", re.IGNORECASE),
    "xxx": re.compile(r"\bXXX\b", re.IGNORECASE),
    "bug": re.compile(r"\bBUG\b", re.IGNORECASE),
    "note": re.compile(r"\bNOTE\b", re.IGNORECASE),
    # catches "...", "…", "dotdotdot" style unfinished code
    "dotdotdot": re.compile(r"(\.\.\.|…)", re.IGNORECASE),
}

# -------------------------
# “Level 2” production/ops signals (also cheap heuristics)
# These are NOT “bugs”; they’re “likely missing best practice”.
# We count files-with to avoid skew from huge files.
# -------------------------

_OPS_SIGNALS = {
    # auth / security-ish
    "auth": re.compile(r"\b(API[-_ ]?KEY|Authorization|require_api_key|oauth|jwt|bearer)\b", re.IGNORECASE),
    "cors": re.compile(r"\bCORSMiddleware\b|\bcors\b", re.IGNORECASE),
    "csrf": re.compile(r"\bcsrf\b", re.IGNORECASE),

    # reliability
    "timeout": re.compile(r"\btimeout\s*=\s*|\bREAD_TIMEOUT\b|\bCONNECT_TIMEOUT\b", re.IGNORECASE),
    "retry": re.compile(r"\bretry\b|\b(backoff|tenacity)\b", re.IGNORECASE),
    "rate_limit": re.compile(r"\brate[-_ ]?limit\b|\bSlowAPI\b|\blimiter\b", re.IGNORECASE),

    # correctness / input safety
    "validation": re.compile(r"\bpydantic\b|\bBaseModel\b|\bvalidate\b|\bschema\b", re.IGNORECASE),

    # observability
    "logging": re.compile(r"\blogging\b|\bstructlog\b|\bloguru\b|\bLOG_JSON\b", re.IGNORECASE),
    "metrics": re.compile(r"\b/prometheus\b|\bprometheus\b|\bmetrics\b", re.IGNORECASE),

    # data / persistence
    "db": re.compile(r"\bsqlalchemy\b|\bAsyncSession\b|\bsessionmaker\b|\bAlembic\b|\bmigration\b", re.IGNORECASE),
    "nplus1": re.compile(r"\bn\+1\b|\bselectinload\b|\bjoinedload\b", re.IGNORECASE),

    # delivery
    "tests": re.compile(r"\bpytest\b|\btest_\w+\b", re.IGNORECASE),
    "ci": re.compile(r"\bgithub/workflows\b|\bactions/checkout\b|\bpytest -q\b|\bruff\b|\bmypy\b", re.IGNORECASE),
    "docker": re.compile(r"\bDockerfile\b|\bdocker-compose\b", re.IGNORECASE),

    # configuration & secrets hygiene
    "config": re.compile(r"\bENV\b|\bDB_URL\b|\bsettings\b|\bBaseSettings\b", re.IGNORECASE),
    "secrets": re.compile(r"\bOPENAI_API_KEY\b|\bGITHUB_TOKEN\b|\bAPI_KEY\b|\bSECRET\b", re.IGNORECASE),
}


def count_markers_in_text(text: str) -> dict[str, bool]:
    """
    Returns booleans of whether each marker appears in the text.
    We count "files_with_X" later using these booleans.
    """
    out: dict[str, bool] = {}
    for k, rx in _MARKERS.items():
        out[k] = bool(rx.search(text or ""))
    return out


def _count_ops_signals_in_text(text: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for k, rx in _OPS_SIGNALS.items():
        out[k] = bool(rx.search(text or ""))
    return out


# -------------------------
# Public API used by routes
# -------------------------

def compute_signal_counts(db: Session, snapshot_id: int) -> dict[str, int]:
    """
    Backwards-compatible DB-based signal counter.

    Returns only the signals dict (no wrapper).
    """
    files = db.scalars(select(RepoFile).where(RepoFile.snapshot_id == snapshot_id)).all()
    return compute_signal_counts_for_files(files)


def compute_signal_counts_for_files(files: list[RepoFile]) -> dict[str, int]:
    """
    Produces "files_with_X" counts for:
      - marker signals (TODO/FIXME/etc)
      - ops signals (auth/timeout/retry/etc)

    Returned dict shape:
      {
        "total_files": <int>,
        "files_with_todo": <int>,
        ...
      }

    NOTE: some callers want just the flattened dict; others want the full wrapper.
    """
    total = len(files)

    # We'll return flattened keys: files_with_<signal>
    counts: dict[str, int] = {"total_files": total}

    # initialize
    for k in _MARKERS.keys():
        counts[f"files_with_{k}"] = 0
    for k in _OPS_SIGNALS.keys():
        counts[f"files_with_{k}"] = 0

    for f in files:
        # only text content is scannable
        text = (getattr(f, "content_text", None) or getattr(f, "content", None) or "")
        if not isinstance(text, str) or not text.strip():
            continue

        marker_hits = count_markers_in_text(text)
        for k, hit in marker_hits.items():
            if hit:
                counts[f"files_with_{k}"] += 1

        ops_hits = _count_ops_signals_in_text(text)
        for k, hit in ops_hits.items():
            if hit:
                counts[f"files_with_{k}"] += 1

    return counts


def compute_signal_counts_full(files: list[RepoFile]) -> dict[str, Any]:
    """
    This is the function your routes/repo.py expects.

    It returns:
      {
        "total_files": <int>,
        "signals": {
           "todo": <files_with_todo>,
           ...
        }
      }

    Why wrapper form?
    - Nice for UI and debug endpoints
    - Stable contract for future expansion (per-file stats later)
    """
    flat = compute_signal_counts_for_files(files)
    total = int(flat.get("total_files", 0))

    signals: dict[str, int] = {}

    # markers
    for k in _MARKERS.keys():
        signals[k] = int(flat.get(f"files_with_{k}", 0))

    # ops signals
    for k in _OPS_SIGNALS.keys():
        signals[k] = int(flat.get(f"files_with_{k}", 0))

    return {"total_files": total, "signals": signals}
