# backend/app/services/patch_generator.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..ai.llm import LLMClient
from ..config import settings
from ..models import RepoFinding
from .repo_chunks import search_chunks, load_chunk_text


class PatchGenerationError(RuntimeError):
    pass


_DIFF_GIT_RE = re.compile(r"^diff --git a/.+ b/.+\s*$", re.MULTILINE)


def _allowlist_dirs() -> list[str]:
    # New env-var names (PATCH_*) with fallback to existing PR_* settings.
    allow = getattr(settings, "PATCH_ALLOWLIST", None)
    if allow:
        if isinstance(allow, str):
            return [x.strip() for x in allow.split(",") if x.strip()]
        return list(allow)
    return list(getattr(settings, "PR_ALLOWLIST_DIRS", ["backend/"]))


def _denylist_patterns() -> list[str]:
    deny = getattr(settings, "PATCH_DENYLIST", None)
    if deny:
        if isinstance(deny, str):
            return [x.strip() for x in deny.split(",") if x.strip()]
        return list(deny)

    # sane defaults if user didn’t set anything
    return [
        r"(^|/)\.env$",
        r"(^|/)secrets?(\.|/)",
        r"(^|/)id_rsa(\.|$)",
        r"(^|/)node_modules(/|$)",
        r"(^|/)\.git(/|$)",
    ]


def _max_files() -> int:
    v = getattr(settings, "PATCH_MAX_FILES", None)
    if v is not None:
        return int(v)
    return int(getattr(settings, "PR_MAX_FILES_CHANGED", 8))


def _max_patch_chars() -> int:
    v = getattr(settings, "PATCH_MAX_CHARS", None)
    if v is not None:
        return int(v)
    # Default: large enough for real diffs, small enough to avoid insane outputs
    return 200_000


def _llm_max_tokens() -> int:
    # reuse your existing LLM knobs if present
    return int(getattr(settings, "PATCH_LLM_MAX_TOKENS", 2600) or 2600)


def _llm_temperature() -> float:
    return float(getattr(settings, "PATCH_LLM_TEMPERATURE", 0.1) or 0.1)


def _rag_top_k() -> int:
    return int(getattr(settings, "PATCH_RAG_TOP_K", 6) or 6)


def _findings_rag_max_chars() -> int:
    # cap the code context sent to the model
    return int(getattr(settings, "FINDING_RAG_MAX_CHARS", 6000) or 6000)


def _path_allowed(path: str, allowlist: list[str], deny_patterns: list[str]) -> bool:
    p = (path or "").strip()
    if not p:
        return False

    if not any(p.startswith(a) for a in allowlist):
        return False

    for pat in deny_patterns:
        if re.search(pat, p):
            return False

    return True


def _build_system_prompt() -> str:
    return (
        "You are a senior software engineer.\n"
        "You must output ONLY a valid unified diff (git-style patch).\n"
        "Absolutely no markdown, no code fences, no commentary.\n\n"
        "Hard requirements:\n"
        "- Output must contain one or more 'diff --git a/<path> b/<path>' sections.\n"
        "- Use correct '--- a/<path>' and '+++ b/<path>' headers and @@ hunks.\n"
        "- Only change the minimal lines needed to satisfy the objective.\n"
        "- Do not reformat unrelated code.\n"
        "- Do not change files outside the allowlist.\n"
        "- Do not touch secrets, env files, node_modules, or .git.\n"
        "- Keep changes small and focused.\n"
    )


def _format_constraints_for_prompt() -> str:
    allow = _allowlist_dirs()
    deny = _denylist_patterns()

    return (
        "Constraints:\n"
        f"- ALLOWLIST_DIRS: {allow}\n"
        f"- DENYLIST_REGEX: {deny}\n"
        f"- MAX_FILES_CHANGED: {_max_files()}\n"
        f"- MAX_PATCH_CHARS: {_max_patch_chars()}\n"
    )


def _format_context_for_prompt(snippets: list[dict[str, Any]]) -> str:
    # snippets items: {path, start_line, end_line, text}
    if not snippets:
        return "Context snippets: (none)\n"

    out: list[str] = ["Context snippets (read-only):"]
    budget = _findings_rag_max_chars()
    used = 0

    for s in snippets:
        path = s.get("path") or ""
        start = int(s.get("start_line") or 1)
        end = int(s.get("end_line") or start)
        text = (s.get("text") or "").strip()
        if not text:
            continue

        block = f"\n--- {path}:{start}-{end} ---\n{text}\n"
        if used + len(block) > budget:
            remaining = max(0, budget - used)
            if remaining > 200:
                out.append(block[:remaining] + "\n…(truncated)…\n")
            break

        out.append(block)
        used += len(block)

    return "\n".join(out) + "\n"


async def _gather_snippets_for_finding(db: Session, snapshot_id: int, finding: RepoFinding) -> list[dict[str, Any]]:
    allow = _allowlist_dirs()
    deny = _denylist_patterns()
    top_k = _rag_top_k()

    # Query seeds: bias retrieval toward the actual finding, not generic “repo hygiene”.
    q_parts = []
    if finding.title:
        q_parts.append(finding.title)
    if finding.category:
        q_parts.append(str(finding.category))
    if finding.recommendation:
        q_parts.append(str(finding.recommendation))
    if finding.path:
        q_parts.append(f"file:{finding.path}")
    query = " | ".join([p for p in q_parts if p and str(p).strip()])[:500]

    mode_used, hits = await search_chunks(
        db=db,
        snapshot_id=snapshot_id,
        query=query,
        top_k=top_k,
        mode="auto",
        path_contains=finding.path if (finding.path or "").strip() else None,
    )

    snippets: list[dict[str, Any]] = []
    for h in (hits or []):
        # h might be dict or ChunkHit dataclass
        hid = h.get("id") if isinstance(h, dict) else getattr(h, "id", None)
        path = h.get("path") if isinstance(h, dict) else getattr(h, "path", "")
        start_line = h.get("start_line") if isinstance(h, dict) else getattr(h, "start_line", 1)
        end_line = h.get("end_line") if isinstance(h, dict) else getattr(h, "end_line", start_line)

        if not _path_allowed(str(path), allow, deny):
            continue

        loaded = load_chunk_text(db=db, chunk_id=int(hid))
        text = (loaded.get("chunk_text") or "").strip() if isinstance(loaded, dict) else str(loaded)

        if text:
            snippets.append(
                {
                    "path": str(path),
                    "start_line": int(start_line or 1),
                    "end_line": int(end_line or int(start_line or 1)),
                    "text": text,
                    "mode_used": mode_used,
                }
            )

    return snippets


def _basic_sanity_check(patch_text: str) -> None:
    if not patch_text or not patch_text.strip():
        raise PatchGenerationError("LLM returned empty patch text.")
    if len(patch_text) > _max_patch_chars():
        raise PatchGenerationError(f"LLM patch too large: {len(patch_text)} > {_max_patch_chars()}")
    if not _DIFF_GIT_RE.search(patch_text):
        raise PatchGenerationError("LLM output does not look like a unified diff (missing 'diff --git').")


async def generate_unified_diff(
    db: Session,
    *,
    snapshot_id: int,
    finding_id: Optional[int] = None,
    objective: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate a unified diff automatically using the LLM.

    You can drive it two ways:
    - finding_id: generate a patch that addresses a specific RepoFinding
    - objective: free-form instruction (still constrained by allow/deny lists)
    """
    llm = LLMClient()
    if not llm.enabled():
        raise PatchGenerationError("LLM not enabled. Set LLM_ENABLED=true plus LLM_BASE_URL and LLM_MODEL.")

    finding: Optional[RepoFinding] = None
    if finding_id is not None:
        finding = db.query(RepoFinding).filter(RepoFinding.id == int(finding_id)).first()
        if not finding:
            raise PatchGenerationError(f"Unknown finding_id={finding_id}")

    if not (finding or (objective and objective.strip())):
        raise PatchGenerationError("Provide either finding_id or objective.")

    allow = _allowlist_dirs()
    deny = _denylist_patterns()

    snippets: list[dict[str, Any]] = []
    if finding is not None:
        snippets = await _gather_snippets_for_finding(db, snapshot_id, finding)

    # Build the user prompt
    finding_block = ""
    if finding is not None:
        finding_block = (
            "Target finding:\n"
            f"- id: {finding.id}\n"
            f"- category: {finding.category}\n"
            f"- severity: {finding.severity}\n"
            f"- title: {finding.title}\n"
            f"- path: {finding.path}\n"
            f"- line: {finding.line}\n"
            f"- evidence: {finding.evidence}\n"
            f"- recommendation: {finding.recommendation}\n"
            f"- acceptance: {getattr(finding, 'acceptance', None)}\n"
        )

    objective_text = (objective or "").strip()
    if not objective_text and finding is not None:
        # If user didn’t provide a free objective, convert the finding into one.
        objective_text = f"Implement the recommendation for the finding titled: {finding.title}"

    user_prompt = (
        f"{_format_constraints_for_prompt()}\n"
        f"{finding_block}\n"
        "Objective:\n"
        f"{objective_text}\n\n"
        "Rules:\n"
        f"- Only edit files under: {allow}\n"
        f"- Do NOT touch files matching deny patterns: {deny}\n"
        f"- Change at most {_max_files()} files.\n"
        "- Prefer small, targeted fixes.\n"
        "- If the fix requires new code, keep it minimal and well-named.\n"
        "- If unsure, add a TODO comment explaining what is missing rather than guessing.\n\n"
        f"{_format_context_for_prompt(snippets)}\n"
        "Now output the unified diff ONLY.\n"
    )

    patch_text = await llm.chat(
        system=_build_system_prompt(),
        user=user_prompt,
        temperature=_llm_temperature(),
        max_tokens=_llm_max_tokens(),
        response_format={"type": "text"},
        force_text_response_format=True,
    )

    patch_text = (patch_text or "").strip()
    _basic_sanity_check(patch_text)

    # quick local guard: ensure paths referenced appear allowlisted (best-effort)
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            p = line.replace("+++ b/", "").strip()
            if not _path_allowed(p, allow, deny):
                raise PatchGenerationError(f"Generated patch touches disallowed path: {p}")

    return {
        "snapshot_id": snapshot_id,
        "finding_id": finding_id,
        "objective": objective_text,
        "patch_text": patch_text,
        "snippets_used": [
            {"path": s["path"], "start_line": s["start_line"], "end_line": s["end_line"]}
            for s in snippets
        ],
    }
