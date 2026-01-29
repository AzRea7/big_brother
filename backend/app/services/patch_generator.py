# backend/app/services/patch_generator.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoFinding, RepoSnapshot
from ..ai.llm import LLMClient  # real implementation lives in backend/app/ai/llm.py
from .repo_chunks import search_chunks  # async in your app


@dataclass(frozen=True)
class PatchGenerationError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$", re.MULTILINE)
_PLUSPLUS_RE = re.compile(r"^\+\+\+ b/(.+?)\s*$", re.MULTILINE)


def _llm_temperature() -> float:
    return float(getattr(settings, "PATCH_LLM_TEMPERATURE", 0.1) or 0.1)


def _llm_max_tokens() -> int:
    return int(getattr(settings, "PATCH_LLM_MAX_TOKENS", 2600) or 2600)


def _max_files() -> int:
    # generator-side cap (validator will enforce PR_MAX_FILES_CHANGED too)
    return int(getattr(settings, "PATCH_MAX_FILES_CHANGED", 3) or 3)


def _forbid_new_files() -> bool:
    return bool(getattr(settings, "PATCH_FORBID_NEW_FILES", True))


def _require_finding_path_only() -> bool:
    # strongest safety default: only allow editing the file where the finding is
    return bool(getattr(settings, "PATCH_REQUIRE_FINDING_PATH_ONLY", True))


def _allowlist_dirs() -> list[str]:
    dirs = getattr(settings, "PATCH_ALLOWLIST_DIRS", None)
    if isinstance(dirs, list) and dirs:
        return [str(d).strip() for d in dirs if str(d).strip()]
    return []


def _denylist_patterns() -> list[str]:
    pats = getattr(settings, "PATCH_DENYLIST_PATTERNS", None)
    if isinstance(pats, list) and pats:
        return [str(p).strip() for p in pats if str(p).strip()]
    return []


def _norm_path(p: str) -> str:
    return (p or "").strip().replace("\\", "/")


def _path_allowed(path: str, allow_dirs: list[str], deny_patterns: list[str]) -> bool:
    p_norm = _norm_path(path)
    if not p_norm:
        return False

    # traversal / absolute path guards
    if p_norm.startswith("/"):
        return False
    if ".." in p_norm.split("/"):
        return False

    # allowlist
    if allow_dirs:
        allow_norm = [_norm_path(a) for a in allow_dirs]
        if not any(p_norm.startswith(a) for a in allow_norm):
            return False

    # denylist regex patterns
    for pat in deny_patterns:
        try:
            if re.search(pat, p_norm):
                return False
        except re.error:
            # misconfigured denylist -> fail closed
            return False

    return True


def _extract_touched_paths(patch_text: str) -> list[str]:
    s = (patch_text or "").strip()
    out: list[str] = []

    for m in _DIFF_GIT_RE.finditer(s):
        a_path = _norm_path(m.group(1))
        b_path = _norm_path(m.group(2))
        if a_path and a_path not in out:
            out.append(a_path)
        if b_path and b_path not in out:
            out.append(b_path)

    # secondary fallback
    for m in _PLUSPLUS_RE.finditer(s):
        p = _norm_path(m.group(1))
        if p and p not in out:
            out.append(p)

    return out


def _reject_new_files(patch_text: str) -> Optional[str]:
    if not _forbid_new_files():
        return None
    s = patch_text or ""
    if "--- /dev/null" in s or "+++ /dev/null" in s:
        return "New/deleted files are forbidden in patch generation."
    return None


def _basic_sanity_check(patch_text: str) -> None:
    s = (patch_text or "").strip()
    if not s:
        raise PatchGenerationError("Empty patch_text from model.")
    if "diff --git " not in s:
        raise PatchGenerationError("Model did not output a unified diff (missing 'diff --git').")


def _build_system_prompt() -> str:
    # Keep this brutally strict. The route enforces allow/deny again.
    return (
        "You are a senior software engineer.\n"
        "Output ONLY a unified diff.\n"
        "No explanations.\n"
        "No markdown fences.\n"
        "Do NOT add or delete files.\n"
        "Do NOT modify CI/workflows, Docker/infra, secrets, keys, credentials.\n"
        "Keep changes minimal and tightly scoped.\n"
    )


def _format_constraints_for_prompt(
    *,
    allow_dirs: list[str],
    deny_patterns: list[str],
    target_paths: list[str],
) -> str:
    return (
        "Constraints:\n"
        f"- ALLOWLIST_DIRS: {allow_dirs}\n"
        f"- DENYLIST_PATTERNS (regex): {deny_patterns}\n"
        f"- TARGET_PATHS (only these may change): {target_paths}\n"
        f"- MAX_FILES_CHANGED: {_max_files()}\n"
        f"- FORBID_NEW_FILES: {_forbid_new_files()}\n"
        "\n"
        "Hard rules:\n"
        "- Output unified diff ONLY.\n"
        "- Do NOT touch any paths outside TARGET_PATHS.\n"
        "- Do NOT create or delete files.\n"
        "- If you cannot implement the fix without violating rules, output an EMPTY diff (no changes).\n"
    )


def _format_context_for_prompt(snippets: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for s in snippets:
        blocks.append(
            "Snippet:\n"
            f"- path: {s.get('path')}\n"
            f"- start_line: {s.get('start_line')}\n"
            f"- end_line: {s.get('end_line')}\n"
            "-----\n"
            f"{s.get('text','')}\n"
            "-----\n"
        )
    return "\n".join(blocks) if blocks else "Snippets: (none)\n"


def _build_query_from_finding(finding: RepoFinding) -> str:
    parts = [
        str(getattr(finding, "title", "") or "").strip(),
        str(getattr(finding, "evidence", "") or "").strip(),
        str(getattr(finding, "recommendation", "") or "").strip(),
    ]
    q = " ".join([p for p in parts if p]).strip()
    return q or f"Fix finding {getattr(finding, 'id', '')}".strip()


async def _search_snippets_for_file(
    *,
    db: Session,
    snapshot_id: int,
    query: str,
    finding_path: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Your repo_chunks.search_chunks is async and returns: (mode_used, hits).
    We prefer exact-path matches, and only fall back to "contains" matches if needed.
    """
    finding_path_norm = _norm_path(finding_path)
    if not finding_path_norm:
        return []

    mode_used, hits = await search_chunks(
        db=db,
        snapshot_id=snapshot_id,
        query=query,
        top_k=top_k,
        mode="auto",
        path_contains=finding_path_norm,  # substring filter in the DB query layer
    )

    snippets: list[dict[str, Any]] = []
    for h in hits or []:
        if isinstance(h, dict):
            path = _norm_path(h.get("path") or "")
            snippets.append(
                {
                    "path": path,
                    "start_line": h.get("start_line"),
                    "end_line": h.get("end_line"),
                    "text": h.get("chunk_text") or "",
                }
            )
        else:
            path = _norm_path(getattr(h, "path", "") or "")
            snippets.append(
                {
                    "path": path,
                    "start_line": getattr(h, "start_line", None),
                    "end_line": getattr(h, "end_line", None),
                    "text": getattr(h, "chunk_text", "") or "",
                }
            )

    # Prefer exact match on the finding file
    exact = [s for s in snippets if s.get("path") == finding_path_norm]
    if exact:
        return exact

    # Fallback: contains match (should be rare if your chunk paths are consistent)
    contains = [s for s in snippets if finding_path_norm in (s.get("path") or "")]
    return contains


async def generate_patch_for_finding(
    *,
    db: Session,
    snapshot_id: int,
    finding_id: int,
    objective: str | None = None,
) -> dict[str, Any]:
    snap = db.query(RepoSnapshot).filter(RepoSnapshot.id == snapshot_id).first()
    if not snap:
        raise PatchGenerationError(f"Unknown snapshot_id={snapshot_id}")

    finding = (
        db.query(RepoFinding)
        .filter(RepoFinding.id == finding_id, RepoFinding.snapshot_id == snapshot_id)
        .first()
    )
    if not finding:
        raise PatchGenerationError(f"Unknown finding_id={finding_id} for snapshot_id={snapshot_id}")

    allow = _allowlist_dirs()
    deny = _denylist_patterns()

    finding_path = _norm_path(str(getattr(finding, "path", "") or ""))
    if not finding_path:
        raise PatchGenerationError("Finding has empty path; refusing to generate patch.")

    # default: only allow touching the finding's path
    target_paths = [finding_path] if _require_finding_path_only() else [finding_path]

    # hard pre-check: if the *target path itself* is disallowed, fail immediately
    if not _path_allowed(finding_path, allow, deny):
        raise PatchGenerationError(f"Finding path is disallowed by policy: {finding_path}")

    q = _build_query_from_finding(finding)

    snippets = await _search_snippets_for_file(
        db=db,
        snapshot_id=snapshot_id,
        query=q,
        finding_path=finding_path,
        top_k=5,
    )

    # refuse to generate without code context (prevents broad hallucinated diffs)
    if not snippets:
        raise PatchGenerationError(
            f"No snippets available for finding path={finding_path}. "
            "Refusing to generate a patch without code context."
        )

    objective_text = (objective or "").strip()
    if not objective_text:
        objective_text = f"Implement the recommendation for the finding titled: {finding.title}"

    user_prompt = (
        f"{_format_constraints_for_prompt(allow_dirs=allow, deny_patterns=deny, target_paths=target_paths)}\n\n"
        "Target finding:\n"
        f"- id: {finding.id}\n"
        f"- category: {finding.category}\n"
        f"- severity: {finding.severity}\n"
        f"- title: {finding.title}\n"
        f"- path: {finding.path}\n"
        f"- line: {finding.line}\n"
        f"- evidence: {finding.evidence}\n"
        f"- recommendation: {finding.recommendation}\n"
        f"- acceptance: {getattr(finding, 'acceptance', None)}\n\n"
        "Objective:\n"
        f"{objective_text}\n\n"
        f"{_format_context_for_prompt(snippets)}\n"
        "Now output the unified diff ONLY.\n"
    )

    llm = LLMClient()
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

    nf_err = _reject_new_files(patch_text)
    if nf_err:
        raise PatchGenerationError(nf_err)

    touched = _extract_touched_paths(patch_text)
    if not touched:
        raise PatchGenerationError("Could not extract any touched paths from diff headers.")

    # enforce "touched ⊆ target_paths"
    for p in touched:
        p_norm = _norm_path(p)
        if p_norm not in target_paths:
            raise PatchGenerationError(f"Generated patch touches non-target path: {p_norm}")

    # enforce allow/deny on every touched path
    for p in touched:
        p_norm = _norm_path(p)
        if not _path_allowed(p_norm, allow, deny):
            raise PatchGenerationError(f"Generated patch touches disallowed path: {p_norm}")

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


async def generate_unified_diff(
    *,
    db: Session,
    snapshot_id: int,
    finding_id: Optional[int] = None,
    objective: Optional[str] = None,
) -> dict[str, Any]:
    """
    This is the function your debug route imports:
        from ..services.patch_generator import generate_unified_diff, PatchGenerationError

    Supported modes:
    - finding_id provided -> generate a patch scoped to that finding.
    - objective-only mode is intentionally NOT implemented yet (fail closed),
      because it allows wandering unless you also provide target paths.

    Returns: { snapshot_id, finding_id, objective, patch_text, snippets_used }
    """
    if finding_id is None:
        raise PatchGenerationError(
            "finding_id is required for now. Objective-only diff generation is disabled to prevent wandering."
        )

    return await generate_patch_for_finding(
        db=db,
        snapshot_id=int(snapshot_id),
        finding_id=int(finding_id),
        objective=objective,
    )
