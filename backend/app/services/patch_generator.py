# backend/app/services/patch_generator.py
from __future__ import annotations

import os
import re
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..ai.llm import LLMClient
from ..config import settings
from ..models import RepoChunk, RepoFinding, RepoSnapshot
from .repo_chunks import search_chunks  # async in your app


class PatchGenerationError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$", re.MULTILINE)
_PLUSPLUS_RE = re.compile(r"^\+\+\+ b/(.+?)\s*$", re.MULTILINE)


def _llm_temperature() -> float:
    return float(getattr(settings, "PATCH_LLM_TEMPERATURE", 0.1) or 0.1)


def _llm_max_tokens() -> int:
    return int(getattr(settings, "PATCH_LLM_MAX_TOKENS", 2600) or 2600)


def _max_files() -> int:
    return int(getattr(settings, "PATCH_MAX_FILES_CHANGED", 3) or 3)


def _forbid_new_files() -> bool:
    return bool(getattr(settings, "PATCH_FORBID_NEW_FILES", True))


def _require_finding_path_only() -> bool:
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
    s = (p or "").strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s


def _path_allowed(path: str, allow_dirs: list[str], deny_patterns: list[str]) -> bool:
    p_norm = _norm_path(path)
    if not p_norm:
        return False

    if p_norm.startswith("/") or p_norm.startswith("\\"):
        return False
    if ".." in p_norm.split("/"):
        return False

    if allow_dirs:
        allow_norm = [_norm_path(a) for a in allow_dirs]
        if not any(p_norm.startswith(a) for a in allow_norm):
            return False

    for pat in deny_patterns:
        try:
            if re.search(pat, p_norm):
                return False
        except re.error:
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


def _extract_unified_diff_from_model_output(text: str) -> str:
    """
    Models often prepend explanations or wrap diffs in fences.
    We aggressively extract the first real unified diff starting at 'diff --git'.

    If the output contains no 'diff --git', return "".
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    # Strip common ``` fences while keeping content
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        lines = raw.splitlines()
        if lines and lines[0].strip().lower() in {"diff", "patch", "git", "udiff", "unified"}:
            raw = "\n".join(lines[1:]).strip()

    idx = raw.find("diff --git")
    if idx == -1:
        m = re.search(r"\bdiff --git\b", raw, flags=re.IGNORECASE)
        if not m:
            return ""
        idx = m.start()

    diff = raw[idx:].lstrip()
    return diff.strip()


def _basic_sanity_check(patch_text: str) -> None:
    s = (patch_text or "").strip()
    if not s:
        raise PatchGenerationError("Empty patch_text from model.")
    if "diff --git " not in s:
        raise PatchGenerationError("Model did not output a unified diff (missing 'diff --git').")


def _build_system_prompt() -> str:
    return (
        "You are a senior software engineer.\n"
        "You MUST output ONLY a unified git diff.\n"
        "The FIRST non-whitespace characters of your response MUST be: diff --git\n"
        "No explanations. No markdown. No code blocks.\n"
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
        "- Your response MUST start with: diff --git\n"
        "- Do NOT touch any paths outside TARGET_PATHS.\n"
        "- Do NOT create or delete files.\n"
        "- If you cannot implement the fix without violating rules, output an EMPTY diff (no changes):\n"
        "  (i.e. output nothing at all)\n"
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


def _has_real_text(t: Any) -> bool:
    s = (t or "")
    return isinstance(s, str) and len(s.strip()) >= 40


async def _search_snippets_for_path(
    *,
    db: Session,
    snapshot_id: int,
    query: str,
    path_contains: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    target = _norm_path(path_contains)

    async def _try_search(q: str) -> list[Any]:
        try:
            _, hits = await search_chunks(
                db=db,
                snapshot_id=snapshot_id,
                query=q,
                top_k=top_k,
                mode="auto",
                path_contains=target,
                include_text=True,  # if supported
            )
            return hits or []
        except TypeError:
            _, hits = await search_chunks(
                db=db,
                snapshot_id=snapshot_id,
                query=q,
                top_k=top_k,
                mode="auto",
                path_contains=target,
            )
            return hits or []

    hits = await _try_search(query)

    if not hits:
        base = os.path.basename(target)
        fallback_queries = [
            "LLMClient",
            "complete_json",
            "provider",
            base if len(base) >= 2 else "",
            " ".join([w for w in query.split()[:6] if len(w) >= 2]),
        ]
        for fq in [x for x in fallback_queries if x.strip()]:
            hits = await _try_search(fq)
            if hits:
                break

    snippets: list[dict[str, Any]] = []
    for h in hits:
        if isinstance(h, dict):
            snippets.append(
                {
                    "path": _norm_path(str(h.get("path") or "")),
                    "start_line": h.get("start_line"),
                    "end_line": h.get("end_line"),
                    "text": h.get("chunk_text") or "",
                }
            )
        else:
            snippets.append(
                {
                    "path": _norm_path(str(getattr(h, "path", "") or "")),
                    "start_line": getattr(h, "start_line", None),
                    "end_line": getattr(h, "end_line", None),
                    "text": getattr(h, "chunk_text", "") or "",
                }
            )

    snippets = [s for s in snippets if _norm_path(str(s.get("path") or "")) == target]

    if not snippets or not any(_has_real_text(s.get("text")) for s in snippets):
        rows = (
            db.execute(
                select(RepoChunk)
                .where(RepoChunk.snapshot_id == snapshot_id)
                .where(RepoChunk.path == target)
                .order_by(RepoChunk.start_line.asc())
                .limit(max(top_k, 1))
            )
            .scalars()
            .all()
        )
        if rows:
            snippets = [
                {
                    "path": _norm_path(r.path),
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "text": r.chunk_text or "",
                }
                for r in rows
            ]

    snippets = [s for s in snippets if _has_real_text(s.get("text"))]
    return snippets


def _build_pr_meta_schema_hint() -> str:
    return (
        "{\n"
        '  "title": "string (<= 90 chars)",\n'
        '  "body": "string (markdown). Must include: Summary, Why, How Verified, Traceability"\n'
        "}\n"
    )


def _build_pr_meta_system_prompt() -> str:
    return (
        "You write high-quality GitHub PR titles and descriptions.\n"
        "Output ONLY valid JSON (no markdown fences).\n"
        "Do not hallucinate tests or verification. If not run, say 'Not run'.\n"
        "Keep it concise and actionable.\n"
    )


def _truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "\n...(truncated)\n"


def _safe_json_dict(obj: Any) -> dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def _fallback_pr_title(*, finding: RepoFinding) -> str:
    base = (str(getattr(finding, "title", "") or "").strip() or "Apply repo finding fix")
    # keep it PR-friendly and short
    if len(base) > 90:
        base = base[:87].rstrip() + "..."
    return base


def _fallback_pr_body(*, finding: RepoFinding, snap: RepoSnapshot, patch_text: str) -> str:
    touched = _extract_touched_paths(patch_text)
    touched_md = "\n".join([f"- `{p}`" for p in touched]) if touched else "- (unable to parse paths from diff)"

    rec = str(getattr(finding, "recommendation", "") or "").strip()
    if not rec:
        rec = "Address an automated repo finding."

    return (
        "## Summary\n"
        f"- Fix: **{str(getattr(finding, 'title', '') or '').strip() or 'Repo finding'}**\n\n"
        "## Why\n"
        f"- {rec}\n\n"
        "## How Verified\n"
        "- Not run. Recommended: run unit tests and targeted smoke checks for the touched area.\n\n"
        "## Files Touched\n"
        f"{touched_md}\n\n"
        "## Traceability\n"
        f"- Repo: {snap.repo}\n"
        f"- Branch: {snap.branch}\n"
        f"- Snapshot: {snap.id}\n"
        f"- Finding: {finding.id}\n"
    )


async def _generate_pr_metadata(
    *,
    finding: RepoFinding,
    snap: RepoSnapshot,
    patch_text: str,
) -> tuple[str, str]:
    """
    Second bounded LLM call: produce PR title/body as JSON.

    IMPORTANT: This must never return (None, None).
    If the LLM fails or returns empty fields, we fall back to deterministic metadata.
    """
    fallback_title = _fallback_pr_title(finding=finding)
    fallback_body = _fallback_pr_body(finding=finding, snap=snap, patch_text=patch_text)

    try:
        llm = LLMClient()
        patch_excerpt = _truncate(patch_text, 6000)

        user_prompt = (
            "Generate PR metadata for this patch.\n\n"
            f"Repo: {snap.repo}\n"
            f"Branch: {snap.branch}\n"
            f"Snapshot ID: {snap.id}\n"
            f"Finding ID: {finding.id}\n"
            f"Finding Title: {finding.title}\n"
            f"Path: {finding.path}\n\n"
            "Finding Recommendation:\n"
            f"{finding.recommendation}\n\n"
            "Patch (excerpt):\n"
            f"{patch_excerpt}\n\n"
            "Body format requirements:\n"
            "## Summary\n- ...\n"
            "## Why\n- ...\n"
            "## How Verified\n- Not run (and what should be run)\n"
            "## Traceability\n- Snapshot: <id>\n- Finding: <id>\n"
        )

        obj = await llm.complete_json(
            system=_build_pr_meta_system_prompt(),
            user=user_prompt,
            schema_hint=_build_pr_meta_schema_hint(),
        )

        d = _safe_json_dict(obj)
        title = str(d.get("title") or "").strip()
        body = str(d.get("body") or "").strip()

        # enforce guardrails
        if title:
            title = title[:300].strip()
        if body:
            body = body.strip()

        if not title:
            title = fallback_title
        if not body:
            body = fallback_body

        return (title, body)
    except Exception:
        return (fallback_title, fallback_body)


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

    target_paths = [finding_path] if _require_finding_path_only() else [finding_path]

    if not _path_allowed(finding_path, allow, deny):
        raise PatchGenerationError(f"Finding path is disallowed by policy: {finding_path}")

    q = " ".join(
        [
            str(getattr(finding, "title", "") or ""),
            str(getattr(finding, "evidence", "") or ""),
            str(getattr(finding, "recommendation", "") or ""),
        ]
    ).strip() or f"Fix finding {finding_id}"

    snippets = await _search_snippets_for_path(
        db=db,
        snapshot_id=snapshot_id,
        query=q,
        path_contains=finding_path,
        top_k=5,
    )

    if not snippets:
        raise PatchGenerationError(
            f"No snippets available for finding path={finding_path}. "
            "Refusing to generate a patch without code context."
        )

    objective_text = (objective or "").strip() or f"Implement the recommendation for the finding titled: {finding.title}"

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
        "IMPORTANT: Output ONLY a unified diff, and start your response with: diff --git\n"
    )

    llm = LLMClient()
    raw_out = await llm.chat(
        system=_build_system_prompt(),
        user=user_prompt,
        temperature=_llm_temperature(),
        max_tokens=_llm_max_tokens(),
        response_format={"type": "text"},
        force_text_response_format=True,
    )

    raw_out = (raw_out or "").strip()
    patch_text = _extract_unified_diff_from_model_output(raw_out)

    if not patch_text:
        raise PatchGenerationError(
            "Model did not return a unified diff. "
            "It must start with 'diff --git'. Try again or tighten the prompt/model."
        )

    _basic_sanity_check(patch_text)

    nf_err = _reject_new_files(patch_text)
    if nf_err:
        raise PatchGenerationError(nf_err)

    touched = _extract_touched_paths(patch_text)
    if not touched:
        raise PatchGenerationError("Could not extract any touched paths from diff headers.")

    for p in touched:
        if _norm_path(p) not in [_norm_path(tp) for tp in target_paths]:
            raise PatchGenerationError(f"Generated patch touches non-target path: {p}")

    for p in touched:
        if not _path_allowed(p, allow, deny):
            raise PatchGenerationError(f"Generated patch touches disallowed path: {p}")

    # PR metadata must never be null/empty now.
    pr_title, pr_body = await _generate_pr_metadata(
        finding=finding,
        snap=snap,
        patch_text=patch_text,
    )

    return {
        "snapshot_id": snapshot_id,
        "finding_id": finding_id,
        "objective": objective_text,
        "patch_text": patch_text,
        "snippets_used": [
            {"path": s["path"], "start_line": s["start_line"], "end_line": s["end_line"]}
            for s in snippets
        ],
        "suggested_pr_title": pr_title,
        "suggested_pr_body": pr_body,
    }


async def generate_unified_diff(
    *,
    db: Session,
    snapshot_id: int,
    finding_id: Optional[int] = None,
    objective: Optional[str] = None,
) -> dict[str, Any]:
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
