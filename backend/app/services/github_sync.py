# backend/app/services/github_sync.py
from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from typing import Any

import httpx
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoFile, RepoSnapshot


@dataclass(frozen=True)
class SyncResult:
    snapshot_id: int
    repo: str
    branch: str
    commit_sha: str | None
    file_count: int
    stored_content_files: int
    warnings: list[str]


def _norm_path(p: str) -> str:
    return (p or "").lstrip("/").replace("\\", "/")


def _is_excluded_path(path: str) -> bool:
    p = _norm_path(path)

    # Exclude by dir name anywhere
    parts = [x for x in p.split("/") if x]
    if any(seg in set(settings.GITHUB_EXCLUDE_DIR_NAMES) for seg in parts):
        return True

    # Exclude by ext
    if "." in p:
        ext = p.rsplit(".", 1)[-1].lower()
        if ext in {x.lower() for x in settings.GITHUB_EXCLUDE_EXTENSIONS}:
            return True

    # Exclude by prefix (boundary-aware)
    for pref in settings.GITHUB_EXCLUDE_PREFIXES:
        pref_n = _norm_path(pref).rstrip("/")
        if not pref_n:
            continue
        if p == pref_n or p.startswith(pref_n + "/"):
            return True

    return False


def _is_included_path(path: str) -> bool:
    """
    ✅ NEW: include allowlist.
    If include prefixes/files is empty => include everything.
    """
    p = _norm_path(path)

    inc_pref = [(_norm_path(x).rstrip("/") + "/") for x in settings.GITHUB_INCLUDE_PREFIXES if x.strip()]
    inc_files = set(_norm_path(x) for x in settings.GITHUB_INCLUDE_FILES if x.strip())

    if not inc_pref and not inc_files:
        return True

    if p in inc_files:
        return True

    for pref in inc_pref:
        if p.startswith(pref):
            return True

    return False


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "goal-autopilot-repo-sync",
    }
    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"
    return headers


def _looks_textual(path: str, raw_bytes: bytes) -> bool:
    ext = (path.rsplit(".", 1)[-1].lower() if "." in path else "")
    if ext in {"png", "jpg", "jpeg", "gif", "webp", "pdf", "zip", "gz", "tar", "7z", "exe", "dll"}:
        return False

    mt, _ = mimetypes.guess_type(path)
    if mt and (mt.startswith("image/") or mt in {"application/pdf", "application/zip"}):
        return False

    if b"\x00" in raw_bytes[:4000]:
        return False

    try:
        raw_bytes.decode("utf-8")
        return True
    except Exception:
        return False


async def _github_get_json(url: str) -> Any:
    timeout = httpx.Timeout(connect=settings.GITHUB_CONNECT_TIMEOUT_S, read=settings.GITHUB_READ_TIMEOUT_S)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, headers=_github_headers())
        r.raise_for_status()
        return r.json()


async def _github_get_content(owner_repo: str, path: str, ref: str) -> tuple[bytes | None, str | None]:
    """
    Returns raw bytes (decoded from GitHub API base64) and sha.
    """
    url = f"https://api.github.com/repos/{owner_repo}/contents/{path}?ref={ref}"
    j = await _github_get_json(url)

    if isinstance(j, dict) and j.get("type") == "file":
        sha = j.get("sha")
        content_b64 = j.get("content")
        enc = j.get("encoding")
        if enc == "base64" and content_b64:
            raw = base64.b64decode(content_b64)
            return raw, sha
    return None, None


async def _github_list_tree(owner_repo: str, ref: str) -> tuple[list[dict[str, Any]], str | None]:
    """
    Use git/trees API to list all files in one shot.
    """
    # Get commit sha for branch/ref
    ref_url = f"https://api.github.com/repos/{owner_repo}/git/refs/heads/{ref}"
    try:
        ref_json = await _github_get_json(ref_url)
        commit_sha = ref_json.get("object", {}).get("sha")
    except Exception:
        commit_sha = None

    # Fallback: allow calling tree on branch name
    tree_ref = commit_sha or ref

    tree_url = f"https://api.github.com/repos/{owner_repo}/git/trees/{tree_ref}?recursive=1"
    tree_json = await _github_get_json(tree_url)

    items = tree_json.get("tree", []) if isinstance(tree_json, dict) else []
    return items, commit_sha


def latest_snapshot(db: Session, repo: str | None = None, branch: str | None = None) -> RepoSnapshot | None:
    q = db.query(RepoSnapshot)
    if repo:
        q = q.filter(RepoSnapshot.repo == repo)
    if branch:
        q = q.filter(RepoSnapshot.branch == branch)
    return q.order_by(desc(RepoSnapshot.id)).first()


def snapshot_file_stats(db: Session, snapshot_id: int) -> dict[str, Any]:
    files = db.scalars(select(RepoFile.path, RepoFile.content_kind, RepoFile.skipped).where(RepoFile.snapshot_id == snapshot_id)).all()
    total = len(files)
    text_files = sum(1 for _p, kind, _s in files if kind == "text")
    binary_files = sum(1 for _p, kind, _s in files if kind == "binary")
    skipped_files = sum(1 for _p, _k, s in files if s)

    # top folder
    counts: dict[str, int] = {}
    for p, _k, _s in files:
        root = p.split("/", 1)[0] if "/" in p else p
        counts[root] = counts.get(root, 0) + 1
    top_folders = [{"folder": k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:12]]

    return {
        "total_files": total,
        "text_files": text_files,
        "binary_files": binary_files,
        "skipped_files": skipped_files,
        "top_folders": top_folders,
    }


async def create_repo_snapshot(db: Session, repo: str | None = None, branch: str | None = None) -> SyncResult:
    """
    This is the function your routes + scheduler import.
    """
    owner_repo = repo or "AzRea7/OneHaven"
    ref = branch or "main"

    warnings: list[str] = []

    items, commit_sha = await _github_list_tree(owner_repo, ref=ref)

    snap = RepoSnapshot(
        repo=owner_repo,
        branch=ref,
        commit_sha=commit_sha,
        file_count=0,
        stored_content_files=0,
        warnings_json=None,
    )
    db.add(snap)
    db.commit()
    db.refresh(snap)

    # Filter only "blob" entries (files)
    blobs = [x for x in items if x.get("type") == "blob"]

    # Apply include/exclude
    kept: list[dict[str, Any]] = []
    for x in blobs:
        p = _norm_path(x.get("path", ""))
        if not p:
            continue
        if not _is_included_path(p):
            continue
        if _is_excluded_path(p):
            continue
        kept.append(x)

    if len(kept) > settings.GITHUB_MAX_FILES:
        warnings.append(f"Too many files after filtering ({len(kept)}). Truncating to {settings.GITHUB_MAX_FILES}.")
        kept = kept[: settings.GITHUB_MAX_FILES]

    stored_text = 0

    for x in kept:
        path = _norm_path(x.get("path", ""))
        size = x.get("size")
        sha = x.get("sha")

        # Default: skipped until proven text
        content_kind = "skipped"
        content = None
        skipped = False
        skip_reason = None
        is_text = True

        if size is not None and size > settings.GITHUB_MAX_TEXT_BYTES:
            skipped = True
            skip_reason = f"too_large>{settings.GITHUB_MAX_TEXT_BYTES}"
            content_kind = "skipped"
        else:
            raw, sha2 = await _github_get_content(owner_repo, path=path, ref=ref)
            if raw is None:
                skipped = True
                skip_reason = "no_content"
                content_kind = "skipped"
            else:
                if not _looks_textual(path, raw):
                    content_kind = "binary"
                    is_text = False
                    skipped = True
                    skip_reason = "binary"
                else:
                    # Store UTF-8 content (truncate if huge)
                    txt = raw.decode("utf-8", errors="replace")
                    if len(txt) > settings.GITHUB_MAX_TEXT_BYTES:
                        txt = txt[: settings.GITHUB_MAX_TEXT_BYTES]
                        skip_reason = "truncated"
                    content = txt
                    content_kind = "text"
                    stored_text += 1
                    sha = sha2 or sha

        db.add(
            RepoFile(
                snapshot_id=snap.id,
                path=path,
                sha=sha,
                size=size,
                is_text=is_text,
                skipped=skipped,
                skip_reason=skip_reason,
                content=content,
                content_kind=content_kind,
                content_text=content,  # optional mirror
            )
        )

    snap.file_count = len(kept)
    snap.stored_content_files = stored_text
    snap.warnings_json = json.dumps(warnings) if warnings else None

    db.commit()

    return SyncResult(
        snapshot_id=snap.id,
        repo=snap.repo,
        branch=snap.branch,
        commit_sha=snap.commit_sha,
        file_count=snap.file_count,
        stored_content_files=snap.stored_content_files,
        warnings=warnings,
    )
