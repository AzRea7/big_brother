# backend/app/services/github_sync.py
from __future__ import annotations

import base64
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import httpx
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoFile, RepoSnapshot


_TEXT_EXT_ALLOW = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
    ".sh",
    ".bash",
    ".zsh",
    ".dockerfile",
    ".sql",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".scss",
    ".csv",
    ".xml",
}


def _now() -> float:
    return time.time()


def _norm_ext(path: str) -> str:
    base = path.lower()
    _, ext = os.path.splitext(base)
    return ext


def _looks_like_text(path: str, b: bytes) -> bool:
    """
    Conservative text detector:
    - allowlist extensions (fast)
    - else: reject if NUL bytes present
    - else: treat as text if decodes as utf-8 reasonably
    """
    ext = _norm_ext(path)
    if ext in _TEXT_EXT_ALLOW:
        return True
    if b"\x00" in b:
        return False
    try:
        b.decode("utf-8")
        return True
    except Exception:
        return False


def _should_exclude_path(path: str) -> tuple[bool, str | None]:
    p = path.replace("\\", "/")

    # prefixes
    for pref in getattr(settings, "GITHUB_EXCLUDE_PREFIXES", []) or []:
        if p.startswith(pref):
            return True, f"excluded_prefix:{pref}"

    # dir names
    parts = p.split("/")
    for dn in getattr(settings, "GITHUB_EXCLUDE_DIR_NAMES", []) or []:
        if dn in parts:
            return True, f"excluded_dir:{dn}"

    # extensions
    ext = _norm_ext(p)
    exts = getattr(settings, "GITHUB_EXCLUDE_EXTENSIONS", []) or []
    if ext:
        ext_no_dot = ext[1:] if ext.startswith(".") else ext
        if ext_no_dot in exts:
            return True, f"excluded_ext:{ext_no_dot}"

    return False, None


def _safe_read_file_bytes(path: str, max_bytes: int) -> tuple[bytes | None, str | None]:
    try:
        st = os.stat(path)
        if st.st_size > max_bytes:
            return None, f"over_max_bytes:{st.st_size}"
    except Exception as e:
        return None, f"stat_failed:{e!r}"

    try:
        with open(path, "rb") as f:
            b = f.read(max_bytes + 1)
        if len(b) > max_bytes:
            return None, f"over_max_bytes:{len(b)}"
        return b, None
    except Exception as e:
        return None, f"read_failed:{e!r}"


def _iter_local_files(root: str) -> Iterable[tuple[str, str]]:
    """
    Yield (relative_path, abs_path) for files under root.
    """
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        # cheap pruning
        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        if rel_dir == ".":
            rel_dir = ""

        # remove excluded dirs early
        pruned = []
        for d in list(dirnames):
            rel_candidate = f"{rel_dir}/{d}" if rel_dir else d
            excluded, _ = _should_exclude_path(rel_candidate + "/")
            if excluded:
                pruned.append(d)
        for d in pruned:
            try:
                dirnames.remove(d)
            except ValueError:
                pass

        for fn in filenames:
            rel = f"{rel_dir}/{fn}" if rel_dir else fn
            rel = rel.replace("\\", "/")
            abs_path = os.path.join(dirpath, fn)
            yield rel, abs_path


def _github_headers() -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "goal-autopilot",
    }
    tok = getattr(settings, "GITHUB_TOKEN", None) or None
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _github_api_base(repo: str) -> str:
    # repo is like "AzRea7/OneHaven"
    return f"https://api.github.com/repos/{repo}"


def _parse_repo(repo: str | None) -> str:
    r = (repo or getattr(settings, "GITHUB_REPO", "") or "").strip()
    if not r:
        raise ValueError("repo is required")
    return r


def _parse_branch(branch: str | None) -> str:
    b = (branch or getattr(settings, "GITHUB_BRANCH", "") or "main").strip()
    return b or "main"


def _get_commit_sha_via_github(repo: str, branch: str) -> str | None:
    try:
        with httpx.Client(
            timeout=httpx.Timeout(
                connect=float(getattr(settings, "GITHUB_CONNECT_TIMEOUT_S", 10.0)),
                read=float(getattr(settings, "GITHUB_READ_TIMEOUT_S", 60.0)),
            ),
            headers=_github_headers(),
        ) as client:
            r = client.get(f"{_github_api_base(repo)}/commits/{branch}")
            if r.status_code >= 400:
                return None
            js = r.json()
            return js.get("sha")
    except Exception:
        return None


def _github_list_tree(repo: str, branch: str) -> list[dict[str, Any]]:
    """
    Uses the git tree API to list all files.
    """
    sha = _get_commit_sha_via_github(repo, branch)
    if not sha:
        return []
    with httpx.Client(
        timeout=httpx.Timeout(
            connect=float(getattr(settings, "GITHUB_CONNECT_TIMEOUT_S", 10.0)),
            read=float(getattr(settings, "GITHUB_READ_TIMEOUT_S", 60.0)),
        ),
        headers=_github_headers(),
    ) as client:
        # recursive tree
        url = f"{_github_api_base(repo)}/git/trees/{sha}?recursive=1"
        r = client.get(url)
        r.raise_for_status()
        js = r.json()
        tree = js.get("tree") or []
        # keep only blobs
        blobs = [t for t in tree if t.get("type") == "blob" and t.get("path")]
        return blobs


def _github_get_file_contents(repo: str, path: str, ref: str) -> tuple[bytes | None, str | None, int | None]:
    """
    Fetch file content via contents API (base64). Returns (bytes, sha, size).
    """
    url = f"{_github_api_base(repo)}/contents/{path}"
    params = {"ref": ref}

    try:
        with httpx.Client(
            timeout=httpx.Timeout(
                connect=float(getattr(settings, "GITHUB_CONNECT_TIMEOUT_S", 10.0)),
                read=float(getattr(settings, "GITHUB_READ_TIMEOUT_S", 60.0)),
            ),
            headers=_github_headers(),
        ) as client:
            r = client.get(url, params=params)
            if r.status_code >= 400:
                return None, None, None
            js = r.json()
            # If it's a directory, js will be a list
            if isinstance(js, list):
                return None, None, None
            content_b64 = js.get("content")
            if not content_b64:
                return None, js.get("sha"), js.get("size")
            # GitHub inserts newlines in base64 content
            raw = base64.b64decode(content_b64.encode("utf-8"))
            return raw, js.get("sha"), js.get("size")
    except Exception:
        return None, None, None


# --------------------------------------------------------------------
# PUBLIC ENTRYPOINT expected by routes/repo.py
# --------------------------------------------------------------------

def sync_repo_to_snapshot(db: Session, repo: str | None = None, branch: str | None = None) -> dict[str, Any]:
    """
    Creates a RepoSnapshot and populates RepoFile rows.

    Prefers local disk scan if settings.REPO_LOCAL_PATH is set (docker mount).
    Falls back to GitHub API if not.

    Returns:
      {
        "snapshot_id": int,
        "repo": str,
        "branch": str,
        "commit_sha": str|None,
        "file_count": int,
        "stored_content_files": int,
        "warnings": list[str]
      }
    """
    repo = _parse_repo(repo)
    branch = _parse_branch(branch)

    warnings: list[str] = []
    max_bytes = int(getattr(settings, "GITHUB_MAX_FILE_BYTES", 150_000) or 150_000)
    max_files = int(getattr(settings, "GITHUB_MAX_FILES_PER_SYNC", 5000) or 5000)

    # create snapshot row first
    commit_sha: str | None = None
    if getattr(settings, "REPO_LOCAL_PATH", None):
        # Optional: user may provide a local git sha manually via env
        commit_sha = getattr(settings, "REPO_LOCAL_GIT_SHA", None) or None
    else:
        commit_sha = _get_commit_sha_via_github(repo, branch)

    snap = RepoSnapshot(
        repo=repo,
        branch=branch,
        commit_sha=commit_sha,
        file_count=0,
        stored_content_files=0,
        warnings_json=None,
    )
    db.add(snap)
    db.commit()
    db.refresh(snap)

    stored = 0
    total = 0

    # -------------------------
    # Local scan (preferred)
    # -------------------------
    local_root = getattr(settings, "REPO_LOCAL_PATH", None) or None
    if local_root:
        root = local_root
        if not os.path.exists(root):
            warnings.append(f"REPO_LOCAL_PATH does not exist: {root}")
        else:
            for rel, abs_path in _iter_local_files(root):
                if total >= max_files:
                    warnings.append(f"hit_max_files:{max_files}")
                    break

                excluded, reason = _should_exclude_path(rel)
                if excluded:
                    db.add(
                        RepoFile(
                            snapshot_id=snap.id,
                            path=rel,
                            sha=None,
                            size=None,
                            is_text=True,
                            skipped=True,
                            skip_reason=reason,
                            content=None,
                            content_kind="skipped",
                            content_text=None,
                        )
                    )
                    total += 1
                    continue

                b, err = _safe_read_file_bytes(abs_path, max_bytes=max_bytes)
                if b is None:
                    db.add(
                        RepoFile(
                            snapshot_id=snap.id,
                            path=rel,
                            sha=None,
                            size=None,
                            is_text=True,
                            skipped=True,
                            skip_reason=err or "read_failed",
                            content=None,
                            content_kind="skipped",
                            content_text=None,
                        )
                    )
                    total += 1
                    continue

                is_text = _looks_like_text(rel, b)
                content_text = None
                kind = "binary"
                if is_text:
                    try:
                        content_text = b.decode("utf-8", errors="replace")
                        kind = "text"
                    except Exception:
                        content_text = None
                        kind = "binary"

                db.add(
                    RepoFile(
                        snapshot_id=snap.id,
                        path=rel,
                        sha=None,
                        size=len(b),
                        is_text=bool(is_text),
                        skipped=False,
                        skip_reason=None,
                        content=None,  # legacy field, keep None
                        content_kind=kind if is_text else "binary",
                        content_text=content_text if is_text else None,
                    )
                )
                total += 1
                if is_text and content_text:
                    stored += 1

            db.commit()

    # -------------------------
    # GitHub API fallback
    # -------------------------
    else:
        tree = _github_list_tree(repo, branch)
        if not tree:
            warnings.append("github_tree_empty_or_failed")

        for item in tree[:max_files]:
            path = (item.get("path") or "").strip()
            if not path:
                continue

            excluded, reason = _should_exclude_path(path)
            if excluded:
                db.add(
                    RepoFile(
                        snapshot_id=snap.id,
                        path=path,
                        sha=item.get("sha"),
                        size=item.get("size"),
                        is_text=True,
                        skipped=True,
                        skip_reason=reason,
                        content=None,
                        content_kind="skipped",
                        content_text=None,
                    )
                )
                total += 1
                continue

            size = item.get("size")
            if isinstance(size, int) and size > max_bytes:
                db.add(
                    RepoFile(
                        snapshot_id=snap.id,
                        path=path,
                        sha=item.get("sha"),
                        size=size,
                        is_text=True,
                        skipped=True,
                        skip_reason=f"over_max_bytes:{size}",
                        content=None,
                        content_kind="skipped",
                        content_text=None,
                    )
                )
                total += 1
                continue

            raw, sha, sz = _github_get_file_contents(repo, path, ref=branch)
            if raw is None:
                db.add(
                    RepoFile(
                        snapshot_id=snap.id,
                        path=path,
                        sha=sha or item.get("sha"),
                        size=sz or item.get("size"),
                        is_text=True,
                        skipped=True,
                        skip_reason="github_fetch_failed",
                        content=None,
                        content_kind="skipped",
                        content_text=None,
                    )
                )
                total += 1
                continue

            is_text = _looks_like_text(path, raw)
            content_text = None
            kind = "binary"
            if is_text:
                content_text = raw.decode("utf-8", errors="replace")
                kind = "text"

            db.add(
                RepoFile(
                    snapshot_id=snap.id,
                    path=path,
                    sha=sha or item.get("sha"),
                    size=len(raw),
                    is_text=bool(is_text),
                    skipped=False,
                    skip_reason=None,
                    content=None,
                    content_kind=kind if is_text else "binary",
                    content_text=content_text if is_text else None,
                )
            )
            total += 1
            if is_text and content_text:
                stored += 1

        db.commit()

    # update snapshot stats + warnings
    snap.file_count = int(total)
    snap.stored_content_files = int(stored)
    if warnings:
        # keep it light; routes can display it
        import json

        snap.warnings_json = json.dumps(warnings)
    db.commit()

    return {
        "snapshot_id": snap.id,
        "repo": repo,
        "branch": branch,
        "commit_sha": commit_sha,
        "file_count": int(total),
        "stored_content_files": int(stored),
        "warnings": warnings,
    }
