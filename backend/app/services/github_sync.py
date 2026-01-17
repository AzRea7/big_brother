# backend/app/services/github_sync.py
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoFile, RepoSnapshot


@dataclass
class RepoSyncResult:
    snapshot_id: int
    repo: str
    branch: str
    commit_sha: str | None
    file_count: int
    stored_content_files: int
    warnings: list[str]


def _norm_path(p: str) -> str:
    return p.replace("\\", "/")


def _has_excluded_dir(path: str) -> bool:
    parts = [x for x in _norm_path(path).split("/") if x]
    return any(part in settings.GITHUB_EXCLUDE_DIR_NAMES for part in parts)


def _is_excluded_path(path: str) -> bool:
    p = _norm_path(path)

    # prefix exclusions
    for pref in settings.GITHUB_EXCLUDE_PREFIXES:
        if p.startswith(_norm_path(pref)):
            return True

    # dir-name exclusions
    if _has_excluded_dir(p):
        return True

    # extension exclusions
    ext = p.rsplit(".", 1)[-1].lower() if "." in p else ""
    if ext and ext in set(settings.GITHUB_EXCLUDE_EXTENSIONS):
        return True

    return False


def _is_included_path(path: str) -> bool:
    # No allowlist. Everything is included unless excluded.
    return True


def _looks_like_text(path: str, raw: bytes) -> bool:
    """
    Cheap heuristic: reject null bytes + reject known binary extensions (already filtered)
    """
    if b"\x00" in raw:
        return False
    return True


def _make_timeout() -> httpx.Timeout:
    """
    Fixes: ValueError: Timeout must either include a default, or set all four parameters explicitly.
    """
    return httpx.Timeout(
        settings.GITHUB_READ_TIMEOUT_S,
        connect=settings.GITHUB_CONNECT_TIMEOUT_S,
    )


async def _github_get_json(client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> Any:
    r = await client.get(url, headers=headers, timeout=_make_timeout())
    r.raise_for_status()
    return r.json()


async def _github_get_bytes(client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> bytes:
    r = await client.get(url, headers=headers, timeout=_make_timeout())
    r.raise_for_status()
    return r.content


async def _get_commit_sha(repo: str, branch: str) -> str | None:
    token = settings.GITHUB_TOKEN
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "goal-autopilot",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{repo}/commits/{branch}"
    async with httpx.AsyncClient() as client:
        try:
            j = await _github_get_json(client, url, headers)
            return j.get("sha")
        except Exception:
            return None


def latest_snapshot(db: Session, repo: str | None = None, branch: str | None = None) -> RepoSnapshot | None:
    q = select(RepoSnapshot).order_by(RepoSnapshot.id.desc())
    if repo:
        q = q.where(RepoSnapshot.repo == repo)
    if branch:
        q = q.where(RepoSnapshot.branch == branch)
    return db.execute(q).scalars().first()


def snapshot_file_stats(db: Session, snapshot_id: int) -> dict[str, Any]:
    rows = db.execute(
        select(RepoFile.path, RepoFile.skipped, RepoFile.is_text).where(RepoFile.snapshot_id == snapshot_id)
    ).all()

    total = len(rows)
    skipped = sum(1 for _, s, _ in rows if s)
    text_files = sum(1 for _, s, t in rows if (not s) and t)
    binary_files = sum(1 for _, s, t in rows if (not s) and (not t))

    folder_counts: dict[str, int] = {}
    for (p, _, _) in rows:
        p = _norm_path(p)
        top = p.split("/", 1)[0] if "/" in p else p
        folder_counts[top] = folder_counts.get(top, 0) + 1

    top_folders = [{"folder": k, "count": v} for k, v in sorted(folder_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]]
    return {
        "total_files": total,
        "text_files": text_files,
        "binary_files": binary_files,
        "skipped_files": skipped,
        "top_folders": top_folders,
    }


async def _scan_local_repo(base_path: str) -> list[dict[str, Any]]:
    """
    Returns: [{path, abs_path, size}]
    Applies include/exclude filters.
    """
    base_path = os.path.abspath(base_path)
    out: list[dict[str, Any]] = []

    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in settings.GITHUB_EXCLUDE_DIR_NAMES]

        for fn in files:
            abs_path = os.path.join(root, fn)
            rel_path = _norm_path(os.path.relpath(abs_path, base_path))

            if not _is_included_path(rel_path):
                continue
            if _is_excluded_path(rel_path):
                continue

            try:
                st = os.stat(abs_path)
            except FileNotFoundError:
                continue

            out.append({"path": rel_path, "abs_path": abs_path, "size": int(st.st_size)})

            if len(out) >= settings.GITHUB_MAX_FILES_PER_SYNC:
                return out

    return out


async def _list_github_repo_paths(repo: str, branch: str) -> list[dict[str, Any]]:
    """
    Uses GitHub Contents API (recursive BFS) to gather file paths.
    Applies include/exclude filters.
    """
    token = settings.GITHUB_TOKEN
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "goal-autopilot",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    queue: list[str] = [""]
    files_out: list[dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        while queue:
            path = queue.pop(0)
            url = (
                f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
                if path
                else f"https://api.github.com/repos/{repo}/contents?ref={branch}"
            )
            data = await _github_get_json(client, url, headers)
            if isinstance(data, dict):
                data = [data]

            for item in data:
                itype = item.get("type")
                ipath = _norm_path(item.get("path", ""))

                if not ipath:
                    continue
                if not _is_included_path(ipath):
                    continue
                if _is_excluded_path(ipath):
                    continue

                if itype == "dir":
                    queue.append(ipath)
                elif itype == "file":
                    files_out.append(
                        {
                            "path": ipath,
                            "sha": item.get("sha"),
                            "size": item.get("size"),
                            "download_url": item.get("download_url"),
                            "url": item.get("url"),
                        }
                    )
                    if len(files_out) >= settings.GITHUB_MAX_FILES_PER_SYNC:
                        return files_out

    return files_out


async def create_repo_snapshot(db: Session, repo: str | None = None, branch: str | None = None) -> RepoSyncResult:
    """
    REAL implementation (kept). The router should call sync_repo_snapshot() below,
    which forwards here.
    """
    repo = repo or settings.GITHUB_REPO
    branch = branch or settings.GITHUB_BRANCH

    warnings: list[str] = []
    commit_sha = settings.REPO_LOCAL_GIT_SHA or await _get_commit_sha(repo, branch)

    snap = RepoSnapshot(
        repo=repo,
        branch=branch,
        commit_sha=commit_sha,
        file_count=0,
        stored_content_files=0,
        warnings_json=(json.dumps(warnings) if warnings else None),
        created_at=datetime.utcnow(),
    )
    db.add(snap)
    db.commit()
    db.refresh(snap)

    stored = 0
    total = 0

    # Prefer local scan if configured
    if settings.REPO_LOCAL_PATH:
        items = await _scan_local_repo(settings.REPO_LOCAL_PATH)
        total = len(items)

        for it in items:
            path = it["path"]
            size = int(it["size"])

            rf = RepoFile(
                snapshot_id=snap.id,
                path=path,
                sha=None,
                size=size,
                is_text=True,
                skipped=False,
                content_kind="skipped",
                content=None,
                content_text=None,
                created_at=datetime.utcnow(),
                skip_reason=None,
            )

            if size <= settings.GITHUB_MAX_FILE_BYTES:
                try:
                    with open(it["abs_path"], "rb") as f:
                        raw = f.read()
                    if _looks_like_text(path, raw):
                        text = raw.decode("utf-8", errors="replace")
                        rf.content_kind = "text"
                        rf.content_text = text
                        rf.content = text
                        stored += 1
                    else:
                        rf.is_text = False
                        rf.skipped = True
                        rf.content_kind = "binary"
                        rf.skip_reason = "binary"
                except Exception as e:
                    rf.skipped = True
                    rf.content_kind = "skipped"
                    rf.skip_reason = f"read_error:{type(e).__name__}"
            else:
                rf.skipped = True
                rf.content_kind = "skipped"
                rf.skip_reason = "too_large"

            db.add(rf)

        snap.file_count = total
        snap.stored_content_files = stored
        snap.warnings_json = json.dumps(warnings) if warnings else None
        db.commit()

        return RepoSyncResult(
            snapshot_id=snap.id,
            repo=repo,
            branch=branch,
            commit_sha=commit_sha,
            file_count=total,
            stored_content_files=stored,
            warnings=warnings,
        )

    # GitHub API path
    items = await _list_github_repo_paths(repo, branch)
    total = len(items)

    token = settings.GITHUB_TOKEN
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "goal-autopilot",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient() as client:
        for it in items:
            path = it["path"]
            sha = it.get("sha")
            size = int(it.get("size") or 0)
            download_url = it.get("download_url")
            api_url = it.get("url")

            rf = RepoFile(
                snapshot_id=snap.id,
                path=path,
                sha=sha,
                size=size,
                is_text=True,
                skipped=False,
                content_kind="skipped",
                content=None,
                content_text=None,
                created_at=datetime.utcnow(),
                skip_reason=None,
            )

            if size > settings.GITHUB_MAX_FILE_BYTES:
                rf.skipped = True
                rf.content_kind = "skipped"
                rf.skip_reason = "too_large"
                db.add(rf)
                continue

            try:
                if download_url:
                    raw = await _github_get_bytes(client, download_url, headers)
                else:
                    j = await _github_get_json(client, api_url, headers)
                    if j.get("encoding") == "base64" and j.get("content"):
                        raw = base64.b64decode(j["content"])
                    else:
                        raw = b""

                if _looks_like_text(path, raw):
                    text = raw.decode("utf-8", errors="replace")
                    rf.content_kind = "text"
                    rf.content_text = text
                    rf.content = text
                    stored += 1
                else:
                    rf.is_text = False
                    rf.skipped = True
                    rf.content_kind = "binary"
                    rf.skip_reason = "binary"

            except Exception as e:
                rf.skipped = True
                rf.content_kind = "skipped"
                rf.skip_reason = f"fetch_error:{type(e).__name__}"

            db.add(rf)

    snap.file_count = total
    snap.stored_content_files = stored
    snap.warnings_json = json.dumps(warnings) if warnings else None
    db.commit()

    return RepoSyncResult(
        snapshot_id=snap.id,
        repo=repo,
        branch=branch,
        commit_sha=commit_sha,
        file_count=total,
        stored_content_files=stored,
        warnings=warnings,
    )


# -------------------------------------------------------------------
# Supported exported sync functions (THIS fixes your "does not export"
# error without changing your internal implementation).
# -------------------------------------------------------------------

async def sync_repo_snapshot(db: Session, repo: str | None = None, branch: str | None = None) -> dict[str, Any]:
    r = await create_repo_snapshot(db, repo=repo, branch=branch)
    return {
        "snapshot_id": r.snapshot_id,
        "repo": r.repo,
        "branch": r.branch,
        "commit_sha": r.commit_sha,
        "file_count": r.file_count,
        "stored_content_files": r.stored_content_files,
        "warnings": r.warnings,
    }


async def sync_snapshot(db: Session, repo: str | None = None, branch: str | None = None) -> dict[str, Any]:
    return await sync_repo_snapshot(db, repo=repo, branch=branch)


async def sync_repo(db: Session, repo: str | None = None, branch: str | None = None) -> dict[str, Any]:
    return await sync_repo_snapshot(db, repo=repo, branch=branch)
