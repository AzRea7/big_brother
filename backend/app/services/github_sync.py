from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

import httpx
from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoFile, RepoSnapshot


@dataclass(frozen=True)
class SnapshotStats:
    total_files: int
    text_files: int
    binary_files: int
    skipped_files: int
    top_folders: list[dict[str, Any]]


def _json_list(x: list[str] | None) -> str | None:
    if not x:
        return None
    return json.dumps(x)


def latest_snapshot(db: Session, repo: str | None = None, branch: str | None = None) -> RepoSnapshot | None:
    q = db.query(RepoSnapshot)
    if repo:
        q = q.filter(RepoSnapshot.repo == repo)
    if branch:
        q = q.filter(RepoSnapshot.branch == branch)
    return q.order_by(desc(RepoSnapshot.id)).first()


def _excluded_path(path: str) -> bool:
    p = path.replace("\\", "/")

    for pref in settings.GITHUB_EXCLUDE_PREFIXES:
        if p.startswith(pref):
            return True

    # extension check
    if "." in p:
        ext = p.rsplit(".", 1)[-1].lower()
        if ext in set(settings.GITHUB_EXCLUDE_EXTENSIONS):
            return True

    # directory name check
    parts = [x for x in p.split("/") if x]
    for dn in settings.GITHUB_EXCLUDE_DIR_NAMES:
        if dn in parts:
            return True

    return False


def _folder_counts(paths: Iterable[str]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for p in paths:
        folder = p.split("/", 1)[0] if "/" in p else p
        counts[folder] = counts.get(folder, 0) + 1
    return [{"folder": k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:12]]


async def _github_api_json(url: str) -> Any:
    headers = {"Accept": "application/vnd.github+json"}
    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"

    timeout = httpx.Timeout(connect=settings.GITHUB_CONNECT_TIMEOUT_S, read=settings.GITHUB_READ_TIMEOUT_S)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.json()


async def _github_api_bytes(url: str) -> bytes:
    headers = {"Accept": "application/vnd.github+json"}
    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"

    timeout = httpx.Timeout(connect=settings.GITHUB_CONNECT_TIMEOUT_S, read=settings.GITHUB_READ_TIMEOUT_S)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.content


async def create_snapshot(db: Session, *, repo: str, branch: str) -> tuple[RepoSnapshot, SnapshotStats]:
    """
    Creates a RepoSnapshot + RepoFile rows.
    Uses either:
      - settings.REPO_LOCAL_PATH (fast, no GitHub token), OR
      - GitHub REST API (requires token for private repos)
    """

    warnings: list[str] = []

    snap = RepoSnapshot(repo=repo, branch=branch, commit_sha=None, file_count=0, stored_content_files=0,
                        warnings_json=_json_list([]))
    db.add(snap)
    db.commit()
    db.refresh(snap)

    # ---- Local scan mode ----
    if settings.REPO_LOCAL_PATH:
        root = settings.REPO_LOCAL_PATH
        if not os.path.isdir(root):
            warnings.append(f"REPO_LOCAL_PATH not found: {root}")
            snap.warnings_json = _json_list(warnings)
            db.commit()
            return snap, SnapshotStats(0, 0, 0, 0, [])

        file_paths: list[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root).replace("\\", "/")
                if _excluded_path(rel):
                    continue
                file_paths.append(rel)
                if len(file_paths) >= settings.GITHUB_MAX_FILES_PER_SYNC:
                    warnings.append(f"Hit GITHUB_MAX_FILES_PER_SYNC={settings.GITHUB_MAX_FILES_PER_SYNC} in local scan.")
                    break

        stored = 0
        for p in file_paths:
            full = os.path.join(root, p)
            try:
                b = open(full, "rb").read()
            except Exception:
                db.add(RepoFile(snapshot_id=snap.id, path=p, sha=None, size=None, content=None,
                                content_kind="skipped", skipped=True, is_text=True))
                continue

            if len(b) > settings.GITHUB_MAX_FILE_BYTES:
                db.add(RepoFile(snapshot_id=snap.id, path=p, sha=None, size=len(b), content=None,
                                content_kind="skipped", skipped=True, is_text=True))
                continue

            try:
                text = b.decode("utf-8", errors="strict")
                db.add(RepoFile(snapshot_id=snap.id, path=p, sha=None, size=len(b), content=text,
                                content_kind="text", skipped=False, is_text=True))
                stored += 1
            except Exception:
                db.add(RepoFile(snapshot_id=snap.id, path=p, sha=None, size=len(b), content=None,
                                content_kind="binary", skipped=True, is_text=False))

        snap.file_count = len(file_paths)
        snap.stored_content_files = stored
        snap.warnings_json = _json_list(warnings)
        db.commit()

        stats = SnapshotStats(
            total_files=len(file_paths),
            text_files=stored,
            binary_files=len(file_paths) - stored,
            skipped_files=len(file_paths) - stored,
            top_folders=_folder_counts(file_paths),
        )
        return snap, stats

    # ---- GitHub API mode ----
    # List tree
    tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    tree = await _github_api_json(tree_url)

    entries = tree.get("tree") or []
    paths: list[dict[str, Any]] = []
    excluded = 0

    for e in entries:
        if e.get("type") != "blob":
            continue
        path = e.get("path") or ""
        if not path:
            continue
        if _excluded_path(path):
            excluded += 1
            continue
        paths.append(e)
        if len(paths) >= settings.GITHUB_MAX_FILES_PER_SYNC:
            warnings.append(f"Hit GITHUB_MAX_FILES_PER_SYNC={settings.GITHUB_MAX_FILES_PER_SYNC}.")
            break

    if excluded:
        warnings.append(f"Excluded {excluded} paths via GITHUB_EXCLUDE_* rules.")

    stored = 0
    for e in paths:
        path = e.get("path")
        sha = e.get("sha")
        size = e.get("size")

        # fetch blob
        try:
            blob = await _github_api_json(f"https://api.github.com/repos/{repo}/git/blobs/{sha}")
            enc = blob.get("encoding")
            content = blob.get("content") or ""
            raw = base64.b64decode(content) if enc == "base64" else content.encode("utf-8", errors="ignore")
        except Exception:
            db.add(RepoFile(snapshot_id=snap.id, path=path, sha=sha, size=size, content=None,
                            content_kind="skipped", skipped=True, is_text=True))
            continue

        if size and size > settings.GITHUB_MAX_FILE_BYTES:
            db.add(RepoFile(snapshot_id=snap.id, path=path, sha=sha, size=size, content=None,
                            content_kind="skipped", skipped=True, is_text=True))
            continue

        try:
            text = raw.decode("utf-8", errors="strict")
            db.add(RepoFile(snapshot_id=snap.id, path=path, sha=sha, size=size, content=text,
                            content_kind="text", skipped=False, is_text=True))
            stored += 1
        except Exception:
            db.add(RepoFile(snapshot_id=snap.id, path=path, sha=sha, size=size, content=None,
                            content_kind="binary", skipped=True, is_text=False))

    snap.file_count = len(paths)
    snap.stored_content_files = stored
    snap.warnings_json = _json_list(warnings)
    db.commit()

    stats = SnapshotStats(
        total_files=len(paths),
        text_files=stored,
        binary_files=len(paths) - stored,
        skipped_files=len(paths) - stored,
        top_folders=_folder_counts([p["path"] for p in paths if p.get("path")]),
    )
    return snap, stats
