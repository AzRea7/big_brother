# backend/app/services/github_sync.py
from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoFile, RepoSnapshot


@dataclass
class SyncResult:
    snapshot_id: int
    repo: str
    branch: str
    file_count: int
    stored_content_files: int
    warnings: list[str]


def _norm_path(p: str) -> str:
    return (p or "").lstrip("/")


def _is_excluded_path(path: str) -> bool:
    p = _norm_path(path)

    # 1) Exclude by directory name anywhere in the path
    # This kills __pycache__, .venv, etc no matter where they appear
    parts = p.split("/")
    if any(seg in set(settings.GITHUB_EXCLUDE_DIR_NAMES) for seg in parts):
        return True

    # 2) Exclude by file extension
    if "." in p:
        ext = p.rsplit(".", 1)[-1].lower()
        if ext in {x.lower() for x in settings.GITHUB_EXCLUDE_EXTENSIONS}:
            return True

    # 3) Exclude by path prefix (repo-relative, boundary-aware)
    for pref in settings.GITHUB_EXCLUDE_PREFIXES:
        pref_n = _norm_path(pref).rstrip("/")
        if not pref_n:
            continue

        if p == pref_n or p.startswith(pref_n + "/"):
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


async def _get_json(client: httpx.AsyncClient, url: str) -> Any:
    r = await client.get(url)

    if r.status_code in (403, 404, 429):
        try:
            detail = r.json()
        except Exception:
            detail = {"raw": r.text[:500]}

        msg = {
            "status": r.status_code,
            "url": url,
            "detail": detail,
            "x_ratelimit_limit": r.headers.get("x-ratelimit-limit"),
            "x_ratelimit_remaining": r.headers.get("x-ratelimit-remaining"),
            "x_ratelimit_reset": r.headers.get("x-ratelimit-reset"),
            "retry_after": r.headers.get("retry-after"),
            "x_github_request_id": r.headers.get("x-github-request-id"),
        }
        raise httpx.HTTPStatusError(json.dumps(msg), request=r.request, response=r)

    r.raise_for_status()
    return r.json()



async def _get_repo_root_listing(client: httpx.AsyncClient, repo: str, path: str, ref: str) -> Any:
    base = "https://api.github.com"
    path_part = path.strip("/")
    if path_part:
        return await _get_json(client, f"{base}/repos/{repo}/contents/{path_part}?ref={ref}")
    return await _get_json(client, f"{base}/repos/{repo}/contents?ref={ref}")


async def _fetch_file_content(
    client: httpx.AsyncClient,
    repo: str,
    path: str,
    ref: str,
) -> tuple[str | None, str, int | None]:
    """
    Returns (content_text_or_none, content_kind, size)
    content_kind: "text" | "binary" | "skipped"
    """
    base = "https://api.github.com"
    data = await _get_json(client, f"{base}/repos/{repo}/contents/{path}?ref={ref}")

    size = data.get("size")
    if size is not None and size > settings.GITHUB_MAX_FILE_BYTES:
        return None, "skipped", int(size)

    enc = (data.get("encoding") or "").lower()
    if enc != "base64":
        return None, "skipped", int(size) if size is not None else None

    b64 = data.get("content") or ""
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception:
        return None, "skipped", int(size) if size is not None else None

    if not _looks_textual(path, raw):
        return None, "binary", int(size) if size is not None else None

    text = raw.decode("utf-8", errors="replace")
    return text, "text", int(size) if size is not None else None


async def create_repo_snapshot(db: Session, repo: str | None = None, branch: str | None = None) -> SyncResult:
    repo = (repo or settings.GITHUB_REPO).strip()
    branch = (branch or settings.GITHUB_BRANCH).strip()

    warnings: list[str] = []

    timeout = httpx.Timeout(
        connect=settings.GITHUB_CONNECT_TIMEOUT_S,
        read=settings.GITHUB_READ_TIMEOUT_S,
        write=30.0,
        pool=10.0,
    )

    async with httpx.AsyncClient(headers=_github_headers(), timeout=timeout) as client:
        # BFS crawl over directories
        queue: list[str] = [""]
        files: list[dict[str, Any]] = []

        excluded_paths = 0

        while queue:
            cur = queue.pop(0)
            listing = await _get_repo_root_listing(client, repo=repo, path=cur, ref=branch)

            if isinstance(listing, dict) and listing.get("type") == "file":
                # Rare: contents API can return a file object if you request a file path
                if _is_excluded_path(listing.get("path", "") or ""):
                    excluded_paths += 1
                else:
                    files.append(listing)
                continue

            if not isinstance(listing, list):
                warnings.append(f"Unexpected listing at '{cur}': {type(listing)}")
                continue

            for item in listing:
                t = item.get("type")
                item_path = item.get("path", "") or ""

                # Skip excluded paths early (prevents crawling huge dirs like node_modules)
                if _is_excluded_path(item_path):
                    excluded_paths += 1
                    continue

                if t == "dir":
                    queue.append(item_path)
                elif t == "file":
                    files.append(item)

            if len(files) + len(queue) > settings.GITHUB_MAX_FILES_PER_SYNC:
                warnings.append(f"Hit max file limit ({settings.GITHUB_MAX_FILES_PER_SYNC}); truncating crawl.")
                break

        if excluded_paths:
            warnings.append(f"Excluded {excluded_paths} paths via GITHUB_EXCLUDE_* rules.")

        # Create snapshot row
        snap = RepoSnapshot(
            repo=repo,
            branch=branch,
            commit_sha=None,
            file_count=len(files),
            stored_content_files=0,
            warnings_json=json.dumps(warnings) if warnings else None,
        )
        db.add(snap)
        db.commit()
        db.refresh(snap)

        stored = 0
        batch = 0
        excluded_store = 0

        for item in files:
            path = item.get("path") or ""
            if _is_excluded_path(path):
                excluded_store += 1
                continue

            sha = item.get("sha")
            size = item.get("size")

            content, kind, final_size = None, "skipped", int(size) if isinstance(size, int) else None
            try:
                content, kind, final_size = await _fetch_file_content(client, repo=repo, path=path, ref=branch)
            except httpx.HTTPStatusError as e:
                warnings.append(f"Failed fetch {path}: HTTP {e.response.status_code}")
                kind = "skipped"
            except Exception as e:
                warnings.append(f"Failed fetch {path}: {e}")
                kind = "skipped"

            if kind == "text" and content is not None:
                stored += 1

            try:
                db.add(
                    RepoFile(
                        snapshot_id=snap.id,
                        path=path,
                        sha=sha,
                        size=final_size,
                        content=content,
                        content_kind=kind,
                    )
                )
            except Exception as e:
                warnings.append(f"DB add failed for {path}: {e}")

            batch += 1
            if batch >= 200:
                try:
                    db.commit()
                except Exception as e:
                    db.rollback()
                    warnings.append(f"DB commit failed after batch: {e}")
                batch = 0

        # final commit
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            warnings.append(f"DB final commit failed: {e}")

        if excluded_store:
            warnings.append(f"Excluded {excluded_store} files during store pass (safety check).")

        snap.stored_content_files = stored
        if warnings:
            snap.warnings_json = json.dumps(warnings)
        db.commit()

        return SyncResult(
            snapshot_id=snap.id,
            repo=repo,
            branch=branch,
            file_count=len(files),
            stored_content_files=stored,
            warnings=warnings,
        )


def latest_snapshot(db: Session, repo: str | None = None, branch: str | None = None) -> RepoSnapshot | None:
    repo = (repo or settings.GITHUB_REPO).strip()
    branch = (branch or settings.GITHUB_BRANCH).strip()

    return db.scalars(
        select(RepoSnapshot)
        .where(RepoSnapshot.repo == repo)
        .where(RepoSnapshot.branch == branch)
        .order_by(RepoSnapshot.created_at.desc())
        .limit(1)
    ).first()


def snapshot_file_stats(db: Session, snapshot_id: int) -> dict[str, Any]:
    files = db.scalars(select(RepoFile).where(RepoFile.snapshot_id == snapshot_id)).all()

    total = len(files)
    text = sum(1 for f in files if f.content_kind == "text")
    binary = sum(1 for f in files if f.content_kind == "binary")
    skipped = sum(1 for f in files if f.content_kind == "skipped")

    folder_counts: dict[str, int] = {}
    for f in files:
        parts = f.path.split("/")
        folder = parts[0] if parts else ""
        if folder:
            folder_counts[folder] = folder_counts.get(folder, 0) + 1

    top_folders = sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)[:12]

    return {
        "total_files": total,
        "text_files": text,
        "binary_files": binary,
        "skipped_files": skipped,
        "top_folders": [{"folder": k, "count": v} for k, v in top_folders],
    }
