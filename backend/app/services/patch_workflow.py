# backend/app/services/patch_workflow.py
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoPatchRun, RepoPullRequest, RepoSnapshot
from .repo_materialize import materialize_snapshot_to_disk

# Matches: diff --git a/path b/path
_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$", re.MULTILINE)


def patch_workflow_enabled() -> bool:
    # Patch apply is its own dangerous capability. Keep it independently gated.
    return bool(getattr(settings, "ENABLE_PATCH_WORKFLOW", False))


def pr_workflow_enabled() -> bool:
    return bool(getattr(settings, "ENABLE_PR_WORKFLOW", False))


def pr_workflow_dry_run() -> bool:
    return bool(getattr(settings, "PR_WORKFLOW_DRY_RUN", True))


def _github_token() -> str:
    tok = str(getattr(settings, "GITHUB_TOKEN", "") or "").strip()
    if not tok:
        raise RuntimeError("GITHUB_TOKEN is required to open PRs.")
    return tok


def _pr_allowlist_dirs() -> list[str]:
    dirs = getattr(settings, "PR_ALLOWLIST_DIRS", None)
    if isinstance(dirs, list) and dirs:
        return [str(d).strip() for d in dirs if str(d).strip()]
    return []


def _reject_path(path: str) -> Optional[str]:
    p = (path or "").strip()
    if not p:
        return "Empty path in diff."
    if p.startswith("/") or p.startswith("\\"):
        return f"Absolute paths not allowed: {p}"
    if ".." in p.replace("\\", "/").split("/"):
        return f"Path traversal not allowed: {p}"
    return None


def _extract_paths(patch_text: str) -> list[str]:
    out: list[str] = []
    for m in _DIFF_HEADER_RE.finditer(patch_text or ""):
        a_path = m.group(1).strip()
        b_path = m.group(2).strip()
        if a_path and a_path not in out:
            out.append(a_path)
        if b_path and b_path not in out:
            out.append(b_path)
    return out


def _count_changed_lines(patch_text: str) -> int:
    s = (patch_text or "").splitlines()
    n = 0
    for line in s:
        if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            n += 1
    return n


def validate_unified_diff(db: Session, snapshot_id: int, patch_text: str) -> dict[str, object]:
    """
    Validates unified diff content and returns:
      { valid, error, files_changed, lines_changed, file_paths }
    """
    s = (patch_text or "").strip()
    if not s:
        return {"valid": False, "error": "Empty patch_text.", "files_changed": 0, "lines_changed": 0, "file_paths": []}

    matches = list(_DIFF_HEADER_RE.finditer(s))
    if not matches:
        return {
            "valid": False,
            "error": "Patch missing 'diff --git a/... b/...'.",
            "files_changed": 0,
            "lines_changed": 0,
            "file_paths": [],
        }

    allow_dirs = _pr_allowlist_dirs()
    max_files = int(getattr(settings, "PR_MAX_FILES_CHANGED", 8) or 8)
    max_lines = int(getattr(settings, "PR_MAX_LINES_CHANGED", 400) or 400)

    file_paths: list[str] = []
    files_changed = 0

    for m in matches:
        a_path = m.group(1).strip()
        b_path = m.group(2).strip()

        err = _reject_path(a_path) or _reject_path(b_path)
        if err:
            return {"valid": False, "error": err, "files_changed": 0, "lines_changed": 0, "file_paths": []}

        # allowlist: only allow touching safe subtrees
        if allow_dirs:
            ok = any(a_path.startswith(d) or b_path.startswith(d) for d in allow_dirs)
            if not ok:
                return {
                    "valid": False,
                    "error": f"Path not in allowlist dirs: {a_path} / {b_path}",
                    "files_changed": 0,
                    "lines_changed": 0,
                    "file_paths": [],
                }

        files_changed += 1
        if a_path not in file_paths:
            file_paths.append(a_path)
        if b_path not in file_paths:
            file_paths.append(b_path)

    if files_changed > max_files:
        return {
            "valid": False,
            "error": f"Too many files changed: {files_changed} > {max_files}",
            "files_changed": files_changed,
            "lines_changed": 0,
            "file_paths": file_paths,
        }

    lines_changed = _count_changed_lines(s)
    if lines_changed > max_lines:
        return {
            "valid": False,
            "error": f"Too many changed lines: {lines_changed} > {max_lines}",
            "files_changed": files_changed,
            "lines_changed": lines_changed,
            "file_paths": file_paths,
        }

    # Snapshot existence check (prevents FK explosions later)
    snap = db.query(RepoSnapshot).filter(RepoSnapshot.id == snapshot_id).first()
    if not snap:
        return {"valid": False, "error": f"Unknown snapshot_id={snapshot_id}", "files_changed": 0, "lines_changed": 0, "file_paths": []}

    return {
        "valid": True,
        "error": None,
        "files_changed": files_changed,
        "lines_changed": lines_changed,
        "file_paths": file_paths,
    }


@dataclass(frozen=True)
class _CmdResult:
    ok: bool
    out: str


def _run(cmd: str, *, cwd: str, timeout_s: int, input_text: Optional[str] = None) -> _CmdResult:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
            text=True,
        )
        out = (p.stdout or "").strip()
        return _CmdResult(ok=(p.returncode == 0), out=out)
    except subprocess.TimeoutExpired:
        return _CmdResult(ok=False, out=f"Command timed out after {timeout_s}s: {cmd}")


async def apply_unified_diff_in_sandbox(
    db: Session,
    snapshot_id: int,
    patch_text: str,
    *,
    run_tests: bool = True,
) -> RepoPatchRun:
    """
    Apply patch to a materialized snapshot (sandbox dir), run tests, persist a RepoPatchRun row.
    """
    if not patch_workflow_enabled():
        raise RuntimeError("Patch workflow disabled (ENABLE_PATCH_WORKFLOW=false).")

    v = validate_unified_diff(db, snapshot_id, patch_text)
    valid = bool(v.get("valid"))
    validation_error = v.get("error")

    # IMPORTANT: only insert a RepoPatchRun after we know snapshot exists (validator checks)
    run = RepoPatchRun(
        snapshot_id=snapshot_id,
        created_at=datetime.utcnow(),
        patch_text=patch_text,
        files_changed=int(v.get("files_changed") or 0),
        lines_changed=int(v.get("lines_changed") or 0),
        file_paths_json=json.dumps(v.get("file_paths") or []),
        valid=bool(valid),
        validation_error=str(validation_error) if validation_error else None,
        applied=False,
        apply_error=None,
        tests_ran=False,
        tests_ok=False,
        test_output=None,
        sandbox_path=None,
        diff_output=None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    if not valid:
        return run

    tmpdir = tempfile.mkdtemp(prefix=f"patchrun_{snapshot_id}_{run.id}_")
    keep = bool(getattr(settings, "KEEP_PATCH_SANDBOX", False))

    try:
        repo_root = materialize_snapshot_to_disk(db=db, snapshot_id=snapshot_id, dest_dir=tmpdir)

        # Initialize git so `git apply` behaves consistently
        _run("git init", cwd=repo_root, timeout_s=30)
        _run("git add -A", cwd=repo_root, timeout_s=30)
        _run('git commit -m "snapshot baseline" --no-gpg-sign', cwd=repo_root, timeout_s=60)

        # Apply patch (via stdin)
        apply_res = _run("git apply --whitespace=nowarn -", cwd=repo_root, timeout_s=60, input_text=patch_text)
        if not apply_res.ok:
            run.applied = False
            run.apply_error = (apply_res.out or "git apply failed")[:20000]
            db.commit()
            return run

        run.applied = True
        run.apply_error = None
        run.sandbox_path = repo_root
        db.commit()

        if run_tests:
            test_cmd = str(getattr(settings, "PR_TEST_CMD", "pytest -q") or "pytest -q")
            timeout_s = int(getattr(settings, "PR_TIMEOUT_SECONDS", 900) or 900)
            t = _run(test_cmd, cwd=repo_root, timeout_s=timeout_s)

            run.tests_ran = True
            run.tests_ok = bool(t.ok)
            run.test_output = (t.out or "")[:20000] if t.out else None
            db.commit()

        return run

    finally:
        if not keep:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_owner_repo(snapshot: RepoSnapshot) -> tuple[str, str]:
    repo = str(getattr(snapshot, "repo", "") or "").strip()
    if "/" in repo:
        owner, name = repo.split("/", 1)
        return owner.strip(), name.strip()

    owner = str(getattr(settings, "GITHUB_OWNER", "") or "").strip()
    name = str(getattr(settings, "GITHUB_REPO", "") or "").strip()
    if owner and name:
        return owner, name

    raise RuntimeError("Cannot determine GitHub owner/repo. Set snapshot.repo to 'owner/name' or set GITHUB_OWNER/GITHUB_REPO.")


def _clone_for_pr(owner: str, repo: str, base_branch: str) -> str:
    tok = _github_token()
    url = f"https://x-access-token:{tok}@github.com/{owner}/{repo}.git"
    tmpdir = tempfile.mkdtemp(prefix=f"prclone_{owner}_{repo}_")

    r = _run(f"git clone --depth 1 --branch {base_branch} {url} .", cwd=tmpdir, timeout_s=300)
    if not r.ok:
        raise RuntimeError(f"git clone failed: {r.out}")
    return tmpdir


def _create_branch_name(patch_run_id: int) -> str:
    return f"goal-autopilot/patchrun-{patch_run_id}"


async def open_pull_request_from_patch_run(
    db: Session,
    snapshot_id: int,
    patch_run_id: int,
    *,
    title: str,
    body: Optional[str] = None,
    base_branch: Optional[str] = None,
) -> dict[str, object]:
    """
    Opens a PR from an existing successful RepoPatchRun.

    Returns: { ok, pr_url, pr_number, error, branch }
    """
    if not pr_workflow_enabled():
        return {"ok": False, "pr_url": None, "pr_number": None, "error": "PR workflow disabled (ENABLE_PR_WORKFLOW=false).", "branch": None}

    if pr_workflow_dry_run():
        return {"ok": False, "pr_url": None, "pr_number": None, "error": "PR workflow dry-run enabled (PR_WORKFLOW_DRY_RUN=true). Set false to open PRs.", "branch": None}

    snap = db.query(RepoSnapshot).filter(RepoSnapshot.id == snapshot_id).first()
    if not snap:
        return {"ok": False, "pr_url": None, "pr_number": None, "error": f"Unknown snapshot_id={snapshot_id}", "branch": None}

    run = db.query(RepoPatchRun).filter(RepoPatchRun.id == patch_run_id).first()
    if not run or int(getattr(run, "snapshot_id", 0) or 0) != int(snapshot_id):
        return {"ok": False, "pr_url": None, "pr_number": None, "error": "Patch run not found for this snapshot.", "branch": None}

    if not run.valid or not run.applied or (run.tests_ran and not run.tests_ok):
        return {"ok": False, "pr_url": None, "pr_number": None, "error": "Patch run is not green (valid+applied+tests_ok required).", "branch": None}

    owner, repo = _parse_owner_repo(snap)
    base = base_branch or str(getattr(snap, "branch", "") or "main")
    branch = _create_branch_name(int(run.id))

    keep = bool(getattr(settings, "KEEP_PR_SANDBOX", False))
    clone_dir: Optional[str] = None

    try:
        clone_dir = _clone_for_pr(owner, repo, base)

        r = _run(f"git checkout -b {branch}", cwd=clone_dir, timeout_s=60)
        if not r.ok:
            raise RuntimeError(f"git checkout -b failed: {r.out}")

        # apply patch
        a = _run("git apply --whitespace=nowarn -", cwd=clone_dir, timeout_s=60, input_text=str(run.patch_text))
        if not a.ok:
            raise RuntimeError(a.out or "git apply failed in PR clone")

        # run tests again in the real checkout
        test_cmd = str(getattr(settings, "PR_TEST_CMD", "pytest -q") or "pytest -q")
        timeout_s = int(getattr(settings, "PR_TIMEOUT_SECONDS", 900) or 900)
        t = _run(test_cmd, cwd=clone_dir, timeout_s=timeout_s)
        if not t.ok:
            raise RuntimeError(f"Tests failed in PR clone:\n{t.out}")

        # commit + push
        _run("git add -A", cwd=clone_dir, timeout_s=30)

        # json.dumps handles quoting safely for shell
        c = _run(f"git commit -m {json.dumps(title)} --no-gpg-sign", cwd=clone_dir, timeout_s=60)
        if not c.ok:
            raise RuntimeError(f"git commit failed: {c.out}")

        sha = _run("git rev-parse HEAD", cwd=clone_dir, timeout_s=30).out.strip()[:64]
        if sha:
            setattr(run, "commit_sha", sha)

        push = _run(f"git push -u origin {branch}", cwd=clone_dir, timeout_s=180)
        if not push.ok:
            raise RuntimeError(f"git push failed: {push.out}")

        # open PR via GitHub API
        tok = _github_token()
        api = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        headers = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
        payload = {"title": title, "head": branch, "base": base, "body": body or ""}

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(api, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(f"GitHub PR create failed {resp.status_code}: {resp.text}")
            data = resp.json()

        pr_url = str(data.get("html_url") or "")
        pr_number = int(data.get("number") or 0) or None

        # persist on run (if your model has these columns)
        if hasattr(run, "pr_url"):
            run.pr_url = pr_url or None
        if hasattr(run, "pr_number"):
            run.pr_number = pr_number
        if hasattr(run, "pr_branch"):
            run.pr_branch = branch
        db.commit()

        # persist a PR row (if your model/table exists)
        try:
            db.add(
                RepoPullRequest(
                    snapshot_id=snapshot_id,
                    patch_run_id=int(run.id),
                    pr_url=pr_url,
                    pr_number=pr_number,
                    created_at=datetime.utcnow(),
                )
            )
            db.commit()
        except Exception:
            db.rollback()

        return {"ok": True, "pr_url": pr_url, "pr_number": pr_number, "error": None, "branch": branch}

    except Exception as e:
        # best-effort error persistence (don’t crash imports/routes)
        try:
            if hasattr(run, "pr_error"):
                run.pr_error = str(e)[:2000]
            db.commit()
        except Exception:
            db.rollback()
        return {"ok": False, "pr_url": None, "pr_number": None, "error": str(e), "branch": branch}

    finally:
        if clone_dir and not keep:
            shutil.rmtree(clone_dir, ignore_errors=True)
