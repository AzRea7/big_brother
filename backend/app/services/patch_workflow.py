# backend/app/services/patch_workflow.py
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Tuple

import httpx
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoPatchRun, RepoPullRequest, RepoSnapshot


_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$", re.MULTILINE)


def patch_workflow_enabled() -> bool:
    return bool(getattr(settings, "PATCH_WORKFLOW_ENABLED", False))


def pr_workflow_enabled() -> bool:
    return bool(getattr(settings, "PR_WORKFLOW_ENABLED", False))


def github_token() -> str:
    tok = str(getattr(settings, "GITHUB_TOKEN", "") or "")
    if not tok:
        raise RuntimeError("GITHUB_TOKEN is required to open PRs.")
    return tok


def _allowlist_prefixes() -> list[str]:
    raw = str(getattr(settings, "PATCH_ALLOWLIST_PREFIXES", "") or "").strip()
    if not raw:
        # default: allow everything (but still blocks absolute paths + .. traversal)
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _reject_path(path: str) -> Optional[str]:
    p = (path or "").strip()
    if not p:
        return "Empty path in diff."
    if p.startswith("/") or p.startswith("\\"):
        return f"Absolute paths not allowed: {p}"
    if ".." in p.replace("\\", "/").split("/"):
        return f"Path traversal not allowed: {p}"
    return None


def validate_patch_text(patch_text: str) -> tuple[bool, Optional[str], int, int]:
    """
    Validates:
    - has diff headers
    - no absolute paths / traversal
    - optional allowlist prefixes
    Also returns:
    - files_changed
    - lines_changed (rough, from +/- lines not in headers)
    """
    s = (patch_text or "").strip()
    if not s:
        return False, "Empty patch_text.", 0, 0

    matches = list(_DIFF_HEADER_RE.finditer(s))
    if not matches:
        return False, "Patch does not look like unified diff (missing 'diff --git a/... b/...').", 0, 0

    allow = _allowlist_prefixes()

    files_changed = 0
    for m in matches:
        a_path = m.group(1).strip()
        b_path = m.group(2).strip()

        err = _reject_path(a_path) or _reject_path(b_path)
        if err:
            return False, err, 0, 0

        # enforce allowlist if provided
        if allow:
            if not any(a_path.startswith(p) or b_path.startswith(p) for p in allow):
                return False, f"Path not in allowlist: {a_path} / {b_path}", 0, 0

        files_changed += 1

    # lines changed: count +/- lines excluding diff metadata
    lines_changed = 0
    for line in s.splitlines():
        if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            lines_changed += 1

    max_files = int(getattr(settings, "PATCH_MAX_FILES", 25))
    max_lines = int(getattr(settings, "PATCH_MAX_LINES", 2500))
    if files_changed > max_files:
        return False, f"Too many files changed ({files_changed} > {max_files}).", files_changed, lines_changed
    if lines_changed > max_lines:
        return False, f"Too many changed lines ({lines_changed} > {max_lines}).", files_changed, lines_changed

    return True, None, files_changed, lines_changed


def _repo_local_path() -> str:
    p = str(getattr(settings, "REPO_LOCAL_PATH", "") or "").strip()
    return p


def _tests_cmd() -> str:
    return str(getattr(settings, "PATCH_TEST_CMD", "") or "pytest -q")


def _run(cmd: list[str], *, cwd: str, timeout_s: int = 900) -> tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    return int(p.returncode), p.stdout


def apply_patch_in_sandbox(
    db: Session,
    *,
    snapshot_id: int,
    patch_text: str,
    run_tests: bool = True,
) -> RepoPatchRun:
    """
    Creates RepoPatchRun row, validates patch, copies REPO_LOCAL_PATH into sandbox,
    applies patch using `git apply`, optionally runs tests.

    Important: This expects REPO_LOCAL_PATH to point to a git checkout of the target repo.
    """
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        raise RuntimeError(f"Snapshot not found: {snapshot_id}")

    run = RepoPatchRun(
        snapshot_id=snapshot_id,
        patch_text=patch_text,
        created_at=datetime.utcnow(),
        valid=False,
        applied=False,
        tests_ran=False,
        tests_ok=False,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    ok, err, _, _ = validate_patch_text(patch_text)
    run.valid = bool(ok)
    run.validation_error = err
    db.commit()

    if not ok:
        return run

    if not patch_workflow_enabled():
        run.applied = False
        run.apply_error = "PATCH_WORKFLOW_ENABLED=false (apply is gated for safety)."
        db.commit()
        return run

    base = _repo_local_path()
    if not base or not os.path.isdir(base):
        run.applied = False
        run.apply_error = "REPO_LOCAL_PATH is not set or does not exist. Patch apply requires a real git checkout."
        db.commit()
        return run

    sandbox_root = str(getattr(settings, "PATCH_SANDBOX_ROOT", "") or "data/patch_sandboxes")
    os.makedirs(sandbox_root, exist_ok=True)

    sandbox_dir = os.path.join(sandbox_root, f"run_{run.id}")
    if os.path.isdir(sandbox_dir):
        shutil.rmtree(sandbox_dir, ignore_errors=True)

    # Copy repo into sandbox (simple + deterministic)
    shutil.copytree(
        base,
        sandbox_dir,
        ignore=shutil.ignore_patterns(".venv", "node_modules", ".pytest_cache", "__pycache__", ".mypy_cache"),
        dirs_exist_ok=False,
    )

    run.sandbox_path = sandbox_dir
    db.commit()

    # Ensure it's a git repo
    if not os.path.isdir(os.path.join(sandbox_dir, ".git")):
        run.applied = False
        run.apply_error = "Sandbox copy is not a git repo (missing .git). Ensure REPO_LOCAL_PATH points to a git checkout."
        db.commit()
        return run

    # Write patch to temp file
    patch_file = os.path.join(sandbox_dir, ".autopatch.diff")
    with open(patch_file, "w", encoding="utf-8") as f:
        f.write(patch_text)

    # Check + apply
    rc, out = _run(["git", "apply", "--check", patch_file], cwd=sandbox_dir, timeout_s=120)
    if rc != 0:
        run.applied = False
        run.apply_error = f"git apply --check failed:\n{out}"
        db.commit()
        return run

    rc, out = _run(["git", "apply", "--whitespace=nowarn", patch_file], cwd=sandbox_dir, timeout_s=120)
    if rc != 0:
        run.applied = False
        run.apply_error = f"git apply failed:\n{out}"
        db.commit()
        return run

    run.applied = True
    run.apply_error = None
    db.commit()

    if run_tests:
        run.tests_ran = True
        cmd = _tests_cmd().strip()
        # allow either "pytest -q" string or JSON list (power users)
        if cmd.startswith("["):
            try:
                cmd_list = list(__import__("json").loads(cmd))
                cmd_list = [str(x) for x in cmd_list]
            except Exception:
                cmd_list = ["pytest", "-q"]
        else:
            cmd_list = cmd.split()

        timeout_s = int(getattr(settings, "PATCH_TEST_TIMEOUT_S", 900))
        rc, out = _run(cmd_list, cwd=sandbox_dir, timeout_s=timeout_s)
        run.tests_ok = (rc == 0)
        run.test_output = out[-25000:]  # cap
        db.commit()

    return run


async def open_pull_request(
    db: Session,
    *,
    run_id: int,
    title: str,
    body: Optional[str] = None,
    base_branch: Optional[str] = None,
) -> tuple[bool, Optional[int], Optional[str], Optional[str]]:
    """
    Requires:
      - PR_WORKFLOW_ENABLED=true
      - GITHUB_TOKEN set
      - RepoPatchRun.applied=true
      - RepoPatchRun.tests_ok=true (optional gate; can relax via settings)
    """
    if not pr_workflow_enabled():
        return False, None, None, "PR_WORKFLOW_ENABLED=false (PR creation gated)."

    run = db.get(RepoPatchRun, run_id)
    if not run:
        return False, None, None, f"Patch run not found: {run_id}"
    if not run.applied:
        return False, None, None, "Patch was not applied. Apply first."
    if not run.sandbox_path or not os.path.isdir(run.sandbox_path):
        return False, None, None, "Sandbox path missing. Apply must create sandbox."

    require_tests_ok = bool(getattr(settings, "PATCH_REQUIRE_TESTS_OK", True))
    if require_tests_ok and not run.tests_ok:
        return False, None, None, "Tests did not pass (PATCH_REQUIRE_TESTS_OK=true)."

    snap = db.get(RepoSnapshot, run.snapshot_id)
    if not snap:
        return False, None, None, "Snapshot missing for run."

    sandbox = run.sandbox_path
    branch = (base_branch or snap.branch or "main").strip()
    new_branch = f"autopatch/run-{run.id}"

    # Create branch, commit, push
    rc, out = _run(["git", "checkout", "-b", new_branch], cwd=sandbox, timeout_s=60)
    if rc != 0:
        return False, None, None, f"git checkout failed:\n{out}"

    rc, out = _run(["git", "add", "-A"], cwd=sandbox, timeout_s=60)
    if rc != 0:
        return False, None, None, f"git add failed:\n{out}"

    rc, out = _run(["git", "commit", "-m", title], cwd=sandbox, timeout_s=60)
    if rc != 0:
        return False, None, None, f"git commit failed:\n{out}"

    # Ensure origin URL contains token (non-interactive)
    tok = github_token()
    # snap.repo like "AzRea7/OneHaven"
    repo_full = snap.repo.strip()
    if "/" not in repo_full:
        return False, None, None, f"Invalid repo format: {repo_full}"
    owner, repo = repo_full.split("/", 1)

    remote_url = f"https://{tok}@github.com/{owner}/{repo}.git"
    _run(["git", "remote", "set-url", "origin", remote_url], cwd=sandbox, timeout_s=30)

    rc, out = _run(["git", "push", "-u", "origin", new_branch], cwd=sandbox, timeout_s=180)
    if rc != 0:
        return False, None, None, f"git push failed:\n{out}"

    # Create PR via GitHub API
    api = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    headers = {
        "Authorization": f"token {tok}",
        "Accept": "application/vnd.github+json",
    }
    payload = {
        "title": title,
        "head": new_branch,
        "base": branch,
        "body": body or "",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(api, headers=headers, json=payload)
        if r.status_code >= 400:
            return False, None, None, f"GitHub PR create failed {r.status_code}: {r.text}"

        j = r.json()
        pr_number = int(j.get("number")) if j.get("number") is not None else None
        pr_url = str(j.get("html_url")) if j.get("html_url") else None

    # Record PR row
    pr = RepoPullRequest(
        snapshot_id=snap.id,
        patch_run_id=run.id,
        repo=repo_full,
        branch=branch,
        pr_number=pr_number,
        pr_url=pr_url,
        title=title,
        body=body,
        created_at=datetime.utcnow(),
    )
    db.add(pr)
    db.commit()

    return True, pr_number, pr_url, None
