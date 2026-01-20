# backend/app/services/patch_workflow.py
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import httpx
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoPatchRun, RepoPullRequest, RepoSnapshot
from .repo_materialize import materialize_snapshot_to_disk


_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$", re.MULTILINE)


def patch_workflow_enabled() -> bool:
    return bool(getattr(settings, "ENABLE_PATCH_WORKFLOW", False))


def pr_workflow_enabled() -> bool:
    return bool(getattr(settings, "ENABLE_PR_WORKFLOW", False))


def _github_token() -> str:
    tok = str(getattr(settings, "GITHUB_TOKEN", "") or "").strip()
    if not tok:
        raise RuntimeError("GITHUB_TOKEN is required to open PRs.")
    return tok


def _allowlist_prefixes() -> list[str]:
    raw = str(getattr(settings, "PATCH_ALLOWLIST_PREFIXES", "") or "").strip()
    if not raw:
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


def validate_unified_diff(patch_text: str) -> tuple[bool, Optional[str], int, int]:
    """
    Validates unified diff content and returns:
      (valid, error, files_changed, lines_changed)
    """
    s = (patch_text or "").strip()
    if not s:
        return False, "Empty patch_text.", 0, 0

    matches = list(_DIFF_HEADER_RE.finditer(s))
    if not matches:
        return False, "Patch missing 'diff --git a/... b/...'.", 0, 0

    allow = _allowlist_prefixes()
    max_files = int(getattr(settings, "PATCH_MAX_FILES", 25) or 25)
    max_lines = int(getattr(settings, "PATCH_MAX_LINES", 1200) or 1200)

    files_changed = 0
    for m in matches:
        a_path = m.group(1).strip()
        b_path = m.group(2).strip()

        err = _reject_path(a_path) or _reject_path(b_path)
        if err:
            return False, err, 0, 0

        if allow and not any(a_path.startswith(p) or b_path.startswith(p) for p in allow):
            return False, f"Path not in allowlist: {a_path} / {b_path}", 0, 0

        files_changed += 1

    if files_changed > max_files:
        return False, f"Too many files changed: {files_changed} > {max_files}", 0, 0

    # rough line delta count
    lines_changed = 0
    for line in s.splitlines():
        if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            lines_changed += 1

    if lines_changed > max_lines:
        return False, f"Too many changed lines: {lines_changed} > {max_lines}", 0, 0

    return True, None, files_changed, lines_changed


def _run_cmd(cmd: str, cwd: str, timeout_s: int = 60) -> tuple[bool, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            text=True,
        )
        ok = p.returncode == 0
        return ok, p.stdout
    except Exception as e:
        return False, f"command failed: {e}"


def apply_unified_diff_in_sandbox(
    db: Session,
    snapshot_id: int,
    patch_text: str,
    *,
    run_tests: bool = True,
) -> dict:
    """
    1) validate diff
    2) materialize snapshot into a temp sandbox
    3) init git + commit baseline
    4) apply patch via git apply
    5) optionally run tests
    Stores RepoPatchRun row with outputs.
    """
    if not patch_workflow_enabled():
        raise RuntimeError("Patch workflow is disabled (ENABLE_PATCH_WORKFLOW=false).")

    valid, err, files_changed, lines_changed = validate_unified_diff(patch_text)
    run = RepoPatchRun(
        snapshot_id=snapshot_id,
        created_at=datetime.utcnow(),
        patch_text=patch_text,
        valid=bool(valid),
        validation_error=err,
        files_changed=int(files_changed),
        lines_changed=int(lines_changed),
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
        return {
            "run_id": run.id,
            "valid": False,
            "validation_error": err,
            "applied": False,
            "apply_error": None,
            "tests_ran": False,
            "tests_ok": False,
            "test_output": None,
        }

    sandbox = tempfile.mkdtemp(prefix="repo_patch_sandbox_")
    run.sandbox_path = sandbox
    db.commit()

    # Write patch to disk
    patch_path = os.path.join(sandbox, "change.patch")
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(patch_text)

    # Materialize snapshot
    materialize_snapshot_to_disk(db, snapshot_id, sandbox)

    # Initialize git repo so `git apply` and `git diff` work reliably
    _run_cmd("git init", cwd=sandbox, timeout_s=30)
    _run_cmd("git add -A", cwd=sandbox, timeout_s=60)
    _run_cmd('git commit -m "baseline"', cwd=sandbox, timeout_s=60)

    applied_ok, apply_out = _run_cmd(f"git apply --whitespace=nowarn {patch_path}", cwd=sandbox, timeout_s=60)
    run.applied = applied_ok
    run.apply_error = None if applied_ok else apply_out
    db.commit()

    if not applied_ok:
        return {
            "run_id": run.id,
            "valid": True,
            "validation_error": None,
            "applied": False,
            "apply_error": apply_out,
            "tests_ran": False,
            "tests_ok": False,
            "test_output": None,
        }

    # capture diff for UI review
    _, diff_out = _run_cmd("git diff", cwd=sandbox, timeout_s=30)
    run.diff_output = diff_out
    db.commit()

    if run_tests:
        test_cmd = str(getattr(settings, "PR_TEST_CMD", "pytest -q") or "pytest -q")
        timeout_s = int(getattr(settings, "PR_TIMEOUT_SECONDS", 900) or 900)
        tests_ok, out = _run_cmd(test_cmd, cwd=sandbox, timeout_s=timeout_s)
        run.tests_ran = True
        run.tests_ok = bool(tests_ok)
        run.test_output = out
        db.commit()

        return {
            "run_id": run.id,
            "valid": True,
            "validation_error": None,
            "applied": True,
            "apply_error": None,
            "tests_ran": True,
            "tests_ok": bool(tests_ok),
            "test_output": out,
        }

    return {
        "run_id": run.id,
        "valid": True,
        "validation_error": None,
        "applied": True,
        "apply_error": None,
        "tests_ran": False,
        "tests_ok": False,
        "test_output": None,
    }


def open_pull_request_from_patch_run(
    db: Session,
    run_id: int,
    *,
    title: str,
    body: Optional[str] = None,
    base_branch: Optional[str] = None,
) -> dict:
    """
    OFF by default. When enabled, this will:
      - push a branch to GitHub using git CLI (requires git + network + token)
      - open a PR via GitHub REST API

    This is intentionally strict and can be tightened further (branch naming, commit signing, etc.).
    """
    if not pr_workflow_enabled():
        raise RuntimeError("PR workflow is disabled (ENABLE_PR_WORKFLOW=false).")

    run = db.get(RepoPatchRun, run_id)
    if not run:
        raise RuntimeError(f"run_id={run_id} not found")
    if not run.applied:
        raise RuntimeError("Patch was not applied; cannot open PR.")
    if run.tests_ran and not run.tests_ok:
        raise RuntimeError("Tests failed; cannot open PR.")

    snap = db.get(RepoSnapshot, run.snapshot_id)
    if not snap:
        raise RuntimeError("Snapshot not found for patch run.")

    repo = snap.repo  # e.g. "AzRea7/OneHaven"
    base = base_branch or snap.branch or "main"
    branch = f"autopilot/patch-run-{run.id}"

    sandbox = run.sandbox_path
    if not sandbox or not os.path.isdir(sandbox):
        raise RuntimeError("Sandbox path missing; cannot push.")

    tok = _github_token()

    # Create commit in sandbox
    _run_cmd("git add -A", cwd=sandbox, timeout_s=60)
    safe_title = title.replace('"', '\\"')
    _run_cmd(f'git commit -m "{safe_title}"', cwd=sandbox, timeout_s=60)

    remote_url = f"https://{tok}@github.com/{repo}.git"
    _run_cmd("git remote remove origin", cwd=sandbox, timeout_s=15)
    _run_cmd(f"git remote add origin {remote_url}", cwd=sandbox, timeout_s=15)
    _run_cmd(f"git checkout -b {branch}", cwd=sandbox, timeout_s=30)

    ok, out = _run_cmd(f"git push -u origin {branch}", cwd=sandbox, timeout_s=120)
    if not ok:
        raise RuntimeError(f"Failed to push branch: {out}")

    # Open PR via GitHub API
    api = "https://api.github.com"
    headers = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    payload = {"title": title, "head": branch, "base": base, "body": body or ""}

    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{api}/repos/{repo}/pulls", headers=headers, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"Failed to open PR {r.status_code}: {r.text}")
        pr = r.json()

    pr_url = pr.get("html_url")
    pr_number = pr.get("number")

    db.add(
        RepoPullRequest(
            patch_run_id=run.id,
            repo=repo,
            pr_number=int(pr_number) if pr_number else None,
            pr_url=str(pr_url) if pr_url else None,
            created_at=datetime.utcnow(),
        )
    )
    db.commit()

    return {"ok": True, "pr_url": pr_url, "pr_number": pr_number}
