# backend/app/services/patch_workflow.py
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import Any, Optional

import httpx
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoPatchRun, RepoPullRequest, RepoSnapshot
from .repo_materialize import materialize_snapshot_to_dir


_DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$")


def _pr_workflow_enabled() -> bool:
    return bool(getattr(settings, "ENABLE_PR_WORKFLOW", False))


def _require_enabled() -> None:
    if not _pr_workflow_enabled():
        raise RuntimeError("PR workflow is disabled (ENABLE_PR_WORKFLOW=false)")


def _count_lines_changed(patch_text: str) -> int:
    changed = 0
    for line in patch_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            changed += 1
        if line.startswith("-") and not line.startswith("---"):
            changed += 1
    return changed


def _run_cmd(cmd: str, cwd: str, timeout_s: int) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode == 0, out.strip()
    except Exception as e:
        return False, f"Command failed: {e!r}"


def validate_unified_diff(
    *,
    db: Optional[Session] = None,        # accepted for route compatibility
    snapshot_id: Optional[int] = None,   # accepted for route compatibility
    patch_text: str,
) -> dict[str, Any]:
    """
    Gate C (mutation safety):
      - unified diff structure
      - allowlist dirs
      - cap files + lines

    Returns a dict your routes can return directly.
    """
    _require_enabled()

    patch_text = str(patch_text or "")
    if not patch_text.strip():
        return {"ok": False, "error": "Empty patch", "files_changed": 0, "lines_changed": 0, "file_paths": []}

    if "diff --git " not in patch_text:
        return {
            "ok": False,
            "error": "Patch must be a unified diff (missing 'diff --git')",
            "files_changed": 0,
            "lines_changed": 0,
            "file_paths": [],
        }

    file_paths: list[str] = []
    for line in patch_text.splitlines():
        m = _DIFF_FILE_RE.match(line)
        if m:
            file_paths.append(m.group(1))

    # dedupe preserve order
    file_paths = list(dict.fromkeys(file_paths))
    if not file_paths:
        return {
            "ok": False,
            "error": "Could not parse any '+++ b/<path>' file entries",
            "files_changed": 0,
            "lines_changed": 0,
            "file_paths": [],
        }

    allow = list(getattr(settings, "PR_ALLOWLIST_DIRS", ["backend/"]))
    for p in file_paths:
        if not any(p.startswith(a) for a in allow):
            return {
                "ok": False,
                "error": f"File not in allowlist: {p}",
                "files_changed": 0,
                "lines_changed": 0,
                "file_paths": file_paths,
            }

    files_changed = len(file_paths)
    max_files = int(getattr(settings, "PR_MAX_FILES_CHANGED", 8))
    if files_changed > max_files:
        return {
            "ok": False,
            "error": f"Too many files changed: {files_changed} > {max_files}",
            "files_changed": files_changed,
            "lines_changed": 0,
            "file_paths": file_paths,
        }

    lines_changed = _count_lines_changed(patch_text)
    max_lines = int(getattr(settings, "PR_MAX_LINES_CHANGED", 400))
    if lines_changed > max_lines:
        return {
            "ok": False,
            "error": f"Too many lines changed: {lines_changed} > {max_lines}",
            "files_changed": files_changed,
            "lines_changed": lines_changed,
            "file_paths": file_paths,
        }

    return {
        "ok": True,
        "error": None,
        "files_changed": files_changed,
        "lines_changed": lines_changed,
        "file_paths": file_paths,
    }


async def apply_unified_diff_in_sandbox(
    *,
    db: Session,
    snapshot_id: int,
    patch_text: str,
    run_tests: bool = True,
) -> dict[str, Any]:
    """
    Gate C continued:
      - materialize snapshot into sandbox
      - git apply patch
      - optionally run tests command
      - persist RepoPatchRun
    """
    _require_enabled()

    v = validate_unified_diff(db=db, snapshot_id=snapshot_id, patch_text=patch_text)
    ok = bool(v.get("ok"))
    err = v.get("error")
    files_changed = v.get("files_changed")
    lines_changed = v.get("lines_changed")
    file_paths = v.get("file_paths") or []

    run = RepoPatchRun(
        snapshot_id=snapshot_id,
        patch_text=str(patch_text or ""),
        valid=ok,
        validation_error=str(err) if err else None,
        files_changed=int(files_changed) if isinstance(files_changed, int) else None,
        lines_changed=int(lines_changed) if isinstance(lines_changed, int) else None,
        file_paths_json=None,
        created_at=datetime.utcnow(),
    )
    # store file_paths_json compactly
    try:
        import json
        run.file_paths_json = json.dumps(file_paths, separators=(",", ":"))
    except Exception:
        run.file_paths_json = None

    db.add(run)
    db.commit()  # run.id exists

    if not ok:
        return {
            "run_id": int(run.id),
            "valid": False,
            "validation_error": err,
            "applied": False,
            "apply_error": None,
            "tests_ran": False,
            "tests_ok": False,
            "test_output": None,
        }

    sandbox = tempfile.mkdtemp(prefix=f"repo_patch_{snapshot_id}_")
    run.sandbox_path = sandbox
    db.commit()

    # Materialize repo snapshot into sandbox
    await materialize_snapshot_to_dir(db, snapshot_id=snapshot_id, out_dir=sandbox)

    patch_path = os.path.join(sandbox, "_patch.diff")
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(str(patch_text or ""))

    applied_ok, apply_out = _run_cmd(f"git apply --whitespace=nowarn {patch_path}", cwd=sandbox, timeout_s=60)
    run.applied = bool(applied_ok)
    run.apply_error = None if applied_ok else apply_out
    run.diff_output = apply_out[:4000] if apply_out else None
    db.commit()

    if not applied_ok:
        return {
            "run_id": int(run.id),
            "valid": True,
            "validation_error": None,
            "applied": False,
            "apply_error": apply_out,
            "tests_ran": False,
            "tests_ok": False,
            "test_output": None,
        }

    if run_tests:
        cmd = str(getattr(settings, "PR_TEST_CMD", "pytest -q") or "pytest -q")
        timeout = int(getattr(settings, "PR_TIMEOUT_SECONDS", 900))
        tests_ok, out = _run_cmd(cmd, cwd=sandbox, timeout_s=timeout)
        run.tests_ran = True
        run.tests_ok = bool(tests_ok)
        run.test_output = (out or "")[:20000]
        db.commit()

        return {
            "run_id": int(run.id),
            "valid": True,
            "validation_error": None,
            "applied": True,
            "apply_error": None,
            "tests_ran": True,
            "tests_ok": bool(tests_ok),
            "test_output": out,
        }

    return {
        "run_id": int(run.id),
        "valid": True,
        "validation_error": None,
        "applied": True,
        "apply_error": None,
        "tests_ran": False,
        "tests_ok": False,
        "test_output": None,
    }


def _github_token() -> str:
    tok = str(getattr(settings, "GITHUB_TOKEN", "") or "").strip()
    if not tok:
        raise RuntimeError("Missing GITHUB_TOKEN in settings/env.")
    return tok


def _assert_pr_ready(run: RepoPatchRun) -> None:
    """
    Hard safety gate before opening a PR.
    This is the difference between “workflow exists” and “Level 3 works safely”.
    """
    if not bool(run.valid):
        raise RuntimeError(f"Patch run {run.id} is not valid: {run.validation_error or 'unknown'}")
    if not bool(run.applied):
        raise RuntimeError(f"Patch run {run.id} was not applied: {run.apply_error or 'unknown'}")
    if not bool(run.tests_ran):
        raise RuntimeError(f"Patch run {run.id} did not run tests; refusing to open PR.")
    if not bool(run.tests_ok):
        raise RuntimeError(f"Patch run {run.id} tests failed; refusing to open PR.")


def open_pull_request_from_patch_run(
    *,
    db: Session,
    snapshot_id: int,
    patch_run_id: int,
    title: str,
    body: str = "",
    base_branch: str = "main",
) -> dict[str, Any]:
    """
    Open a PR ONLY after:
      valid diff → applied → tests passed.

    (We enforce this here, not in the route, so nobody can bypass it.)
    """
    _require_enabled()

    snap = db.get(RepoSnapshot, int(snapshot_id))
    if not snap:
        raise RuntimeError(f"Snapshot {snapshot_id} not found")

    run = db.get(RepoPatchRun, int(patch_run_id))
    if not run or int(run.snapshot_id) != int(snapshot_id):
        raise RuntimeError("Patch run not found for this snapshot")

    _assert_pr_ready(run)

    repo = str(snap.repo or "").strip()
    if not repo:
        raise RuntimeError("Snapshot repo missing")

    base = str(base_branch or snap.branch or "main").strip() or "main"
    branch = f"autopilot/patch-run-{run.id}"

    sandbox = run.sandbox_path
    if not sandbox or not os.path.isdir(sandbox):
        raise RuntimeError("Sandbox path missing; cannot push.")

    tok = _github_token()

    # Create commit in sandbox
    _run_cmd("git add -A", cwd=sandbox, timeout_s=60)
    safe_title = str(title or "Autopilot patch").replace('"', '\\"')
    ok, out = _run_cmd(f'git commit -m "{safe_title}"', cwd=sandbox, timeout_s=60)
    if not ok:
        raise RuntimeError(f"Failed to commit: {out}")

    remote_url = f"https://{tok}@github.com/{repo}.git"
    _run_cmd("git remote remove origin", cwd=sandbox, timeout_s=15)
    _run_cmd(f"git remote add origin {remote_url}", cwd=sandbox, timeout_s=15)
    _run_cmd(f"git checkout -b {branch}", cwd=sandbox, timeout_s=30)

    ok, out = _run_cmd(f"git push -u origin {branch}", cwd=sandbox, timeout_s=120)
    if not ok:
        raise RuntimeError(f"Failed to push branch: {out}")

    api = "https://api.github.com"
    headers = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    payload = {"title": str(title), "head": branch, "base": base, "body": str(body or "")}

    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{api}/repos/{repo}/pulls", headers=headers, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"Failed to open PR {r.status_code}: {r.text}")
        pr = r.json()

    pr_url = pr.get("html_url")
    pr_number = pr.get("number")

    db.add(
        RepoPullRequest(
            snapshot_id=int(snapshot_id),
            patch_run_id=int(run.id),
            repo=repo,
            branch=branch,
            pr_number=int(pr_number) if pr_number else None,
            pr_url=str(pr_url) if pr_url else None,
            title=str(title),
            body=str(body or "") if body is not None else None,
            created_at=datetime.utcnow(),
        )
    )
    db.commit()

    return {"ok": True, "pr_url": pr_url, "pr_number": pr_number, "branch": branch}
