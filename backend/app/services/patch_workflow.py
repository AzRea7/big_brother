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

_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$", re.MULTILINE)


def patch_workflow_enabled() -> bool:
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


def _github_repo() -> str:
    repo = str(getattr(settings, "GITHUB_REPO", "") or "").strip()
    if not repo:
        raise RuntimeError("GITHUB_REPO is required (e.g. AzRea7/OneHaven).")
    return repo


def _github_branch() -> str:
    return str(getattr(settings, "GITHUB_BRANCH", "main") or "main").strip() or "main"


def _pr_allowlist_dirs() -> list[str]:
    dirs = getattr(settings, "PR_ALLOWLIST_DIRS", None)
    if isinstance(dirs, list) and dirs:
        return [str(d).strip() for d in dirs if str(d).strip()]
    return []


def _reject_path(path: str) -> Optional[str]:
    p = (path or "").strip()
    if not p:
        return "Empty path in diff."
    p = p.replace("\\", "/")
    if p.startswith("/"):
        return f"Absolute paths not allowed: {p}"
    if ".." in p.split("/"):
        return f"Path traversal not allowed: {p}"
    return None


def _extract_paths(patch_text: str) -> list[str]:
    out: list[str] = []
    for m in _DIFF_HEADER_RE.finditer(patch_text or ""):
        a_path = (m.group(1) or "").strip()
        b_path = (m.group(2) or "").strip()
        if a_path and a_path not in out:
            out.append(a_path)
        if b_path and b_path not in out:
            out.append(b_path)
    return out


def _count_changed_lines(patch_text: str) -> int:
    n = 0
    for line in (patch_text or "").splitlines():
        if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            n += 1
    return n


def validate_unified_diff(db: Session, snapshot_id: int, patch_text: str) -> dict:
    s = (patch_text or "").strip()
    if not s:
        return {"valid": False, "error": "Empty diff.", "files_changed": 0, "lines_changed": 0, "file_paths": []}

    matches = list(_DIFF_HEADER_RE.finditer(s))
    if not matches:
        return {
            "valid": False,
            "error": "Missing diff headers ('diff --git a/... b/...').",
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
        a_path = (m.group(1) or "").strip()
        b_path = (m.group(2) or "").strip()

        err = _reject_path(a_path) or _reject_path(b_path)
        if err:
            return {"valid": False, "error": err, "files_changed": 0, "lines_changed": 0, "file_paths": []}

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
        if a_path and a_path not in file_paths:
            file_paths.append(a_path)
        if b_path and b_path not in file_paths:
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

    snap = db.query(RepoSnapshot).filter(RepoSnapshot.id == snapshot_id).first()
    if not snap:
        return {
            "valid": False,
            "error": f"Unknown snapshot_id={snapshot_id}",
            "files_changed": 0,
            "lines_changed": 0,
            "file_paths": [],
        }

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


def _materialize_repo_root(db: Session, snapshot_id: int, tmpdir: str) -> str:
    """
    Back-compat: older materialize_snapshot_to_disk() versions may not accept dest_dir.
    We try dest_dir first; if it TypeErrors, we call the legacy signature.
    """
    try:
        return materialize_snapshot_to_disk(db=db, snapshot_id=snapshot_id, dest_dir=tmpdir)  # type: ignore[arg-type]
    except TypeError:
        return materialize_snapshot_to_disk(db=db, snapshot_id=snapshot_id)


async def apply_unified_diff_in_sandbox(
    db: Session,
    snapshot_id: int,
    patch_text: str,
    *,
    run_tests: bool = True,
    finding_id: Optional[int] = None,
    objective: Optional[str] = None,
    suggested_pr_title: Optional[str] = None,
    suggested_pr_body: Optional[str] = None,
) -> RepoPatchRun:
    """
    Apply patch to a materialized snapshot (sandbox dir), run tests, persist a RepoPatchRun row.
    """
    if not patch_workflow_enabled():
        raise RuntimeError("Patch workflow disabled (ENABLE_PATCH_WORKFLOW=false).")

    v = validate_unified_diff(db, snapshot_id, patch_text)
    valid = bool(v.get("valid"))
    validation_error = v.get("error")

    run = RepoPatchRun(
        snapshot_id=snapshot_id,
        created_at=datetime.utcnow(),
        patch_text=patch_text,
        finding_id=int(finding_id) if finding_id is not None else None,
        objective=(objective or None),
        suggested_pr_title=(suggested_pr_title or None),
        suggested_pr_body=(suggested_pr_body or None),
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
        repo_root = _materialize_repo_root(db, snapshot_id, tmpdir)

        _run("git init", cwd=repo_root, timeout_s=30)
        _run("git add -A", cwd=repo_root, timeout_s=30)
        _run('git commit -m "snapshot baseline" --no-gpg-sign', cwd=repo_root, timeout_s=60)

        run.sandbox_path = repo_root
        db.commit()

        patch_path = os.path.join(repo_root, "_patch.diff")
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(patch_text)

        r = _run(f"git apply --whitespace=nowarn {patch_path}", cwd=repo_root, timeout_s=60)
        run.applied = bool(r.ok)
        run.apply_error = None if r.ok else r.out
        db.commit()

        if not r.ok:
            return run

        d = _run("git diff --no-color", cwd=repo_root, timeout_s=30)
        run.diff_output = d.out
        db.commit()

        if run_tests:
            cmd = str(getattr(settings, "PR_TEST_CMD", "pytest -q") or "pytest -q")
            timeout_s = int(getattr(settings, "PR_TIMEOUT_SECONDS", 240) or 240)
            t = _run(cmd, cwd=repo_root, timeout_s=timeout_s)
            run.tests_ran = True
            run.tests_ok = bool(t.ok)
            run.test_output = t.out
            db.commit()

        return run

    finally:
        if not keep:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass


async def open_pull_request_from_patch_run(
    db: Session,
    *,
    patch_run_id: int,
    title: Optional[str] = None,
    body: Optional[str] = None,
) -> dict:
    """
    Takes an existing RepoPatchRun (already applied + tested), and opens a PR on GitHub.

    NOTE: Your current implementation is still "not implemented" for pushing commits.
    This function prefers run.suggested_pr_title/body when caller doesn't provide them.
    """
    if not pr_workflow_enabled():
        raise RuntimeError("PR workflow disabled (ENABLE_PR_WORKFLOW=false).")

    run = db.query(RepoPatchRun).filter(RepoPatchRun.id == int(patch_run_id)).first()
    if not run:
        raise RuntimeError(f"Unknown patch_run_id={patch_run_id}")

    if not run.valid:
        raise RuntimeError(f"PatchRun not valid: {run.validation_error}")

    if not run.applied:
        raise RuntimeError(f"PatchRun not applied: {run.apply_error}")

    if run.tests_ran and not run.tests_ok:
        raise RuntimeError("Tests failed in sandbox. Refusing to open PR.")

    repo = _github_repo()
    base_branch = _github_branch()

    pr_title = (title or "").strip() or (getattr(run, "suggested_pr_title", None) or "").strip() or f"Fix: repo finding (snapshot {run.snapshot_id})"
    pr_body = (body or "").strip() or (getattr(run, "suggested_pr_body", None) or "").strip() or (
        "Automated patch from Goal Autopilot.\n\n"
        "- Generated from repo finding\n"
        "- Applied in sandbox\n"
        "- Tests passed (if configured)\n"
    )

    pr_body = (
        pr_body.rstrip()
        + "\n\n---\n"
        + f"- Snapshot: {run.snapshot_id}\n"
        + f"- PatchRun: {run.id}\n"
        + (f"- Finding: {run.finding_id}\n" if getattr(run, "finding_id", None) else "")
    )

    if pr_workflow_dry_run():
        return {
            "dry_run": True,
            "repo": repo,
            "base": base_branch,
            "title": pr_title,
            "body": pr_body,
            "note": "Set PR_WORKFLOW_DRY_RUN=false to actually open PRs.",
        }

    token = _github_token()
    api = httpx.Client(
        base_url="https://api.github.com",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=30.0,
    )
    _ = api  # placeholder until push+create is implemented

    pr = RepoPullRequest(
        snapshot_id=run.snapshot_id,
        patch_run_id=run.id,
        created_at=datetime.utcnow(),
        provider="github",
        repo=repo,
        base_branch=base_branch,
        head_branch=None,
        pr_url=None,
        pr_number=None,
        status="not_implemented",
        error="PR creation pipeline not implemented (set up git object push).",
    )
    db.add(pr)
    db.commit()
    db.refresh(pr)

    return {
        "dry_run": False,
        "opened": False,
        "status": "not_implemented",
        "pr_id": pr.id,
        "error": pr.error,
        "title": pr_title,
        "body": pr_body,
    }
