# backend/app/services/pr_workflow.py
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional

from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoPatchRun
from .repo_materialize import materialize_snapshot_to_dir  # existing service in your repo


_DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$")


def _require_enabled() -> None:
    if not settings.ENABLE_PR_WORKFLOW:
        raise RuntimeError("PR workflow is disabled (ENABLE_PR_WORKFLOW=false)")


def _count_lines_changed(patch_text: str) -> int:
    changed = 0
    for line in patch_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            changed += 1
        if line.startswith("-") and not line.startswith("---"):
            changed += 1
    return changed


def validate_unified_diff(patch_text: str) -> Tuple[bool, Optional[str], int, int, List[str]]:
    """
    Validates:
    - looks like unified diff
    - only modifies files in allowlisted dirs
    - caps number of files/lines
    Returns: (ok, error, files_changed, lines_changed, file_paths)
    """
    _require_enabled()

    if not patch_text.strip():
        return False, "Empty patch", 0, 0, []

    if "diff --git " not in patch_text:
        return False, "Patch must be a unified diff (missing 'diff --git')", 0, 0, []

    file_paths: List[str] = []
    for line in patch_text.splitlines():
        m = _DIFF_FILE_RE.match(line)
        if m:
            file_paths.append(m.group(1))

    file_paths = list(dict.fromkeys(file_paths))  # dedupe preserve order
    if not file_paths:
        return False, "Could not parse any '+++ b/<path>' file entries", 0, 0, []

    # Allowlist
    allow = settings.PR_ALLOWLIST_DIRS
    for p in file_paths:
        if not any(p.startswith(a) for a in allow):
            return False, f"File not in allowlist: {p}", 0, 0, file_paths

    files_changed = len(file_paths)
    if files_changed > settings.PR_MAX_FILES_CHANGED:
        return False, f"Too many files changed: {files_changed} > {settings.PR_MAX_FILES_CHANGED}", files_changed, 0, file_paths

    lines_changed = _count_lines_changed(patch_text)
    if lines_changed > settings.PR_MAX_LINES_CHANGED:
        return False, f"Too many lines changed: {lines_changed} > {settings.PR_MAX_LINES_CHANGED}", files_changed, lines_changed, file_paths

    return True, None, files_changed, lines_changed, file_paths


def _run_cmd(cmd: str, cwd: str, timeout_s: int) -> Tuple[bool, str]:
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


async def apply_patch_and_test(db: Session, snapshot_id: int, patch_text: str, run_tests: bool = True) -> Dict:
    _require_enabled()

    ok, err, files_changed, lines_changed, _paths = validate_unified_diff(patch_text)

    run = RepoPatchRun(snapshot_id=snapshot_id, patch_text=patch_text, valid=ok, validation_error=err)
    db.add(run)
    db.commit()  # run.id exists

    if not ok:
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

    sandbox = tempfile.mkdtemp(prefix=f"repo_patch_{snapshot_id}_")
    run.sandbox_path = sandbox
    db.commit()

    try:
        # Materialize repo snapshot into sandbox
        await materialize_snapshot_to_dir(db, snapshot_id=snapshot_id, out_dir=sandbox)

        # Apply patch
        patch_path = os.path.join(sandbox, "_patch.diff")
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(patch_text)

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

        # Run tests (optional)
        if run_tests:
            tests_ok, out = _run_cmd(settings.PR_TEST_CMD, cwd=sandbox, timeout_s=settings.PR_TIMEOUT_SECONDS)
            run.tests_ran = True
            run.tests_ok = tests_ok
            run.test_output = out
            db.commit()

            return {
                "run_id": run.id,
                "valid": True,
                "validation_error": None,
                "applied": True,
                "apply_error": None,
                "tests_ran": True,
                "tests_ok": tests_ok,
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

    finally:
        # Keep sandbox by default for debugging; you can later add cleanup logic.
        pass
