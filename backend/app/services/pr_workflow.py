# backend/app/services/pr_workflow.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoPatchRun

# New canonical implementation lives here:
from .patch_workflow import (
    apply_unified_diff_in_sandbox,
    patch_workflow_enabled,
    pr_workflow_enabled,
    pr_workflow_dry_run,
    validate_unified_diff as _validate_unified_diff_dict,
)


def _require_enabled() -> None:
    # Keep legacy behavior: this file historically gated "PR workflow" as a whole.
    # In the new design, patch apply and PR open are separately gated.
    if not patch_workflow_enabled():
        raise RuntimeError("Patch workflow is disabled (ENABLE_PATCH_WORKFLOW=false)")
    # Note: we do NOT require PR workflow enabled unless opening PRs.


def validate_unified_diff(db: Session, snapshot_id: int, patch_text: str) -> Dict[str, object]:
    """
    Canonical validator shape used by API:
      { valid, error, files_changed, lines_changed, file_paths }
    Delegates to patch_workflow.validate_unified_diff.
    """
    _require_enabled()
    return _validate_unified_diff_dict(db=db, snapshot_id=int(snapshot_id), patch_text=str(patch_text))


def validate_unified_diff_tuple(patch_text: str) -> Tuple[bool, Optional[str], int, int, List[str]]:
    """
    Legacy tuple validator for any old callers:
      (ok, error, files_changed, lines_changed, file_paths)

    Note: This variant cannot check snapshot existence or DB-backed allowlists,
    so it is best-effort only.
    """
    s = (patch_text or "").strip()
    if not s:
        return False, "Empty patch", 0, 0, []
    if "diff --git " not in s:
        return False, "Patch must be a unified diff (missing 'diff --git')", 0, 0, []

    # minimal parse (fallback)
    file_paths: List[str] = []
    for line in s.splitlines():
        if line.startswith("+++ b/"):
            file_paths.append(line[len("+++ b/") :].strip())

    # dedupe preserve order
    seen = set()
    file_paths = [p for p in file_paths if not (p in seen or seen.add(p))]

    if not file_paths:
        return False, "Could not parse any '+++ b/<path>' file entries", 0, 0, []

    files_changed = len(file_paths)

    # rough line count
    lines_changed = 0
    for line in s.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            lines_changed += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_changed += 1

    return True, None, files_changed, lines_changed, file_paths


def _serialize_patch_run(run: RepoPatchRun) -> Dict[str, Any]:
    return {
        "run_id": getattr(run, "id", None),
        "snapshot_id": getattr(run, "snapshot_id", None),
        "valid": bool(getattr(run, "valid", False)),
        "validation_error": getattr(run, "validation_error", None),
        "files_changed": int(getattr(run, "files_changed", 0) or 0),
        "lines_changed": int(getattr(run, "lines_changed", 0) or 0),
        "file_paths_json": getattr(run, "file_paths_json", None),
        "applied": bool(getattr(run, "applied", False)),
        "apply_error": getattr(run, "apply_error", None),
        "tests_ran": bool(getattr(run, "tests_ran", False)),
        "tests_ok": bool(getattr(run, "tests_ok", False)),
        "test_output": getattr(run, "test_output", None),
        "sandbox_path": getattr(run, "sandbox_path", None),
        "created_at": getattr(run, "created_at", None).isoformat() if getattr(run, "created_at", None) else None,
    }


async def apply_patch_and_test(db: Session, snapshot_id: int, patch_text: str, run_tests: bool = True) -> Dict[str, Any]:
    """
    Legacy entrypoint kept for compatibility.

    New canonical flow:
      apply_unified_diff_in_sandbox() persists RepoPatchRun and returns it.
    """
    _require_enabled()

    run = await apply_unified_diff_in_sandbox(
        db=db,
        snapshot_id=int(snapshot_id),
        patch_text=str(patch_text),
        run_tests=bool(run_tests),
    )

    return {
        "run_id": getattr(run, "id", None),
        "valid": bool(getattr(run, "valid", False)),
        "validation_error": getattr(run, "validation_error", None),
        "applied": bool(getattr(run, "applied", False)),
        "apply_error": getattr(run, "apply_error", None),
        "tests_ran": bool(getattr(run, "tests_ran", False)),
        "tests_ok": bool(getattr(run, "tests_ok", False)),
        "test_output": getattr(run, "test_output", None),
        "run": _serialize_patch_run(run),
        "workflow_flags": {
            "patch_workflow_enabled": patch_workflow_enabled(),
            "pr_workflow_enabled": pr_workflow_enabled(),
            "pr_workflow_dry_run": pr_workflow_dry_run(),
            "pr_test_cmd": getattr(settings, "PR_TEST_CMD", None),
        },
    }
