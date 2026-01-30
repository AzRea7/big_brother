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

# Only lines that are valid unified diff lines (prevents trailing junk breaking git apply)
_VALID_DIFF_LINE_RE = re.compile(
    r"^(diff --git |index |--- |\+\+\+ |@@ |\+|-| |\\ No newline at end of file$)"
)

# Import guard: detect removed imports from patch
_REMOVED_FROM_IMPORT_RE = re.compile(r"^-from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+)\s*$")
_REMOVED_IMPORT_RE = re.compile(r"^-import\s+(.+)\s*$")


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


def _sanitize_unified_diff(patch_text: str) -> str:
    """
    Prevent "patch fragment without header" and other apply failures caused by trailing junk.
    Keeps only real diff lines and stops at first non-diff line.
    """
    s = (patch_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not s:
        return ""

    idx = s.find("diff --git")
    if idx == -1:
        return ""
    s = s[idx:].lstrip()

    lines = s.splitlines()
    if not lines or not lines[0].lstrip().startswith("diff --git"):
        return ""

    kept: list[str] = []
    for ln in lines:
        if _VALID_DIFF_LINE_RE.match(ln):
            kept.append(ln)
        else:
            break

    out = "\n".join(kept).strip()
    if out and not out.endswith("\n"):
        out += "\n"
    return out


def validate_unified_diff(db: Session, snapshot_id: int, patch_text: str) -> dict:
    s = _sanitize_unified_diff(patch_text)
    if not s:
        return {
            "valid": False,
            "error": "Empty or non-diff patch.",
            "files_changed": 0,
            "lines_changed": 0,
            "file_paths": [],
        }

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

    # ✅ NEW: reject “header-only” patches here too
    if lines_changed <= 0:
        return {
            "valid": False,
            "error": "Patch has 0 changed lines (+/-). Header-only diff is not allowed.",
            "files_changed": files_changed,
            "lines_changed": 0,
            "file_paths": file_paths,
        }

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


def _run(
    cmd: str,
    *,
    cwd: str,
    timeout_s: int,
    input_text: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
) -> _CmdResult:
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
            env=env,
        )
        out = (p.stdout or "").strip()
        return _CmdResult(ok=(p.returncode == 0), out=out)
    except subprocess.TimeoutExpired:
        return _CmdResult(ok=False, out=f"Command timed out after {timeout_s}s: {cmd}")


def _materialize_repo_root(db: Session, snapshot_id: int, tmpdir: str) -> str:
    """
    Materialize snapshot into the sandbox dir.

    Tries (in order):
      1) dest_root=tmpdir
      2) dest_dir=tmpdir
      3) positional (db, snapshot_id, tmpdir)
      4) last resort: call without dest ONLY if the function supports a default
    """
    try:
        return materialize_snapshot_to_disk(db=db, snapshot_id=snapshot_id, dest_root=tmpdir)
    except TypeError:
        pass

    try:
        return materialize_snapshot_to_disk(db=db, snapshot_id=snapshot_id, dest_dir=tmpdir)  # type: ignore[arg-type]
    except TypeError:
        pass

    try:
        return materialize_snapshot_to_disk(db, snapshot_id, tmpdir)  # type: ignore[misc]
    except TypeError:
        pass

    return materialize_snapshot_to_disk(db=db, snapshot_id=snapshot_id)  # type: ignore[call-arg]


def _removed_import_symbols_from_patch(patch_text: str) -> dict[str, set[str]]:
    """
    Parse removed import lines for Python files and return {path -> {symbol names removed}}.
    """
    s = _sanitize_unified_diff(patch_text)
    if not s:
        return {}

    current_path: str | None = None
    removed: dict[str, set[str]] = {}

    for ln in s.splitlines():
        m = _DIFF_HEADER_RE.match(ln)
        if m:
            current_path = (m.group(2) or "").strip()
            continue

        if not current_path or not current_path.endswith(".py"):
            continue

        fm = _REMOVED_FROM_IMPORT_RE.match(ln)
        if fm:
            sym_part = (fm.group(2) or "").split("#", 1)[0].strip().strip("()")
            names = [x.strip() for x in sym_part.split(",") if x.strip()]
            for n in names:
                if " as " in n:
                    a, b = [p.strip() for p in n.split(" as ", 1)]
                    for nm in (a, b):
                        removed.setdefault(current_path, set()).add(nm)
                else:
                    removed.setdefault(current_path, set()).add(n)
            continue

        im = _REMOVED_IMPORT_RE.match(ln)
        if im:
            part = (im.group(1) or "").split("#", 1)[0].strip()
            chunks = [x.strip() for x in part.split(",") if x.strip()]
            for ch in chunks:
                if " as " in ch:
                    a, b = [p.strip() for p in ch.split(" as ", 1)]
                    removed.setdefault(current_path, set()).add(a.split(".")[0])
                    removed.setdefault(current_path, set()).add(b.split(".")[0])
                else:
                    removed.setdefault(current_path, set()).add(ch.split(".")[0])
            continue

    return removed


def _symbol_still_used_in_file(repo_root: str, path: str, symbol: str) -> bool:
    fs_path = os.path.join(repo_root, path)
    if not os.path.exists(fs_path):
        return False
    try:
        txt = open(fs_path, "r", encoding="utf-8").read()
    except Exception:
        return False
    return re.search(rf"\b{re.escape(symbol)}\b", txt) is not None


def _quality_gates(repo_root: str, patch_text: str) -> tuple[bool, str]:
    """
    Quick sanity checks after apply, before tests.

    Gates:
      - python -m compileall .
      - optional ruff check .
      - optional removed-import-still-referenced guard
    """
    logs: list[str] = []

    comp_to = int(getattr(settings, "PR_COMPILEALL_TIMEOUT_SECONDS", 120) or 120)
    comp = _run("python -m compileall .", cwd=repo_root, timeout_s=comp_to)
    logs.append("== compileall ==")
    logs.append(comp.out or "(no output)")
    if not comp.ok:
        return (False, "\n".join(logs).strip())

    if bool(getattr(settings, "PR_RUFF_ENABLED", False)):
        ruff_cmd = str(getattr(settings, "PR_RUFF_CMD", "ruff check .") or "ruff check .")
        ruff_to = int(getattr(settings, "PR_RUFF_TIMEOUT_SECONDS", 120) or 120)
        rf = _run(ruff_cmd, cwd=repo_root, timeout_s=ruff_to)
        logs.append("\n== ruff ==")
        logs.append(rf.out or "(no output)")
        if not rf.ok:
            return (False, "\n".join(logs).strip())

    if bool(getattr(settings, "PR_IMPORT_REF_GUARD", True)):
        removed = _removed_import_symbols_from_patch(patch_text)
        for pth, syms in removed.items():
            for sym in syms:
                if _symbol_still_used_in_file(repo_root, pth, sym):
                    logs.append("\n== import-ref-guard ==")
                    logs.append(f"Removed import symbol still referenced: {sym} in {pth}")
                    return (False, "\n".join(logs).strip())

    return (True, "\n".join(logs).strip())


def _git_apply_with_fallback(repo_root: str, patch_file: str) -> tuple[bool, str]:
    """
    Apply a patch in a robust way:
      1) git apply --check
      2) git apply
      3) if (2) fails, try git apply --3way
    """
    chk = _run(f"git apply --check --whitespace=nowarn {patch_file}", cwd=repo_root, timeout_s=60)
    if not chk.ok:
        return (False, chk.out)

    ap = _run(f"git apply --whitespace=nowarn {patch_file}", cwd=repo_root, timeout_s=60)
    if ap.ok:
        return (True, ap.out or "")

    # 3-way can sometimes rescue small context drift
    ap3 = _run(f"git apply --3way --whitespace=nowarn {patch_file}", cwd=repo_root, timeout_s=120)
    if ap3.ok:
        return (True, ap3.out or "")

    combined = (ap.out or "").strip()
    combined3 = (ap3.out or "").strip()
    out = combined + ("\n\n== 3way attempt ==\n" + combined3 if combined3 else "")
    return (False, out.strip())


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
    Apply patch to a materialized snapshot (sandbox dir), run gates/tests, persist a RepoPatchRun row.

    Pipeline:
      sanitize -> validate -> apply-check/apply -> quality gates -> tests
    """
    if not patch_workflow_enabled():
        raise RuntimeError("Patch workflow disabled (ENABLE_PATCH_WORKFLOW=false).")

    sanitized = _sanitize_unified_diff(patch_text)

    v = validate_unified_diff(db, snapshot_id, sanitized)
    valid = bool(v.get("valid"))
    validation_error = v.get("error")

    run = RepoPatchRun(
        snapshot_id=snapshot_id,
        created_at=datetime.utcnow(),
        patch_text=sanitized,
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

        # Make the sandbox a tiny git repo so git apply / diff work consistently
        _run("git init", cwd=repo_root, timeout_s=30)
        _run("git add -A", cwd=repo_root, timeout_s=30)
        _run('git commit -m "snapshot baseline" --no-gpg-sign', cwd=repo_root, timeout_s=60)

        run.sandbox_path = repo_root
        db.commit()

        patch_path = os.path.join(repo_root, "_patch.diff")
        with open(patch_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(sanitized)

        ok, apply_out = _git_apply_with_fallback(repo_root, "_patch.diff")
        run.applied = bool(ok)
        run.apply_error = None if ok else apply_out
        db.commit()

        if not ok:
            return run

        d = _run("git diff --no-color", cwd=repo_root, timeout_s=30)
        run.diff_output = d.out
        db.commit()

        gates_ok, gates_log = _quality_gates(repo_root, sanitized)
        if gates_log:
            run.test_output = gates_log
            db.commit()
        if not gates_ok:
            run.tests_ran = True
            run.tests_ok = False
            db.commit()
            return run

        if run_tests:
            cmd = str(getattr(settings, "PR_TEST_CMD", "pytest -q") or "pytest -q")
            timeout_s = int(getattr(settings, "PR_TIMEOUT_SECONDS", 240) or 240)
            t = _run(cmd, cwd=repo_root, timeout_s=timeout_s)
            run.tests_ran = True
            run.tests_ok = bool(t.ok)
            combined = (run.test_output or "").rstrip()
            combined = combined + ("\n\n" if combined else "") + "== tests ==\n" + (t.out or "(no output)")
            run.test_output = combined
            db.commit()

        return run

    finally:
        if not keep:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass


def _safe_branch_name(snapshot_id: int, patch_run_id: int) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"autopatch/s{snapshot_id}/r{patch_run_id}-{ts}"


def _git_remote_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def _push_with_token(repo_root: str, repo: str, token: str, branch: str) -> _CmdResult:
    """
    Push without leaking token in logs.
    """
    remote = _git_remote_url(repo)
    auth_remote = remote.replace("https://", f"https://{token}@")
    res = _run(
        f'git push "{auth_remote}" HEAD:{branch}',
        cwd=repo_root,
        timeout_s=int(getattr(settings, "PR_PUSH_TIMEOUT_SECONDS", 180) or 180),
    )
    masked = res.out.replace(token, "***") if res.out else res.out
    return _CmdResult(ok=res.ok, out=masked)


async def open_pull_request_from_patch_run(
    db: Session,
    *,
    patch_run_id: int,
    title: Optional[str] = None,
    body: Optional[str] = None,
    base_branch: Optional[str] = None,
) -> dict:
    """
    Open a REAL GitHub PR from a patch run by:
      - re-materializing the snapshot
      - re-applying the stored sanitized patch
      - creating a branch
      - committing changes
      - pushing branch
      - creating PR via GitHub API

    Hard gate: patch run must have passed sandbox gates/tests.
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

    if not run.tests_ran or not run.tests_ok:
        raise RuntimeError("PatchRun did not pass gates/tests. Refusing to open PR.")

    repo = _github_repo()
    base = (base_branch or "").strip() or _github_branch()

    pr_title = (
        (title or "").strip()
        or (getattr(run, "suggested_pr_title", None) or "").strip()
        or f"Fix: repo finding (snapshot {run.snapshot_id})"
    )

    pr_body = (body or "").strip() or (getattr(run, "suggested_pr_body", None) or "").strip()
    if not pr_body:
        pr_body = (
            "## Summary\n"
            "- Automated patch from Goal Autopilot.\n\n"
            "## Why\n"
            "- Addresses a stored repo finding.\n\n"
            "## How Verified\n"
            "- Sandbox gates/tests: Passed ✅\n\n"
            "## Traceability\n"
            f"- Snapshot: {run.snapshot_id}\n"
            f"- PatchRun: {run.id}\n"
            + (f"- Finding: {run.finding_id}\n" if getattr(run, "finding_id", None) else "")
        )
    else:
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
            "base": base,
            "title": pr_title,
            "body": pr_body,
            "note": "Set PR_WORKFLOW_DRY_RUN=false to actually push a branch and open PRs.",
        }

    token = _github_token()

    tmpdir = tempfile.mkdtemp(prefix=f"pr_open_{run.snapshot_id}_{run.id}_")
    keep = bool(getattr(settings, "KEEP_PATCH_SANDBOX", False))

    try:
        repo_root = _materialize_repo_root(db, int(run.snapshot_id), tmpdir)

        _run("git init", cwd=repo_root, timeout_s=30)
        _run("git add -A", cwd=repo_root, timeout_s=30)
        _run('git commit -m "snapshot baseline" --no-gpg-sign', cwd=repo_root, timeout_s=60)

        patch_path = os.path.join(repo_root, "_patch.diff")
        with open(patch_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(_sanitize_unified_diff(str(run.patch_text or "")))

        ok, out = _git_apply_with_fallback(repo_root, "_patch.diff")
        if not ok:
            raise RuntimeError(f"PR open failed: patch does not apply:\n{out}")

        # (Optional) re-run gates for safety in PR open, but keep it cheap
        if bool(getattr(settings, "PR_OPEN_RERUN_GATES", True)):
            gates_ok, gates_log = _quality_gates(repo_root, str(run.patch_text or ""))
            if not gates_ok:
                raise RuntimeError(f"PR open failed: gates failed:\n{gates_log}")

        branch = _safe_branch_name(int(run.snapshot_id), int(run.id))
        _run(f'git checkout -b "{branch}"', cwd=repo_root, timeout_s=30)

        _run("git add -A", cwd=repo_root, timeout_s=30)
        cm = _run(f'git commit -m "{pr_title}" --no-gpg-sign', cwd=repo_root, timeout_s=60)
        if not cm.ok:
            raise RuntimeError(f"PR open failed: could not commit changes:\n{cm.out}")

        push = _push_with_token(repo_root, repo, token, branch)
        if not push.ok:
            raise RuntimeError(f"PR open failed: git push failed:\n{push.out}")

        api = httpx.Client(
            base_url="https://api.github.com",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=30.0,
        )

        owner, name = repo.split("/", 1)
        resp = api.post(
            f"/repos/{owner}/{name}/pulls",
            json={
                "title": pr_title,
                "head": branch,
                "base": base,
                "body": pr_body,
            },
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"GitHub PR create failed ({resp.status_code}): {resp.text}")

        data = resp.json()
        pr_url = data.get("html_url")
        pr_number = data.get("number")

        pr = RepoPullRequest(
            snapshot_id=run.snapshot_id,
            patch_run_id=run.id,
            created_at=datetime.utcnow(),
            provider="github",
            repo=repo,
            base_branch=base,
            head_branch=branch,
            pr_url=pr_url,
            pr_number=int(pr_number) if pr_number is not None else None,
            status="opened",
            error=None,
        )
        db.add(pr)
        db.commit()
        db.refresh(pr)

        return {
            "dry_run": False,
            "opened": True,
            "repo": repo,
            "base": base,
            "head": branch,
            "pr_id": pr.id,
            "pr_number": pr.pr_number,
            "pr_url": pr.pr_url,
            "title": pr_title,
        }

    finally:
        if not keep:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
