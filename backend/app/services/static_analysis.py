# backend/app/services/static_analysis.py
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoFinding
from .repo_materialize import materialize_snapshot_to_disk


@dataclass(frozen=True)
class ToolResult:
    tool: str
    created: int
    duration_s: float
    errors: list[str]


def _fingerprint(snapshot_id: int, path: str, line: int, category: str, title: str) -> str:
    raw = f"{snapshot_id}|{path}|{line}|{category}|{title}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def _upsert_finding(
    db: Session,
    *,
    snapshot_id: int,
    path: str,
    line: int,
    category: str,
    severity: int,
    title: str,
    evidence: str,
    recommendation: str,
) -> bool:
    fp = _fingerprint(snapshot_id, path, line, category, title)
    exists = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id, RepoFinding.fingerprint == fp)
        .first()
    )
    if exists:
        return False

    db.add(
        RepoFinding(
            snapshot_id=snapshot_id,
            path=path,
            line=line,
            category=category,
            severity=str(severity),
            title=title,
            evidence=evidence,
            recommendation=recommendation,
            fingerprint=fp,
        )
    )
    return True


def _repo_root_for_scan(db: Session, snapshot_id: int) -> str:
    # Prefer scanning a real checked-out repo
    if settings.REPO_LOCAL_PATH:
        return settings.REPO_LOCAL_PATH

    # Otherwise materialize snapshot from DB to disk
    dest = f"./data/snapshots/{snapshot_id}"
    return materialize_snapshot_to_disk(db, snapshot_id, dest_root=dest)


def run_ruff(db: Session, snapshot_id: int) -> ToolResult:
    start = time.time()
    root = _repo_root_for_scan(db, snapshot_id)
    errors: list[str] = []
    created = 0

    # ruff JSON output
    cmd = ["ruff", "check", ".", "--output-format=json"]
    try:
        p = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=settings.STATIC_SCAN_MAX_SECONDS,
        )
    except Exception as e:
        return ToolResult("ruff", 0, time.time() - start, [f"ruff failed: {e}"])

    if p.returncode not in (0, 1):
        errors.append(p.stderr.strip()[:2000])

    try:
        data = json.loads(p.stdout or "[]")
    except Exception:
        data = []
        if p.stdout.strip():
            errors.append(f"ruff JSON parse failed; stdout[:2000]={p.stdout[:2000]}")

    for item in data:
        # ruff format varies slightly by version but generally includes:
        # filename, location.row, message, code
        filename = str(item.get("filename", ""))
        loc = item.get("location") or {}
        row = int(loc.get("row") or 1)
        code = str(item.get("code") or "RUFF")
        msg = str(item.get("message") or "").strip()

        # severity heuristic: F/E -> higher, otherwise medium
        sev = 4 if code.startswith(("F", "E")) else 3

        title = f"ruff {code}: {msg}"
        evidence = json.dumps(item, indent=2)[:4000]
        rec = "Fix the lint issue (format, unused imports, unsafe patterns)."

        if _upsert_finding(
            db,
            snapshot_id=snapshot_id,
            path=filename,
            line=row,
            category="quality/ruff",
            severity=sev,
            title=title,
            evidence=evidence,
            recommendation=rec,
        ):
            created += 1

    db.commit()
    return ToolResult("ruff", created, time.time() - start, errors)


def run_bandit(db: Session, snapshot_id: int) -> ToolResult:
    start = time.time()
    root = _repo_root_for_scan(db, snapshot_id)
    errors: list[str] = []
    created = 0

    cmd = ["bandit", "-r", ".", "-f", "json", "-q"]
    try:
        p = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=settings.STATIC_SCAN_MAX_SECONDS,
        )
    except Exception as e:
        return ToolResult("bandit", 0, time.time() - start, [f"bandit failed: {e}"])

    if p.returncode not in (0, 1):
        errors.append(p.stderr.strip()[:2000])

    try:
        data = json.loads(p.stdout or "{}")
        results = data.get("results") or []
    except Exception:
        results = []
        if p.stdout.strip():
            errors.append(f"bandit JSON parse failed; stdout[:2000]={p.stdout[:2000]}")

    # bandit severity/confidence are strings: LOW/MEDIUM/HIGH
    sev_map = {"LOW": 2, "MEDIUM": 3, "HIGH": 4}

    for r in results:
        filename = str(r.get("filename", ""))
        line = int(r.get("line_number") or 1)
        test_id = str(r.get("test_id") or "BANDIT")
        issue = str(r.get("issue_text") or "").strip()
        severity = sev_map.get(str(r.get("issue_severity") or "MEDIUM"), 3)

        title = f"bandit {test_id}: {issue}"
        evidence = json.dumps(r, indent=2)[:4000]
        rec = "Address the security finding (avoid injection, weak crypto, unsafe subprocess, etc.)."

        if _upsert_finding(
            db,
            snapshot_id=snapshot_id,
            path=filename,
            line=line,
            category="security/bandit",
            severity=severity,
            title=title,
            evidence=evidence,
            recommendation=rec,
        ):
            created += 1

    db.commit()
    return ToolResult("bandit", created, time.time() - start, errors)


def run_mypy(db: Session, snapshot_id: int) -> ToolResult:
    start = time.time()
    root = _repo_root_for_scan(db, snapshot_id)
    errors: list[str] = []
    created = 0

    # mypy doesn't have a clean stable JSON mode across versions
    # so we parse the standard "file:line: col: error: msg  [code]" format.
    cmd = [
        "mypy",
        ".",
        "--show-error-codes",
        "--no-error-summary",
        "--hide-error-context",
        "--no-pretty",
    ]
    try:
        p = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=settings.STATIC_SCAN_MAX_SECONDS,
        )
    except Exception as e:
        return ToolResult("mypy", 0, time.time() - start, [f"mypy failed: {e}"])

    # mypy returns 1 when errors exist (normal)
    if p.returncode not in (0, 1, 2):
        errors.append(p.stderr.strip()[:2000])

    stdout = p.stdout or ""
    for line in stdout.splitlines():
        # Typical:
        # path/to/file.py:123: error: Incompatible types ...  [assignment]
        # path/to/file.py:123:45: error: ...  [name-defined]
        parts = line.split(":")
        if len(parts) < 3:
            continue

        file_path = parts[0]
        try:
            row = int(parts[1])
        except Exception:
            row = 1

        # Find "error:" marker
        if "error:" not in line:
            continue

        msg = line.split("error:", 1)[1].strip()
        sev = 3

        title = f"mypy: {msg}"
        evidence = line[:4000]
        rec = "Fix type issues (annotations, narrowing, Optional handling, protocol mismatches)."

        if _upsert_finding(
            db,
            snapshot_id=snapshot_id,
            path=file_path,
            line=row,
            category="typing/mypy",
            severity=sev,
            title=title,
            evidence=evidence,
            recommendation=rec,
        ):
            created += 1

    db.commit()
    return ToolResult("mypy", created, time.time() - start, errors)


def run_static_analysis_all(db: Session, snapshot_id: int) -> dict[str, Any]:
    """
    Run all deterministic analyzers and return a summary payload.
    """
    results = [
        run_ruff(db, snapshot_id),
        run_mypy(db, snapshot_id),
        run_bandit(db, snapshot_id),
    ]
    return {
        "snapshot_id": snapshot_id,
        "results": [
            {
                "tool": r.tool,
                "created": r.created,
                "duration_s": round(r.duration_s, 3),
                "errors": r.errors,
            }
            for r in results
        ],
    }
