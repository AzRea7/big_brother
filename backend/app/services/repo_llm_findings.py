# backend/app/services/repo_llm_findings.py
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from ..ai.llm import chat_completion_json
from ..ai.prompts import REPO_FINDINGS_SYSTEM, repo_findings_user
from ..config import settings
from ..models import RepoFile, RepoFinding, Task


_SEVERITY_MAP = {
    "low": 2,
    "med": 3,
    "medium": 3,
    "high": 4,
    "critical": 5,
}


def _fingerprint(f: dict[str, Any]) -> str:
    key = "|".join(
        [
            str(f.get("category") or ""),
            str(f.get("severity") or ""),
            str(f.get("title") or ""),
            str(f.get("file_path") or f.get("path") or ""),
            str(f.get("line_start") or f.get("line") or ""),
            str(f.get("line_end") or ""),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:64]


def _coerce_severity(x: Any) -> int:
    """
    Normalize severity into int 1..5.
    Accepts:
      - int/float
      - numeric strings: "4"
      - words: low/med/high/critical
    """
    if x is None:
        return 3

    if isinstance(x, (int, float)):
        v = int(x)
        return max(1, min(5, v))

    s = str(x).strip().lower()
    if not s:
        return 3

    if s in _SEVERITY_MAP:
        return _SEVERITY_MAP[s]

    try:
        v = int(float(s))
        return max(1, min(5, v))
    except Exception:
        return 3


def _pick_top_files(db: Session, snapshot_id: int) -> list[RepoFile]:
    """
    Prefer files that have content and are larger (often more "core" code).
    Hard cap to keep prompt bounded.
    """
    q = (
        db.query(RepoFile)
        .filter(RepoFile.snapshot_id == snapshot_id)
        .filter(RepoFile.content_text.isnot(None))
        .order_by(RepoFile.size.desc())
    )
    files = q.limit(settings.REPO_TASK_MAX_FILES * 3).all()
    return files[: settings.REPO_TASK_MAX_FILES]


def _build_prompt_context(files: list[RepoFile]) -> str:
    chunks: list[str] = []
    total = 0

    for rf in files:
        text = rf.content_text or ""
        excerpt = text[: settings.REPO_TASK_EXCERPT_CHARS]

        block = f"\n--- FILE: {rf.path}\n{excerpt}\n"
        if total + len(block) > settings.REPO_TASK_MAX_TOTAL_CHARS:
            break

        chunks.append(block)
        total += len(block)

    return "".join(chunks)


def _normalize_finding(f: dict[str, Any]) -> dict[str, Any] | None:
    """
    Drop un-actionable garbage:
      - missing title
      - missing file_path/path
    Normalize:
      - severity -> int 1..5
      - line_start/line -> int|None
    """
    if not isinstance(f, dict):
        return None

    title = str(f.get("title") or "").strip()
    if not title:
        return None

    path = f.get("file_path") or f.get("path")
    path = str(path or "").strip()
    if not path or path == "null":
        return None

    sev = _coerce_severity(f.get("severity"))

    line = None
    line_start = f.get("line_start") or f.get("line")
    if line_start is not None:
        try:
            line = int(line_start)
        except Exception:
            line = None

    return {
        "category": str(f.get("category") or "maintainability")[:48],
        "severity": int(sev),
        "title": title[:240],
        "path": path,
        "line": line,
        "evidence": (str(f.get("evidence")) if f.get("evidence") else None),
        "recommendation": (str(f.get("recommendation")) if f.get("recommendation") else None),
        "acceptance": (str(f.get("acceptance")) if f.get("acceptance") else None),
    }


async def scan_snapshot_to_findings(db: Session, snapshot_id: int) -> dict[str, int]:
    """
    Core worker: calls LLM and inserts RepoFinding rows.
    Returns {inserted, total_findings}.
    """
    files = _pick_top_files(db, snapshot_id)
    context = _build_prompt_context(files)

    resp = await chat_completion_json(
        system=REPO_FINDINGS_SYSTEM,
        user=repo_findings_user(context),
    )

    findings = resp.get("findings") or []
    inserted = 0

    for f_raw in findings:
        if not isinstance(f_raw, dict):
            continue

        f = _normalize_finding(f_raw)
        if not f:
            continue

        fp = _fingerprint(
            {
                "category": f["category"],
                "severity": f["severity"],
                "title": f["title"],
                "file_path": f["path"],
                "line_start": f["line"] or "",
                "line_end": "",
            }
        )

        exists = (
            db.query(RepoFinding)
            .filter(RepoFinding.snapshot_id == snapshot_id)
            .filter(RepoFinding.fingerprint == fp)
            .first()
        )
        if exists:
            continue

        row = RepoFinding(
            snapshot_id=snapshot_id,
            path=f["path"],
            line=f["line"],
            category=f["category"],
            severity=int(f["severity"]),
            title=f["title"],
            evidence=f["evidence"],
            recommendation=f["recommendation"],
            acceptance=f["acceptance"],
            fingerprint=fp,
            created_at=datetime.utcnow(),
        )
        db.add(row)
        inserted += 1

    db.commit()

    total_findings = db.query(RepoFinding).filter(RepoFinding.snapshot_id == snapshot_id).count()
    return {"inserted": inserted, "total_findings": total_findings}


# --------------------------------------------------------------------
# Public API expected by routes/repo.py (compatibility preserved)
# --------------------------------------------------------------------

async def run_llm_scan(db: Session, snapshot_id: int) -> dict[str, Any]:
    """
    Original public entrypoint (used by older routes).
    """
    return await scan_snapshot_to_findings(db, snapshot_id)


async def run_llm_repo_scan(db: Session, snapshot_id: int) -> dict[str, Any]:
    """
    NEW compatibility alias: routes/repo.py imports this name.
    Keep this forever (or update routes) to avoid import-time crashes.
    """
    return await run_llm_scan(db, snapshot_id)


def list_findings(db: Session, snapshot_id: int, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    q = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .order_by(RepoFinding.id.desc())
    )
    rows = q.offset(offset).limit(limit).all()
    total = db.query(RepoFinding).filter(RepoFinding.snapshot_id == snapshot_id).count()

    return {
        "snapshot_id": snapshot_id,
        "count": int(total),
        "findings": [
            {
                "id": r.id,
                "snapshot_id": r.snapshot_id,
                "path": r.path,
                "line": r.line,
                "category": r.category,
                "severity": _coerce_severity(r.severity),
                "title": r.title,
                "evidence": r.evidence,
                "recommendation": r.recommendation,
                "fingerprint": r.fingerprint,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


def tasks_from_findings(db: Session, snapshot_id: int, project: str) -> dict[str, Any]:
    """
    Convert RepoFinding rows into real Task rows.
    Deterministic; doesn't need an LLM.
    """
    findings = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .order_by(RepoFinding.id.desc())
        .all()
    )

    findings.sort(key=lambda r: _coerce_severity(getattr(r, "severity", None)), reverse=True)

    created = 0
    skipped = 0

    for f in findings:
        fp_tag = f"finding:{f.fingerprint}"

        existing = (
            db.query(Task)
            .filter(Task.project == project)
            .filter(Task.tags.like(f"%{fp_tag}%"))
            .first()
        )
        if existing:
            skipped += 1
            continue

        sev = _coerce_severity(f.severity)
        pri = int(max(1, min(5, sev)))

        link = f"repo://snapshot/{snapshot_id}#{f.path}" + (f":L{f.line}" if f.line else "")
        tags = ",".join(["repo", "autogen", f"category:{f.category}", f"severity:{sev}", fp_tag])

        notes_parts = []
        if f.evidence:
            notes_parts.append(f"Evidence:\n{f.evidence}")
        if f.recommendation:
            notes_parts.append(f"Recommendation:\n{f.recommendation}")
        notes = "\n\n".join(notes_parts) if notes_parts else "Generated from repo findings."

        db.add(
            Task(
                title=f"[RepoFinding] {f.title}",
                notes=notes,
                priority=pri,
                estimated_minutes=90 if pri >= 4 else 45,
                blocks_me=(pri >= 5),
                project=project,
                tags=tags,
                link=link,
                starter="Open the file and reproduce/confirm the issue in 2–5 minutes.",
                dod="Change is implemented + minimal test/verification step is documented in the task notes.",
                created_at=datetime.utcnow(),
                completed=False,
            )
        )
        created += 1

    db.commit()
    return {"snapshot_id": snapshot_id, "project": project, "created": created, "skipped": skipped}
