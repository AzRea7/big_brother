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
            str(f.get("file_path") or ""),
            str(f.get("line_start") or ""),
            str(f.get("line_end") or ""),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:64]


def _pick_top_files(db: Session, snapshot_id: int) -> list[RepoFile]:
    """
    Simple heuristic: prefer files that have content and are larger (often more "core" code).
    We cap hard so prompts stay bounded.
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

    for f in findings:
        if not isinstance(f, dict):
            continue

        fp = _fingerprint(f)

        exists = (
            db.query(RepoFinding)
            .filter(RepoFinding.snapshot_id == snapshot_id)
            .filter(RepoFinding.fingerprint == fp)
            .first()
        )
        if exists:
            continue

        severity_str = str(f.get("severity") or "med").lower().strip()
        sev = _SEVERITY_MAP.get(severity_str, 3)

        path = str(f.get("file_path") or "").strip() or "unknown"
        line = None
        if f.get("line_start"):
            try:
                line = int(f["line_start"])
            except Exception:
                line = None

        row = RepoFinding(
            snapshot_id=snapshot_id,
            path=path,
            line=line,
            category=str(f.get("category") or "maintainability")[:48],
            severity=int(sev),
            title=str(f.get("title") or "Finding")[:240],
            evidence=(str(f.get("evidence")) if f.get("evidence") else None),
            recommendation=(str(f.get("recommendation")) if f.get("recommendation") else None),
            fingerprint=fp,
            created_at=datetime.utcnow(),
        )
        db.add(row)
        inserted += 1

    db.commit()

    total_findings = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .count()
    )
    return {"inserted": inserted, "total_findings": total_findings}


# --------------------------------------------------------------------
# Public API expected by routes/repo.py
# --------------------------------------------------------------------

async def run_llm_scan(db: Session, snapshot_id: int) -> dict[str, Any]:
    return await scan_snapshot_to_findings(db, snapshot_id)


def list_findings(db: Session, snapshot_id: int, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    q = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .order_by(RepoFinding.severity.desc(), RepoFinding.id.desc())
    )
    rows = q.offset(offset).limit(limit).all()
    total = db.query(RepoFinding).filter(RepoFinding.snapshot_id == snapshot_id).count()

    return {
        "snapshot_id": snapshot_id,
        "count": total,
        "findings": [
            {
                "id": r.id,
                "snapshot_id": r.snapshot_id,
                "path": r.path,
                "line": r.line,
                "category": r.category,
                "severity": r.severity,
                "title": r.title,
                "evidence": r.evidence,
                "recommendation": r.recommendation,
                "fingerprint": r.fingerprint,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ],
    }


def tasks_from_findings(db: Session, snapshot_id: int, project: str) -> dict[str, Any]:
    """
    Convert RepoFinding rows into real Task rows.
    This is intentionally deterministic and doesn't need an LLM.
    """
    findings = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .order_by(RepoFinding.severity.desc(), RepoFinding.id.desc())
        .all()
    )

    created = 0
    skipped = 0

    for f in findings:
        # Dedupe: if we already created a task with this fingerprint tag, skip.
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

        # Priority mapping 1..5 (your Task model uses 1..5)
        # RepoFinding.severity is 1..5 already; clamp just in case.
        pri = int(max(1, min(5, f.severity)))

        link = f"repo://snapshot/{snapshot_id}#{f.path}" + (f":L{f.line}" if f.line else "")
        tags = ",".join(
            [
                "repo",
                "autogen",
                f"category:{f.category}",
                f"severity:{f.severity}",
                fp_tag,
            ]
        )

        notes_parts = []
        if f.evidence:
            notes_parts.append(f"Evidence:\n{f.evidence}")
        if f.recommendation:
            notes_parts.append(f"Recommendation:\n{f.recommendation}")
        notes = "\n\n".join(notes_parts) if notes_parts else "Generated from repo findings."

        t = Task(
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
        db.add(t)
        created += 1

    db.commit()

    return {"snapshot_id": snapshot_id, "project": project, "created": created, "skipped": skipped}
