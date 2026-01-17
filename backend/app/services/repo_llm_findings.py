# backend/app/services/repo_llm_findings.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from sqlalchemy.orm import Session

from ..ai.llm import chat_completion_json
from ..ai.prompts import REPO_FINDINGS_SYSTEM, repo_findings_user
from ..config import settings
from ..models import RepoFile, RepoFinding, Task


def _fingerprint(f: dict[str, Any]) -> str:
    """
    Stable fingerprint so the same finding doesn't get re-inserted every scan.
    """
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
    Simple heuristic:
      - only files with stored content_text
      - prefer larger files (more surface area), but we will excerpt anyway
      - cap to REPO_TASK_MAX_FILES
    """
    q = (
        db.query(RepoFile)
        .filter(RepoFile.snapshot_id == snapshot_id)
        .filter(RepoFile.content_text.isnot(None))
    )

    # Some DBs/models use `size` not `size_bytes`. Handle both.
    if hasattr(RepoFile, "size_bytes"):
        q = q.order_by(getattr(RepoFile, "size_bytes").desc())  # type: ignore[attr-defined]
    else:
        q = q.order_by(RepoFile.size.desc())

    files = q.limit(settings.REPO_TASK_MAX_FILES * 3).all()
    return files[: settings.REPO_TASK_MAX_FILES]


def _build_prompt_context(files: list[RepoFile]) -> str:
    chunks: list[str] = []
    total = 0

    for rf in files:
        text = rf.content_text or rf.content or ""
        excerpt = text[: settings.REPO_TASK_EXCERPT_CHARS]

        block = f"\n--- FILE: {rf.path}\n{excerpt}\n"
        if total + len(block) > settings.REPO_TASK_MAX_TOTAL_CHARS:
            break

        chunks.append(block)
        total += len(block)

    return "".join(chunks)


def _severity_rank(sev: str | None) -> int:
    """
    Higher is worse.
    """
    s = (sev or "").lower().strip()
    return {
        "critical": 50,
        "high": 40,
        "medium": 30,
        "low": 20,
        "info": 10,
    }.get(s, 20)


def _task_key(title: str, link: str | None) -> str:
    return (title or "").strip().lower() + "|" + (link or "")


def _repo_link(snapshot_id: int, file_path: str | None, line_start: int | None) -> str | None:
    """
    Your UI uses repo:// links in other places; keep it consistent.
    """
    if not file_path:
        return None
    # We don't have repo/branch/sha here without joining RepoSnapshot; keep it minimal + stable.
    # If you want richer links later, pass snapshot in and build: repo://{repo}?branch=...&commit_sha=...#{path}:L{line}
    if line_start:
        return f"repo://snapshot/{snapshot_id}#{file_path}:L{line_start}"
    return f"repo://snapshot/{snapshot_id}#{file_path}"


async def scan_snapshot_to_findings(db: Session, snapshot_id: int) -> dict[str, int]:
    """
    Core scan: runs LLM over a bounded excerpt set and inserts new RepoFinding rows.
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

        row = RepoFinding(
            snapshot_id=snapshot_id,
            category=str(f.get("category") or "maintainability")[:32],
            severity=str(f.get("severity") or "low")[:16],
            title=str(f.get("title") or "Finding")[:256],
            file_path=(str(f.get("file_path"))[:1024] if f.get("file_path") else None),
            line_start=(int(f["line_start"]) if f.get("line_start") else None),
            line_end=(int(f["line_end"]) if f.get("line_end") else None),
            evidence=(str(f.get("evidence")) if f.get("evidence") else None),
            recommendation=(str(f.get("recommendation")) if f.get("recommendation") else None),
            acceptance=(str(f.get("acceptance")) if f.get("acceptance") else None),
            fingerprint=fp,
            created_at=datetime.utcnow(),
            is_resolved=False,
        )
        db.add(row)
        inserted += 1

    db.commit()

    total_findings = db.query(RepoFinding).filter(RepoFinding.snapshot_id == snapshot_id).count()
    return {"inserted": inserted, "total_findings": total_findings}


# -------------------------------------------------------------------
# Exports expected by routes/repo.py
# -------------------------------------------------------------------

async def run_llm_scan(db: Session, snapshot_id: int) -> dict[str, Any]:
    """
    Backwards-compatible name used by routes.
    """
    return await scan_snapshot_to_findings(db=db, snapshot_id=snapshot_id)


def list_findings(
    db: Session,
    snapshot_id: int,
    limit: int = 50,
    offset: int = 0,
    include_resolved: bool = False,
) -> dict[str, Any]:
    q = db.query(RepoFinding).filter(RepoFinding.snapshot_id == snapshot_id)

    if not include_resolved:
        q = q.filter(RepoFinding.is_resolved.is_(False))

    # Order by severity first, then newest
    q = q.order_by(RepoFinding.severity.desc(), RepoFinding.id.desc())

    rows = q.offset(offset).limit(limit).all()

    def row_to_dict(r: RepoFinding) -> dict[str, Any]:
        return {
            "id": r.id,
            "snapshot_id": r.snapshot_id,
            "category": r.category,
            "severity": r.severity,
            "title": r.title,
            "file_path": getattr(r, "file_path", None),
            "line_start": getattr(r, "line_start", None),
            "line_end": getattr(r, "line_end", None),
            "evidence": r.evidence,
            "recommendation": r.recommendation,
            "acceptance": getattr(r, "acceptance", None),
            "fingerprint": r.fingerprint,
            "created_at": (r.created_at.isoformat() if r.created_at else None),
            "is_resolved": bool(getattr(r, "is_resolved", False)),
        }

    return {
        "snapshot_id": snapshot_id,
        "count": len(rows),
        "findings": [row_to_dict(r) for r in rows],
    }


async def tasks_from_findings(
    db: Session,
    snapshot_id: int,
    project: str,
    limit: int = 25,
) -> tuple[int, int]:
    """
    Convert findings -> Tasks with dedupe.
    """
    findings = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .filter(RepoFinding.is_resolved.is_(False))
        .all()
    )

    if not findings:
        return 0, 0

    # Sort: critical/high first, then newest
    findings.sort(key=lambda f: (_severity_rank(f.severity), f.id), reverse=True)

    existing = db.query(Task).filter(Task.project == project).all()
    existing_keys = {_task_key(t.title or "", t.link) for t in existing}

    created = 0
    skipped = 0

    for f in findings[:limit]:
        file_path = getattr(f, "file_path", None)
        line_start = getattr(f, "line_start", None)

        link = _repo_link(snapshot_id, file_path, line_start)

        # Build a strong ticket
        title = f"[{(f.severity or 'low').upper()}] {f.title}".strip()[:240]
        notes_parts = []
        if f.category:
            notes_parts.append(f"Category: {f.category}")
        if file_path:
            notes_parts.append(f"File: {file_path}")
        if line_start:
            notes_parts.append(f"Lines: {line_start}" + (f"-{f.line_end}" if getattr(f, "line_end", None) else ""))

        if f.evidence:
            notes_parts.append("\nEvidence:\n" + str(f.evidence).strip())
        if f.recommendation:
            notes_parts.append("\nRecommendation:\n" + str(f.recommendation).strip())
        if getattr(f, "acceptance", None):
            notes_parts.append("\nAcceptance:\n" + str(getattr(f, "acceptance")).strip())

        notes = "\n".join(notes_parts).strip()

        # Reasonable defaults
        sev = (f.severity or "").lower().strip()
        priority = 5 if sev in ("critical", "high") else 4 if sev == "medium" else 3
        est = 120 if sev in ("critical", "high") else 90 if sev == "medium" else 60
        blocks = sev in ("critical", "high")

        key = _task_key(title, link)
        if key in existing_keys:
            skipped += 1
            continue

        db.add(
            Task(
                title=title,
                notes=notes,
                project=project,
                tags=f"repo,autogen,findings,{(f.category or 'misc')}",
                link=link,
                priority=priority,
                estimated_minutes=est,
                blocks_me=blocks,
                starter="Open the linked file/lines, reproduce or confirm the issue, then implement the recommendation.",
                dod="Acceptance criteria met; tests updated/added where applicable; no regression in CI.",
                completed=False,
            )
        )
        existing_keys.add(key)
        created += 1

    db.commit()
    return created, skipped
