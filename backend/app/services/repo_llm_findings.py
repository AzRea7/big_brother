# backend/app/services/repo_llm_findings.py
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from ..ai.llm import chat_completion_json
from ..ai.prompts import REPO_FINDINGS_SYSTEM, repo_findings_user
from ..config import settings
from ..models import RepoFile, RepoFinding


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
    # Simple heuristic: prefer files that have content and are not huge.
    q = (
        db.query(RepoFile)
        .filter(RepoFile.snapshot_id == snapshot_id)
        .filter(RepoFile.content_text.isnot(None))
        .order_by(RepoFile.size_bytes.desc())
    )
    files = q.limit(settings.REPO_TASK_MAX_FILES * 3).all()

    # Keep it deterministic and cap to REPO_TASK_MAX_FILES.
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

    total_findings = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .count()
    )

    return {"inserted": inserted, "total_findings": total_findings}
