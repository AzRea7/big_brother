# backend/app/services/repo_findings.py
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..ai.llm import LLMClient
from ..config import settings
from ..models import RepoFile, RepoSnapshot, Task

# If you already have RepoFinding model in models.py, import it.
# If not, you MUST add it (see note at bottom).
from ..models import RepoFinding  # type: ignore


def _repo_link(snapshot: RepoSnapshot, path: str, line: int | None = None) -> str:
    base = f"repo://{snapshot.repo}?branch={snapshot.branch}&commit_sha={snapshot.commit_sha or ''}"
    return f"{base}#{path}:L{line}" if line is not None else f"{base}#{path}"


def _fingerprint(snapshot_id: int, path: str, line: int | None, title: str) -> str:
    raw = f"{snapshot_id}|{path}|{line or 0}|{title}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:40]


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_json_object_lenient(raw: str) -> dict[str, Any]:
    """
    Local models often wrap JSON in text/fences.
    We take the first {...} blob and parse it.
    """
    s = _strip_code_fences(raw)
    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError(f"No JSON object found. First 400 chars: {raw[:400]}")
    return json.loads(s[first : last + 1])


def _pick_files_for_scan(files: list[RepoFile]) -> list[RepoFile]:
    """
    Choose a small set of high-signal files.
    We bias toward:
      - backend/app/* (fastapi, db, routes, services)
      - infra (docker, ci)
      - auth/config/env
    """
    def score(p: str) -> int:
        p = (p or "").lower()
        s = 0
        if "backend" in p or "/app/" in p:
            s += 30
        if "routes" in p or "main.py" in p:
            s += 25
        if "db" in p or "models" in p or "migrations" in p:
            s += 22
        if "auth" in p or "jwt" in p or "api_key" in p:
            s += 20
        if "docker" in p or "compose" in p:
            s += 12
        if ".github/workflows" in p or "ci" in p:
            s += 12
        if p.endswith(".py"):
            s += 5
        return s

    text_files = [f for f in files if f.content_kind == "text" and (f.content_text or f.content)]
    text_files.sort(key=lambda f: score(f.path), reverse=True)

    max_files = int(getattr(settings, "REPO_TASK_MAX_FILES", 18))
    return text_files[:max_files]


async def run_llm_scan(db: Session, snapshot_id: int) -> dict[str, Any]:
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        return {"created": 0, "skipped": 0, "detail": "snapshot not found"}

    files = db.scalars(select(RepoFile).where(RepoFile.snapshot_id == snapshot_id)).all()
    chosen = _pick_files_for_scan(files)

    excerpt_chars = int(getattr(settings, "REPO_TASK_EXCERPT_CHARS", 800))
    max_total = int(getattr(settings, "REPO_TASK_MAX_TOTAL_CHARS", 12_000))

    payload_files: list[dict[str, Any]] = []
    total_chars = 0
    for f in chosen:
        text = (f.content_text or f.content or "")
        if not text:
            continue
        excerpt = text[:excerpt_chars]
        total_chars += len(excerpt)
        if total_chars > max_total:
            break
        payload_files.append({"path": f.path, "excerpt": excerpt})

    system = (
        "You are a senior staff engineer performing a production-readiness code review. "
        "Return ONLY JSON. No markdown. No commentary."
    )

    user = {
        "repo": snap.repo,
        "branch": snap.branch,
        "commit_sha": snap.commit_sha,
        "snapshot_id": snap.id,
        "instructions": [
            "Analyze the provided code excerpts and infer production issues + missing pieces.",
            "Do NOT invent files or functions that are not shown. Use only what you see.",
            "Prefer actionable findings: security, correctness, reliability, observability, tests, config, deployment.",
            "Each finding must have: path, line (or null), category, severity 1-5 (5 is highest), title, evidence, recommendation.",
            "Fingerprint should be stable-ish (but you can omit; backend will compute if missing).",
        ],
        "files": payload_files,
        "output_schema": {
            "findings": [
                {
                    "path": "string",
                    "line": "int|null",
                    "category": "string",
                    "severity": "int 1..5",
                    "title": "string",
                    "evidence": "string",
                    "recommendation": "string",
                    "fingerprint": "string(optional)",
                }
            ]
        },
    }

    llm = LLMClient()
    if not llm.enabled():
        raise RuntimeError("LLM not enabled. Set LLM_ENABLED=true, OPENAI_BASE_URL, OPENAI_MODEL.")

    raw_text = await llm.chat(system=system, user=json.dumps(user), temperature=0.2, max_tokens=1800)

    try:
        data = json.loads(_strip_code_fences(raw_text))
    except Exception:
        data = _extract_json_object_lenient(raw_text)

    findings = data.get("findings") or []
    if not isinstance(findings, list):
        raise RuntimeError(f"LLM returned invalid findings: {str(data)[:400]}")

    existing = db.scalars(select(RepoFinding).where(RepoFinding.snapshot_id == snapshot_id)).all()
    existing_keys = {f.fingerprint for f in existing if f.fingerprint}

    created = 0
    skipped = 0

    for f in findings:
        path = str(f.get("path") or "").strip()[:600]
        line = f.get("line")
        line_int = int(line) if isinstance(line, int) else None
        category = str(f.get("category") or "general").strip()[:64]
        severity = int(f.get("severity") or 3)
        severity = max(1, min(5, severity))
        title = str(f.get("title") or "Finding").strip()[:240]
        evidence = str(f.get("evidence") or "").strip()
        recommendation = str(f.get("recommendation") or "").strip()

        fp = str(f.get("fingerprint") or "").strip()
        if not fp:
            fp = _fingerprint(snapshot_id, path, line_int, title)

        if fp in existing_keys:
            skipped += 1
            continue

        db.add(
            RepoFinding(
                snapshot_id=snapshot_id,
                path=path,
                line=line_int,
                category=category,
                severity=severity,
                title=title,
                evidence=evidence,
                recommendation=recommendation,
                fingerprint=fp,
            )
        )
        existing_keys.add(fp)
        created += 1

    db.commit()

    return {
        "snapshot_id": snapshot_id,
        "repo": snap.repo,
        "created": created,
        "skipped": skipped,
        "scanned_files": len(payload_files),
    }


def list_findings(db: Session, snapshot_id: int, limit: int = 50) -> list[RepoFinding]:
    return (
        db.scalars(
            select(RepoFinding)
            .where(RepoFinding.snapshot_id == snapshot_id)
            .order_by(RepoFinding.severity.desc(), RepoFinding.id.desc())
            .limit(limit)
        )
        .all()
    )


def tasks_from_findings(db: Session, snapshot_id: int, project: str = "haven", limit: int = 12) -> dict[str, Any]:
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        return {"created": 0, "skipped": 0, "detail": "snapshot not found"}

    findings = list_findings(db, snapshot_id=snapshot_id, limit=limit)

    existing_tasks = db.scalars(select(Task).where(Task.project == project)).all()
    existing_keys = {(t.title or "").strip().lower() + "|" + (t.link or "") for t in existing_tasks}

    created = 0
    skipped = 0

    for f in findings:
        link = _repo_link(snap, f.path or "unknown", f.line)
        title = f"[P{int(f.severity or 3)}] {f.title}".strip()[:240]
        key = title.lower() + "|" + link

        if key in existing_keys:
            skipped += 1
            continue

        notes = (
            f"**Category:** {f.category}\n"
            f"**Severity:** {f.severity}/5\n\n"
            f"**Evidence:**\n{(f.evidence or '').strip()}\n\n"
            f"**Recommendation:**\n{(f.recommendation or '').strip()}\n"
        )

        db.add(
            Task(
                title=title,
                notes=notes,
                project=project,
                tags=f"repo,autogen,findings,{(f.category or 'general')}",
                link=link,
                priority=min(5, max(1, int(f.severity or 3))),
                estimated_minutes=90 if int(f.severity or 3) >= 4 else 60,
                blocks_me=bool(int(f.severity or 3) >= 5),
                starter="Open the linked file/line, verify the evidence, identify the correct fix scope.",
                dod="Fix implemented, regression risk assessed, and a test/log/guardrail added if applicable.",
                completed=False,
            )
        )
        existing_keys.add(key)
        created += 1

    db.commit()
    return {"snapshot_id": snapshot_id, "project": project, "created": created, "skipped": skipped}
