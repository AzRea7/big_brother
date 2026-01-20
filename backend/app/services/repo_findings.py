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
from ..ai.repo_tasks import generate_repo_tasks_json
from ..config import settings
from ..models import RepoChunk, RepoFile, RepoFinding, RepoSnapshot, Task

from .code_signals import compute_signal_counts_for_files, count_markers_in_text
from .repo_rag import search_chunks


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
    cap = int(getattr(settings, "REPO_SCAN_MAX_FILES", 14))
    return text_files[:cap]


def scan_repo_findings_llm(db: Session, snapshot_id: int, max_files: int = 14) -> dict[str, Any]:
    """
    LLM-driven findings: pick a small set of high-signal files, send excerpts, store RepoFinding rows.
    """
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        return {"snapshot_id": snapshot_id, "created": 0, "skipped": 0, "detail": "snapshot not found"}

    files = db.scalars(select(RepoFile).where(RepoFile.snapshot_id == snapshot_id)).all()
    picked = _pick_files_for_scan(files)[:max_files]

    payload_files: list[dict[str, Any]] = []
    for f in picked:
        txt = f.content_text or (f.content.decode("utf-8", errors="replace") if f.content else "")
        excerpt = (txt or "")[: int(getattr(settings, "REPO_SCAN_EXCERPT_CHARS", 900))]
        payload_files.append({"path": f.path, "excerpt": excerpt})

    llm = LLMClient()
    if not llm.enabled():
        raise RuntimeError("LLM not enabled. Set LLM_ENABLED=true, OPENAI_BASE_URL, OPENAI_MODEL.")

    # NOTE: keep your existing prompt contract (your repo already has this)
    system = "You are a strict code reviewer. Output JSON only."
    user = {"snapshot_id": snapshot_id, "repo": snap.repo, "branch": snap.branch, "files": payload_files}

    raw_text = llm.chat(system=system, user=json.dumps(user), temperature=0.2, max_tokens=1800)

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
        acceptance = str(f.get("acceptance") or "").strip()

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
                acceptance=acceptance,
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
    """
    Deterministic conversion: RepoFinding -> Task (no LLM).
    This is your safe fallback / baseline.
    """
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

        db.add(
            Task(
                project=project,
                title=title,
                notes=(f.recommendation or "")[:2000],
                link=link,
                tags=f"repo,autogen,finding,{(f.category or 'general')}",
                priority=int(f.severity or 3),
                estimated_minutes=45,
                blocks_me=True if int(f.severity or 3) >= 4 else False,
                starter="Open the link and confirm the issue exists in code/logs.",
                dod=(f.acceptance or "Change implemented; verified with a test/curl/log evidence.")[:1000],
            )
        )
        existing_keys.add(key)
        created += 1

    db.commit()
    return {"snapshot_id": snapshot_id, "created": created, "skipped": skipped}


# -------------------------
# ✅ Finding-driven retrieval → LLM task generation
# -------------------------

@dataclass
class SuggestedTask:
    title: str
    notes: str
    starter: str
    dod: str
    priority: int
    estimated_minutes: int
    link: str | None = None
    tags: str | None = None
    path: str | None = None
    line: int | None = None


def _finding_query(f: RepoFinding) -> str:
    """
    Turn a finding into a retrieval query (title/category/evidence/reco).
    This is the upgrade you asked for: retrieval keyed off actual findings,
    not generic seed queries.
    """
    parts = [
        f.title or "",
        f.category or "",
        f.path or "",
        (f.evidence or "")[:180],
        (f.recommendation or "")[:180],
    ]
    q = " ".join(p for p in parts if p).strip()
    return q[:600]


def _format_finding_block(f: RepoFinding) -> str:
    return (
        f"[FINDING]\n"
        f"id={f.id} severity={f.severity} category={f.category}\n"
        f"path={f.path} line={f.line}\n"
        f"title={f.title}\n"
        f"evidence={(f.evidence or '')[:400]}\n"
        f"recommendation={(f.recommendation or '')[:400]}\n"
        f"acceptance={(f.acceptance or '')[:300]}\n"
    )


async def _llm_generate_tasks_from_findings_with_retrieval(
    db: Session,
    snapshot: RepoSnapshot,
    *,
    project: str,
    max_findings: int = 10,
    chunks_per_finding: int = 3,
) -> list[SuggestedTask]:
    """
    Generates tasks using:
      findings → query → retrieve chunks → send excerpts to LLM

    Uses your existing generate_repo_tasks_json() contract by creating synthetic
    "file_summaries" with excerpt text drawn from chunks.

    If chunks do not exist yet, raise (caller can fall back).
    """
    if not getattr(settings, "LLM_ENABLED", False):
        return []
    if not getattr(settings, "OPENAI_MODEL", None):
        return []

    # We need chunks to exist for RAG.
    chunk_exists = db.scalar(select(RepoChunk.id).where(RepoChunk.snapshot_id == snapshot.id).limit(1))
    if chunk_exists is None:
        raise RuntimeError("No RepoChunk rows found. Run /debug/repo/chunks/build first.")

    findings = db.scalars(
        select(RepoFinding)
        .where(RepoFinding.snapshot_id == snapshot.id)
        .where(RepoFinding.is_resolved == False)  # noqa: E712
        .order_by(RepoFinding.severity.desc(), RepoFinding.id.desc())
        .limit(max_findings)
    ).all()

    if not findings:
        return []

    # Build synthetic evidence “files”
    file_summaries: list[dict[str, Any]] = []
    for f in findings:
        q = _finding_query(f)

        # First try scoped to the finding's file path (when present)
        search = search_chunks(
            db=db,
            snapshot_id=snapshot.id,
            query=q,
            top_k=chunks_per_finding,
            mode="auto",
            path_contains=f.path,
        )
        chunks = search.get("results", []) or []

        # If nothing found, broaden search
        if not chunks:
            search = search_chunks(db=db, snapshot_id=snapshot.id, query=q, top_k=chunks_per_finding, mode="auto")
            chunks = search.get("results", []) or []

        finding_header = _format_finding_block(f)

        for c in chunks:
            excerpt = (
                f"{finding_header}\n"
                f"[CODE_CHUNK]\n"
                f"path={c.get('path')} lines={c.get('start_line')}-{c.get('end_line')}\n"
                f"{c.get('chunk_text','')}"
            )
            excerpt = excerpt[: int(getattr(settings, "REPO_TASK_EXCERPT_CHARS", 1200))]

            sig = count_markers_in_text(excerpt)
            file_summaries.append(
                {
                    "path": str(c.get("path") or f.path or "unknown"),
                    "excerpt": excerpt,
                    **sig,
                }
            )

    # Provide global signal counts (stable)
    files = db.scalars(
        select(RepoFile)
        .where(RepoFile.snapshot_id == snapshot.id)
        .where(RepoFile.content_kind == "text")
    ).all()
    signal_counts = compute_signal_counts_for_files(files)

    raw = await generate_repo_tasks_json(
        repo_name=snapshot.repo,
        branch=snapshot.branch,
        commit_sha=snapshot.commit_sha,
        snapshot_id=snapshot.id,
        signal_counts=signal_counts,
        file_summaries=file_summaries,
    )

    if isinstance(raw, str):
        raw = json.loads(raw)

    out: list[SuggestedTask] = []
    for t in (raw or {}).get("tasks", []):
        title = str(t.get("title") or "").strip()[:240]
        if not title:
            continue
        out.append(
            SuggestedTask(
                title=title,
                notes=str(t.get("notes") or "")[:4000],
                starter=str(t.get("starter") or "MISSING")[:1000],
                dod=str(t.get("dod") or "MISSING")[:1000],
                priority=max(1, min(5, int(t.get("priority") or 3))),
                estimated_minutes=max(15, min(240, int(t.get("estimated_minutes") or 60))),
                link=str(t.get("link") or "")[:1000] or None,
                tags=str(t.get("tags") or "")[:300] or None,
                path=str(t.get("path") or "")[:600] or None,
                line=int(t.get("line")) if isinstance(t.get("line"), int) else None,
            )
        )
    return out


async def generate_tasks_from_findings_llm(
    db: Session,
    snapshot_id: int,
    *,
    project: str = "haven",
    max_findings: int = 10,
    chunks_per_finding: int = 3,
) -> dict[str, Any]:
    """
    Public service: use retrieval-driven LLM generation, then CREATE Task rows (deduped).
    """
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        return {"snapshot_id": snapshot_id, "created": 0, "skipped": 0, "detail": "snapshot not found"}

    suggestions = await _llm_generate_tasks_from_findings_with_retrieval(
        db=db,
        snapshot=snap,
        project=project,
        max_findings=max_findings,
        chunks_per_finding=chunks_per_finding,
    )

    if not suggestions:
        return {"snapshot_id": snapshot_id, "created": 0, "skipped": 0, "detail": "no suggestions"}

    existing_tasks = db.scalars(select(Task).where(Task.project == project)).all()
    existing_keys = {(t.title or "").strip().lower() + "|" + (t.link or "") for t in existing_tasks}

    created = 0
    skipped = 0
    created_titles: list[str] = []

    for s in suggestions:
        link = s.link or ""
        key = s.title.strip().lower() + "|" + link
        if key in existing_keys:
            skipped += 1
            continue

        db.add(
            Task(
                project=project,
                title=s.title[:240],
                notes=(s.notes or "")[:4000],
                link=(s.link or "")[:1000] or None,
                tags=(s.tags or "repo,autogen,llm")[:300],
                priority=int(s.priority or 3),
                estimated_minutes=int(s.estimated_minutes or 60),
                blocks_me=True if int(s.priority or 3) >= 4 else False,
                starter=(s.starter or "MISSING")[:1000],
                dod=(s.dod or "MISSING")[:1000],
            )
        )
        existing_keys.add(key)
        created += 1
        created_titles.append(s.title[:240])

    db.commit()

    return {
        "snapshot_id": snapshot_id,
        "created": created,
        "skipped": skipped,
        "titles": created_titles,
        "mode": "llm+retrieval",
        "max_findings": max_findings,
        "chunks_per_finding": chunks_per_finding,
    }
