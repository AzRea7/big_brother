# backend/app/services/repo_findings.py
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..ai.llm import chat_completion_json
from ..ai.prompts import REPO_FINDINGS_SYSTEM, repo_findings_user
from ..ai.repo_tasks import generate_repo_tasks_json
from ..config import settings
from ..models import RepoChunk, RepoFinding, RepoSnapshot, Task


@dataclass(frozen=True)
class ChunkHit:
    id: int
    snapshot_id: int
    path: str
    start_line: int
    end_line: int
    chunk_text: str
    symbols_json: Optional[str]
    score: Optional[float]


@dataclass
class SuggestedTask:
    title: str
    notes: str
    priority: int
    estimated_minutes: int
    blocks_me: bool
    tags: str
    starter: str
    dod: str
    link: Optional[str] = None
    path: Optional[str] = None
    line: Optional[int] = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SuggestedTask":
        def _int(x: Any, default: int) -> int:
            try:
                return int(x)
            except Exception:
                return default

        def _bool(x: Any, default: bool) -> bool:
            if isinstance(x, bool):
                return x
            s = str(x).strip().lower()
            if s in {"true", "1", "yes", "y"}:
                return True
            if s in {"false", "0", "no", "n"}:
                return False
            return default

        title = str(d.get("title") or "").strip()[:240]
        notes = str(d.get("notes") or "").strip()[:4000]
        tags = str(d.get("tags") or "").strip()[:800]
        starter = str(d.get("starter") or "").strip()[:800]
        dod = str(d.get("dod") or "").strip()[:800]

        path = d.get("path")
        path = str(path).strip() if path is not None else None
        if path:
            path = path[:400]

        line = d.get("line")
        if line is not None:
            try:
                line = int(line)
            except Exception:
                line = None

        return SuggestedTask(
            title=title or "Repo task",
            notes=notes or "Generated from repo signals.",
            priority=max(1, min(5, _int(d.get("priority"), 3))),
            estimated_minutes=max(10, _int(d.get("estimated_minutes"), 60)),
            blocks_me=_bool(d.get("blocks_me"), False),
            tags=tags,
            starter=starter or "Open the referenced file and confirm the issue in 2–5 minutes.",
            dod=dod or "Verify: pytest -q",
            link=str(d.get("link")).strip()[:500] if d.get("link") else None,
            path=path or None,
            line=line,
        )


# -------------------------
# Basic repo snapshot helpers
# -------------------------

def repo_name_for_snapshot(db: Session, snapshot_id: int) -> Optional[str]:
    row = db.query(RepoSnapshot).filter(RepoSnapshot.id == snapshot_id).first()
    return getattr(row, "repo", None) if row else None


def branch_for_snapshot(db: Session, snapshot_id: int) -> Optional[str]:
    row = db.query(RepoSnapshot).filter(RepoSnapshot.id == snapshot_id).first()
    return getattr(row, "branch", None) if row else None


def commit_sha_for_snapshot(db: Session, snapshot_id: int) -> Optional[str]:
    row = db.query(RepoSnapshot).filter(RepoSnapshot.id == snapshot_id).first()
    return getattr(row, "commit_sha", None) if row else None


def _repo_link(snapshot_id: int, path: Optional[str], line: Optional[int]) -> str:
    p = (path or "").strip()
    if not p:
        return f"repo://snapshot/{snapshot_id}"
    if line is None:
        return f"repo://snapshot/{snapshot_id}#{p}"
    return f"repo://snapshot/{snapshot_id}#{p}:L{int(line)}"


# -------------------------
# Signals + retrieval
# -------------------------

_SIGNAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "todo": re.compile(r"\bTODO\b", re.IGNORECASE),
    "fixme": re.compile(r"\bFIXME\b", re.IGNORECASE),
    "hack": re.compile(r"\bHACK\b", re.IGNORECASE),
    "xxx": re.compile(r"\bXXX\b", re.IGNORECASE),
    "bug": re.compile(r"\bBUG\b", re.IGNORECASE),
    "note": re.compile(r"\bNOTE\b", re.IGNORECASE),
    "dotdotdot": re.compile(r"\.\.\.", re.IGNORECASE),
    "auth": re.compile(r"\bauth\b|\bauthoriz|\btoken\b|\bapi[-_ ]?key\b|\bx-api-key\b|\bbearer\b|\bjwt\b", re.IGNORECASE),
    "cors": re.compile(r"\bcors\b", re.IGNORECASE),
    "csrf": re.compile(r"\bcsrf\b", re.IGNORECASE),
    "timeout": re.compile(r"\btimeout\b", re.IGNORECASE),
    "retry": re.compile(r"\bretry\b|\bbackoff\b", re.IGNORECASE),
    "rate_limit": re.compile(r"\brate limit\b|\b429\b|\bthrottle\b|\bquota\b", re.IGNORECASE),
    "validation": re.compile(r"\bvalidate\b|\bpydantic\b|\bschema\b|\binput\b|\brequest\b", re.IGNORECASE),
    "logging": re.compile(r"\blog(ger|ging)?\b", re.IGNORECASE),
    "metrics": re.compile(r"\bmetrics?\b|\bprometheus\b", re.IGNORECASE),
    "db": re.compile(r"\bsqlalchemy\b|\bselect\b|\bjoin\b|\bmigration\b|\balembic\b|\bpostgres\b|\bsql\b", re.IGNORECASE),
    "nplus1": re.compile(r"\bn\+1\b|\bnplus1\b", re.IGNORECASE),
    "tests": re.compile(r"\bpytest\b|\btest_\b|\btests/\b", re.IGNORECASE),
    "ci": re.compile(r"\bgithub actions\b|\bworkflow\b|\bci\b", re.IGNORECASE),
    "docker": re.compile(r"\bdockerfile\b|\bdocker-compose\b|\bdocker compose\b", re.IGNORECASE),
    "config": re.compile(r"\bsettings\b|\benv\b|\bconfig\b|\bdotenv\b", re.IGNORECASE),
    "secrets": re.compile(r"\bsecret\b|\bpassword\b|\btoken\b|\bapi[-_ ]?key\b|\bprivate key\b", re.IGNORECASE),
}


def count_markers_in_text(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, pat in _SIGNAL_PATTERNS.items():
        out[k] = len(pat.findall(text or ""))
    return out


def signal_counts_for_snapshot(db: Session, snapshot_id: int) -> dict[str, Any]:
    rows = db.query(RepoChunk).filter(RepoChunk.snapshot_id == snapshot_id).all()
    totals: dict[str, int] = {k: 0 for k in _SIGNAL_PATTERNS.keys()}
    total_files = 0
    seen_paths: set[str] = set()

    for r in rows:
        if r.path and r.path not in seen_paths:
            seen_paths.add(r.path)
            total_files += 1
        c = count_markers_in_text(r.chunk_text or "")
        for k, v in c.items():
            totals[k] += int(v)

    return {"total_files": total_files, "signals": totals}


def search_chunks(
    db: Session,
    *,
    snapshot_id: int,
    query: str,
    limit: int = 20,
    mode: str = "auto",
) -> tuple[str, list[ChunkHit]]:
    """
    Retrieval provider socket:
      - today: Postgres-ish FTS fallback to LIKE
      - later: embeddings can be added behind mode="auto" with merge/dedupe

    Returns (mode_used, hits).
    """
    q = (query or "").strip()
    if not q:
        return ("none", [])

    # Minimal “FTS-like” fallback:
    # (In your real implementation, this should call your proper pg FTS query.)
    like = f"%{q[:128]}%"
    rows = (
        db.query(RepoChunk)
        .filter(RepoChunk.snapshot_id == snapshot_id)
        .filter(RepoChunk.chunk_text.ilike(like))
        .limit(limit)
        .all()
    )

    hits: list[ChunkHit] = []
    for r in rows:
        hits.append(
            ChunkHit(
                id=r.id,
                snapshot_id=r.snapshot_id,
                path=r.path,
                start_line=r.start_line,
                end_line=r.end_line,
                chunk_text=r.chunk_text or "",
                symbols_json=getattr(r, "symbols_json", None),
                score=1.0,
            )
        )
    return ("fts_like", hits)


# -------------------------
# Findings list (existing)
# -------------------------

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
                "severity": int(r.severity or 3),
                "title": r.title,
                "evidence": r.evidence,
                "recommendation": r.recommendation,
                "acceptance": getattr(r, "acceptance", None),
                "fingerprint": r.fingerprint,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


# -------------------------
# Deterministic findings -> tasks (existing)
# -------------------------

def tasks_from_findings(db: Session, snapshot_id: int, project: str) -> dict[str, Any]:
    findings = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .order_by(RepoFinding.id.desc())
        .all()
    )
    findings.sort(key=lambda r: int(getattr(r, "severity", 3) or 3), reverse=True)

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

        sev = int(getattr(f, "severity", 3) or 3)
        pri = int(max(1, min(5, sev)))

        link = _repo_link(snapshot_id, f.path, f.line)
        tags = ",".join(["repo", "autogen", f"category:{f.category}", f"severity:{sev}", fp_tag])

        notes_parts = []
        if f.evidence:
            notes_parts.append(f"Evidence:\n{f.evidence}")
        if f.recommendation:
            notes_parts.append(f"Recommendation:\n{f.recommendation}")
        notes_parts.append(getattr(f, "acceptance", None) or "Verify: pytest -q")

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
                dod="Verify: pytest -q",
                created_at=datetime.utcnow(),
                completed=False,
            )
        )
        created += 1

    db.commit()
    return {"snapshot_id": snapshot_id, "project": project, "created": created, "skipped": skipped}


# -------------------------
# LLM task generation (UPGRADED: chunk-grounded + traceable)
# -------------------------

def _specificity_score(task: "SuggestedTask") -> int:
    """Heuristic: how actionable/traceable a generated task is."""
    score = 0
    if (task.path or "").strip() and task.path != "unknown":
        score += 1
    if task.line is not None:
        score += 1

    notes = (task.notes or "")
    if "[EVIDENCE" in notes or "EVIDENCE:" in notes:
        score += 1

    dod = (task.dod or "")
    if re.search(r"\bpytest\b|\bcurl\b|\buvicorn\b|\bmake\b|\bpython\b", dod):
        score += 1
    return score


def _ensure_dod_is_testable(task: "SuggestedTask") -> "SuggestedTask":
    """Make DoD falsifiable: always include a runnable verification command."""
    dod = (task.dod or "").strip()
    if not dod or dod.lower() in {"false", "true", "null", "none"}:
        dod = ""

    if not re.search(r"\bpytest\b|\bcurl\b|\bmake\b|\bpython\b", dod):
        # Default to pytest; the point is to give the PR bot a pass/fail hook later.
        dod = (dod + " " if dod else "") + "Verify: pytest -q"
    task.dod = dod[:600]
    return task


def _prepend_evidence(task: "SuggestedTask", evidence_lines: list[str]) -> "SuggestedTask":
    """Ensure notes contain evidence anchors so humans can trust/act quickly."""
    notes = (task.notes or "").strip()
    if "[EVIDENCE" in notes:
        return task
    ev = "\n".join([ln for ln in evidence_lines if ln.strip()])[:1200]
    if ev:
        notes = (ev + "\n\n" + notes).strip()
    task.notes = notes[:4000]
    return task


async def _llm_generate_tasks_from_findings_with_retrieval(
    db: Session,
    snapshot_id: int,
    project: str,
    *,
    max_findings: int = 6,
    chunks_per_finding: int = 3,
) -> list["SuggestedTask"]:
    """
    Generate tasks from *existing* RepoFinding rows, but force every task to be:
      - grounded in retrieved RepoChunk excerpts (evidence)
      - stamped with (path, line) from a primary chunk (traceability)
      - testable (DoD includes a runnable command)
      - safe for later automation (priority capped if too vague)

    This is the "trust anchor" layer between findings and real Task rows.
    """
    findings = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .order_by(RepoFinding.severity.desc(), RepoFinding.id.desc())
        .limit(max_findings)
        .all()
    )
    if not findings:
        return []

    # Build per-finding retrieval packs (primary + related + global-best)
    packs: list[dict[str, Any]] = []
    primary_for_index: list[dict[str, Any]] = []

    for f in findings:
        # Two-query approach: precision (path/title) + recall (category/reco)
        exact_q = " ".join(
            [
                str(f.path or ""),
                str(f.title or ""),
                str(f.evidence or "")[:200],
            ]
        ).strip()
        recall_q = " ".join(
            [
                str(f.category or ""),
                str(f.recommendation or "")[:300],
                str(f.title or ""),
            ]
        ).strip()

        # Retrieve hits (FTS today). We keep mode so embeddings can plug in later.
        mode1, hits_exact = search_chunks(db, snapshot_id=snapshot_id, query=exact_q, limit=50, mode="auto")
        mode2, hits_recall = search_chunks(db, snapshot_id=snapshot_id, query=recall_q, limit=50, mode="auto")

        # Merge/dedupe hits by (path,start,end)
        merged: list[ChunkHit] = []
        seen = set()
        for h in (hits_exact or []) + (hits_recall or []):
            k = (h.path, int(h.start_line), int(h.end_line))
            if k in seen:
                continue
            seen.add(k)
            merged.append(h)
            if len(merged) >= 50:
                break

        if not merged:
            # No chunks at all => hard fail; tasks would be ungrounded.
            raise RuntimeError("No RepoChunk hits found for finding retrieval. Run /debug/repo/chunks/build first.")

        # Primary chunk: prefer same-file, else best overall.
        same_file = [h for h in merged if (h.path or "").strip() == (f.path or "").strip()]
        primary = same_file[0] if same_file else merged[0]

        # Related chunk: prefer a different file (config/auth/db neighbors), else next best.
        related = None
        for h in merged:
            if h.path != primary.path:
                related = h
                break
        related = related or primary

        # Global-best chunk: just the top merged hit.
        global_best = merged[0]

        primary_for_index.append(
            {
                "path": primary.path,
                "line": int(primary.start_line) if primary.start_line is not None else None,
                "range": f"L{primary.start_line}-L{primary.end_line}",
            }
        )

        def fmt_chunk(label: str, h: ChunkHit) -> str:
            excerpt = (h.chunk_text or "")[:1100]
            return (
                f"[{label}] path={h.path} lines={h.start_line}-{h.end_line} score={h.score}\n"
                f"""{excerpt}"""
            )

        # Pack is a single "file summary" unit; the LLM must produce ONE task per pack.
        pack_text = "\n".join(
            [
                f"[FINDING] fingerprint={f.fingerprint} severity={f.severity} category={f.category} path={f.path} line={f.line}",
                f"title: {f.title}",
                (f"evidence: {f.evidence}" if f.evidence else "evidence: (none)"),
                (f"recommendation: {f.recommendation}" if f.recommendation else "recommendation: (none)"),
                "",
                fmt_chunk("CHUNK_PRIMARY", primary),
                "",
                fmt_chunk("CHUNK_RELATED", related),
                "",
                fmt_chunk("CHUNK_GLOBAL", global_best),
            ]
        )

        packs.append(
            {
                "path": primary.path,
                "excerpt": pack_text,
                "signal:category": str(f.category or ""),
                "signal:severity": int(f.severity or 3),
                "signal:fingerprint": str(f.fingerprint or ""),
                "retrieval_mode": f"{mode1}/{mode2}",
            }
        )

    signal_counts = signal_counts_for_snapshot(db, snapshot_id)
    out = await generate_repo_tasks_json(
        repo_name=repo_name_for_snapshot(db, snapshot_id) or "unknown",
        branch=branch_for_snapshot(db, snapshot_id) or "unknown",
        commit_sha=commit_sha_for_snapshot(db, snapshot_id),
        snapshot_id=snapshot_id,
        signal_counts=signal_counts,
        file_summaries=packs,
        extra_evidence=None,
    )

    raw_tasks = out.get("tasks") or []
    suggestions: list[SuggestedTask] = []

    # We *expect* 1 task per finding pack, in order. If the model under/over-shoots,
    # we still salvage deterministically by zipping.
    for i, (t, pri) in enumerate(zip(raw_tasks, primary_for_index)):
        if not isinstance(t, dict):
            continue

        s = SuggestedTask.from_dict(t)

        # Hard enforce traceability: path/line come from primary chunk.
        if not (s.path and s.path != "unknown"):
            s.path = pri["path"]
        if s.line is None and pri.get("line") is not None:
            s.line = int(pri["line"])

        # Ensure notes cite evidence.
        ev_lines = [
            f"[EVIDENCE {pri['path']}:{pri['range']}]",
        ]
        s = _prepend_evidence(s, ev_lines)

        # Ensure DoD is testable.
        s = _ensure_dod_is_testable(s)

        # Priority sanity: cap vague tasks; add needs_triage tag.
        spec = _specificity_score(s)
        if spec < 2:
            s.priority = min(int(s.priority or 3), 3)
            if "needs_triage" not in (s.tags or ""):
                s.tags = (s.tags + ",needs_triage").strip(",") if s.tags else "needs_triage"

        suggestions.append(s)

    return suggestions


async def generate_tasks_from_findings_llm(
    db: Session,
    snapshot_id: int,
    project: str,
    *,
    max_findings: int = 6,
    chunks_per_finding: int = 3,
) -> dict[str, Any]:
    """
    Main entrypoint used by /debug/repo/tasks_generate.
    Creates real Task rows from LLM suggestions (which are now chunk-grounded).
    """
    suggestions = await _llm_generate_tasks_from_findings_with_retrieval(
        db,
        snapshot_id=snapshot_id,
        project=project,
        max_findings=max_findings,
        chunks_per_finding=chunks_per_finding,
    )

    created = 0
    skipped = 0
    titles: list[str] = []

    for s in suggestions:
        # Deterministic trust anchor: link is computed from snapshot + primary (path,line).
        primary_path = (s.path or "").strip()
        link = _repo_link(snapshot_id, primary_path, s.line) if primary_path else (s.link or "")
        key = s.title.strip().lower() + "|" + link

        # Dedup by title+link
        fp = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        fp_tag = f"taskfp:{fp}"

        existing = (
            db.query(Task)
            .filter(Task.project == project)
            .filter(Task.tags.like(f"%{fp_tag}%"))
            .first()
        )
        if existing:
            skipped += 1
            continue

        # Always ensure DoD stays testable at write time too
        s = _ensure_dod_is_testable(s)

        tags = ",".join([t for t in ["repo", "autogen", s.tags, fp_tag] if t])

        db.add(
            Task(
                title=s.title,
                notes=s.notes,
                priority=max(1, min(5, int(s.priority or 3))),
                estimated_minutes=max(10, int(s.estimated_minutes or 60)),
                blocks_me=bool(s.blocks_me),
                completed=False,
                created_at=datetime.utcnow(),
                project=project,
                tags=tags[:900],
                link=link[:500] if link else None,
                starter=s.starter,
                dod=s.dod,
            )
        )
        created += 1
        titles.append(s.title)

    db.commit()
    return {
        "snapshot_id": snapshot_id,
        "created": created,
        "skipped": skipped,
        "titles": titles,
        "mode": "llm+retrieval",
        "max_findings": max_findings,
        "chunks_per_finding": chunks_per_finding,
    }
