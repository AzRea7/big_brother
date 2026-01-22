# backend/app/services/repo_findings.py
from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RepoFinding, RepoFile, RepoSnapshot, Task
from ..services.repo_chunks import search_chunks
from ..ai.llm import LLMClient
from ..ai.repo_tasks import generate_repo_tasks_json


# -----------------------
# JSON parsing utilities
# -----------------------
def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*```$", "", s).strip()
    return s


def _balanced_json_span(text: str, start_idx: int) -> tuple[int, int] | None:
    if start_idx < 0 or start_idx >= len(text):
        return None
    opener = text[start_idx]
    if opener not in "{[":
        return None

    stack = [opener]
    i = start_idx + 1
    in_str = False
    esc = False

    while i < len(text):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            i += 1
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            top = stack[-1]
            expected = "}" if top == "{" else "]"
            if ch != expected:
                return None
            stack.pop()
            if not stack:
                return (start_idx, i)
        i += 1

    return None


def _repair_truncated_json(s: str, start_idx: int) -> str:
    prefix = s[:start_idx]
    body = s[start_idx:]

    if not body or body[0] not in "{[":
        return s

    stack: list[str] = [body[0]]
    in_str = False
    esc = False
    i = 1

    while i < len(body):
        ch = body[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            i += 1
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                top = stack[-1]
                expected = "}" if top == "{" else "]"
                if ch == expected:
                    stack.pop()
                    if not stack:
                        return prefix + body
        i += 1

    repaired = body
    if in_str:
        if esc:
            repaired += "\\\\"
        repaired += '"'

    while stack:
        top = stack.pop()
        repaired += "}" if top == "{" else "]"

    return prefix + repaired


def _json_loads_best_effort(raw: str) -> Any:
    s = _strip_code_fences(str(raw or ""))

    if s.startswith("{") or s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            repaired = _repair_truncated_json(s, 0)
            return json.loads(repaired)

    first_obj = s.find("{")
    first_arr = s.find("[")
    starts = [i for i in (first_obj, first_arr) if i != -1]
    if not starts:
        raise ValueError(f"No JSON found. First 400 chars:\n{s[:400]}")

    start = min(starts)

    span = _balanced_json_span(s, start)
    if span:
        cand = s[span[0] : span[1] + 1]
        return json.loads(cand)

    repaired = _repair_truncated_json(s, start)
    return json.loads(repaired[start:])


# -----------------------
# Repo Findings: LLM scan
# -----------------------
_ALLOWED_CATEGORIES = {"security", "reliability", "db", "api", "tests", "ops", "perf", "style"}


def _coerce_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _coerce_severity(v: Any) -> int:
    sev = _coerce_int(v, 3)
    return max(1, min(5, sev))


def _normalize_acceptance(s: Any) -> str:
    raw = ("" if s is None else str(s)).strip()
    low = raw.lower()
    if not raw or low in ("false", "true", "null", "none"):
        return "Fix implemented and verified with a concrete command (pytest or curl reproduction)."
    if len(raw) < 12:
        return "Fix implemented and verified with a concrete command (pytest or curl reproduction)."
    return raw[:280]


def _normalize_category(s: Any) -> str:
    raw = ("" if s is None else str(s)).strip().lower()
    if raw in _ALLOWED_CATEGORIES:
        return raw
    # map common junk into allowed buckets
    if "auth" in raw or "secret" in raw or "security" in raw:
        return "security"
    if "timeout" in raw or "retry" in raw or "error" in raw or "exception" in raw:
        return "reliability"
    if "db" in raw or "migration" in raw or "sql" in raw:
        return "db"
    if "test" in raw:
        return "tests"
    if "perf" in raw:
        return "perf"
    if "ops" in raw or "docker" in raw or "ci" in raw:
        return "ops"
    if "style" in raw or "lint" in raw or "format" in raw:
        return "style"
    return "api"


def _fingerprint_finding(snapshot_id: int, path: str, line: int | None, title: str) -> str:
    base = f"{snapshot_id}|{path}|{line or 0}|{title}".encode("utf-8", errors="replace")
    return hashlib.sha1(base).hexdigest()[:16]


def _pick_files_for_scan(files: list[RepoFile]) -> list[RepoFile]:
    def score(p: str) -> int:
        p = (p or "").lower()
        s = 0
        if "backend" in p or "/app/" in p:
            s += 30
        if "routes" in p or "main.py" in p:
            s += 25
        if "db" in p or "models" in p or "migrations" in p:
            s += 22
        if "auth" in p or "jwt" in p or "api_key" in p or "security" in p:
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


async def scan_repo_findings_llm(db: Session, snapshot_id: int, max_files: int = 14) -> dict[str, Any]:
    """
    Public API expected by routes/debug.py
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
        return {
            "snapshot_id": snapshot_id,
            "created": 0,
            "skipped": 0,
            "scanned_files": len(payload_files),
            "detail": "LLM not enabled (set LLM_ENABLED=true, OPENAI_BASE_URL, OPENAI_MODEL)",
        }

    # Lock down to high-signal categories first; style only if nothing else exists.
    system = (
        "You are a strict production-readiness reviewer.\n"
        "Return ONLY valid JSON (no markdown, no backticks) with schema:\n"
        '{"findings":[{"path":"...","line":123,"category":"security|reliability|db|api|tests|ops|perf|style","severity":1,'
        '"title":"...","evidence":"...","recommendation":"...","acceptance":"..."}]}\n'
        "Rules:\n"
        "- Keep findings <= 8\n"
        "- Prefer security/auth/secrets, validation, retries/timeouts, DB integrity, error handling, tests\n"
        "- Style-only findings allowed ONLY if no other issues exist\n"
        "- Keep each string <= 240 chars to avoid truncation\n"
        "- line must be int or null\n"
    )
    user = {"snapshot_id": snapshot_id, "repo": snap.repo, "branch": snap.branch, "files": payload_files}

    raw_text = llm.chat(system=system, user=json.dumps(user), temperature=0.1, max_tokens=1200)
    if inspect.isawaitable(raw_text):
        raw_text = await raw_text
    raw_str = str(raw_text)

    findings: list[dict[str, Any]] = []
    parse_error: str | None = None

    try:
        data = _json_loads_best_effort(raw_str)
        if isinstance(data, dict) and isinstance(data.get("findings"), list):
            findings = [x for x in data["findings"] if isinstance(x, dict)]
        else:
            findings = []
    except Exception as e:
        parse_error = f"{type(e).__name__}: {e}"
        findings = []

    inserted = 0
    skipped = 0

    # If the model returns nothing, don't create junk.
    if not findings:
        return {
            "snapshot_id": snapshot_id,
            "created": 0,
            "skipped": 0,
            "scanned_files": len(payload_files),
            "parse_error": parse_error,
            "detail": "No findings produced.",
        }

    # Normalize + de-dup.
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for f in findings:
        path = str(f.get("path") or "").strip()[:600]
        if not path:
            continue
        line = f.get("line")
        line_i: int | None = int(line) if isinstance(line, int) else None
        title = str(f.get("title") or "").strip()[:240]
        if not title:
            continue

        cat = _normalize_category(f.get("category"))
        sev = _coerce_severity(f.get("severity"))
        # style cannot be high severity
        if cat == "style":
            sev = min(sev, 2)

        evidence = str(f.get("evidence") or "").strip()[:280]
        reco = str(f.get("recommendation") or "").strip()[:280]
        acc = _normalize_acceptance(f.get("acceptance"))

        key = f"{path}|{line_i or 0}|{title}"
        if key in seen:
            continue
        seen.add(key)

        out.append(
            {
                "path": path,
                "line": line_i,
                "category": cat,
                "severity": sev,
                "title": title,
                "evidence": evidence,
                "recommendation": reco,
                "acceptance": acc,
            }
        )

    # Insert findings into DB (dedupe by fingerprint).
    for f in out[:50]:
        fp = _fingerprint_finding(snapshot_id, f["path"], f["line"], f["title"])
        existing = (
            db.query(RepoFinding)
            .filter(RepoFinding.snapshot_id == snapshot_id)
            .filter(RepoFinding.fingerprint == fp)
            .first()
        )
        if existing:
            skipped += 1
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
    return {
        "snapshot_id": snapshot_id,
        "inserted": inserted,
        "skipped": skipped,
        "total_findings": int(total_findings),
        "parse_error": parse_error,
    }


# --------------------------------------------------------------------
# Public API expected by older routes (compat)
# --------------------------------------------------------------------
async def run_llm_scan(db: Session, snapshot_id: int) -> dict[str, Any]:
    return await scan_repo_findings_llm(db, snapshot_id)


async def run_llm_repo_scan(db: Session, snapshot_id: int) -> dict[str, Any]:
    return await run_llm_scan(db, snapshot_id)


def list_findings(db: Session, snapshot_id: int, limit: int = 50, offset: int = 0) -> list[RepoFinding]:
    q = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .order_by(RepoFinding.id.desc())
    )
    return q.offset(offset).limit(limit).all()


# -----------------------
# Findings -> Tasks (trust-anchored)
# -----------------------
@dataclass(frozen=True)
class EvidenceChunk:
    path: str
    start_line: int
    end_line: int
    excerpt: str
    score: float | None = None


def _repo_link(snapshot_id: int, path: str, line: int | None) -> str:
    # Deterministic internal link format; can later be rendered into real GitHub links if you want.
    if line is None:
        return f"repo://snapshot/{snapshot_id}#{path}"
    return f"repo://snapshot/{snapshot_id}#{path}:L{line}"


def _extract_signal_tags_from_text(text: str) -> set[str]:
    t = (text or "").lower()
    out: set[str] = set()
    for k in ("auth", "secret", "secrets", "timeout", "retry", "validation", "cors", "jwt", "api key"):
        if k in t:
            if k == "secrets":
                out.add("signal:secrets")
            elif k == "secret":
                out.add("signal:secrets")
            elif k == "api key":
                out.add("signal:auth")
            else:
                out.add(f"signal:{k}")
    return out


def _specificity_score(task: dict[str, Any]) -> int:
    s = 0
    path = str(task.get("path") or "")
    if path and path != "unknown":
        s += 1
    if isinstance(task.get("line"), int):
        s += 1
    notes = str(task.get("notes") or "")
    if "[EVIDENCE " in notes:
        s += 1
    dod = str(task.get("dod") or "").lower()
    if "pytest" in dod or "curl" in dod:
        s += 1
    return s


def _priority_floor_from_tags(tags: str) -> int:
    t = (tags or "").lower()
    if "signal:secrets" in t:
        return 4
    if "signal:auth" in t:
        return 4
    if "signal:validation" in t:
        return 3
    if "signal:timeout" in t or "signal:retry" in t:
        return 3
    return 1


def _merge_tags(*parts: str) -> str:
    items: list[str] = []
    for p in parts:
        for tok in (p or "").split(","):
            tok = tok.strip()
            if tok:
                items.append(tok)
    # stable-ish dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for tok in items:
        low = tok.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(tok)
    return ",".join(out)[:300]


def tasks_from_findings(db: Session, snapshot_id: int, project: str, limit: int = 12) -> dict[str, Any]:
    """
    Deterministic fallback: turn stored findings into tasks without the LLM.
    Still stamps link so you can click into code.
    """
    findings = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .order_by(RepoFinding.id.desc())
        .all()
    )
    findings.sort(key=lambda r: _coerce_severity(getattr(r, "severity", None)), reverse=True)
    findings = findings[:limit]

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
        pri = max(1, min(5, sev))
        tags = _merge_tags("repo,autogen", fp_tag, f"cat:{f.category}", _extract_signal_tags_from_text(f"{f.title} {f.evidence}"))

        link = _repo_link(snapshot_id, f.path, f.line)

        notes = (
            f"{f.title}\n"
            f"[EVIDENCE {f.path}:L{f.line or 1}-{(f.line or 1) + 6}] {str(f.evidence or '')[:240]}\n"
            f"Recommendation: {str(f.recommendation or '')[:240]}"
        )[:4000]

        dod = (str(getattr(f, "acceptance", "") or "").strip() or "Fix verified with pytest or curl.")[:1000]
        if "pytest" not in dod.lower() and "curl" not in dod.lower():
            dod = (dod + " Verify with: pytest -q (or a curl reproduction).")[:1000]

        t = Task(
            goal_id=None,
            title=str(f.title or "Repo finding").strip()[:240],
            notes=notes,
            due_date=None,
            priority=pri,
            estimated_minutes=90,
            blocks_me=True if pri >= 4 else False,
            completed=False,
            completed_at=None,
            created_at=datetime.utcnow(),
            project=project,
            tags=tags,
            link=link,
            starter="Open the referenced file at the linked line and reproduce the issue (5 min).",
            dod=dod,
            impact_score=None,
            confidence=None,
            energy=None,
            parent_task_id=None,
        )
        db.add(t)
        created += 1

    db.commit()
    return {"snapshot_id": snapshot_id, "created": created, "skipped": skipped}


async def generate_tasks_from_findings_llm(
    *,
    db: Session,
    snapshot_id: int,
    project: str,
    max_findings: int = 10,
    chunks_per_finding: int = 3,
) -> dict[str, Any]:
    """
    LLM path: build per-finding retrieval packs, force primary chunk, and stamp link deterministically.
    """
    snap = db.get(RepoSnapshot, snapshot_id)
    if not snap:
        return {"snapshot_id": snapshot_id, "created": 0, "skipped": 0, "detail": "snapshot not found"}

    findings = (
        db.query(RepoFinding)
        .filter(RepoFinding.snapshot_id == snapshot_id)
        .order_by(RepoFinding.id.desc())
        .limit(max_findings)
        .all()
    )
    if not findings:
        return {"snapshot_id": snapshot_id, "created": 0, "skipped": 0, "detail": "no findings"}

    # Build retrieval packs: 1 same-file query + 1 semantic-ish query
    packs: list[dict[str, Any]] = []
    primary_by_fp: dict[str, EvidenceChunk] = {}

    for f in findings:
        fp = f.fingerprint
        title = str(f.title or "").strip()
        cat = str(f.category or "").strip()
        path_hint = str(f.path or "").strip()

        # Query A: exact-ish
        q_exact = " ".join([path_hint, title]).strip()
        # Query B: semantic-ish
        q_sem = " ".join([cat, str(f.recommendation or ""), str(f.evidence or "")]).strip()
        q_sem = re.sub(r"\s+", " ", q_sem)[:500]

        hits: list[EvidenceChunk] = []

        # same-file bias
        mode_used_a, hit_rows_a = await search_chunks(
            db=db,
            snapshot_id=snapshot_id,
            query=q_exact if len(q_exact) >= 2 else title,
            top_k=max(3, chunks_per_finding),
            mode="auto",
            path_contains=path_hint if path_hint else None,
        )
        for h in hit_rows_a or []:
            hits.append(
                EvidenceChunk(
                    path=str(getattr(h, "path", "")),
                    start_line=int(getattr(h, "start_line", 1)),
                    end_line=int(getattr(h, "end_line", 1)),
                    excerpt=str(getattr(h, "chunk_text", ""))[:900],
                    score=getattr(h, "score", None),
                )
            )

        # broader recall
        mode_used_b, hit_rows_b = await search_chunks(
            db=db,
            snapshot_id=snapshot_id,
            query=q_sem if len(q_sem) >= 2 else title,
            top_k=max(3, chunks_per_finding),
            mode="auto",
            path_contains=None,
        )
        for h in hit_rows_b or []:
            hits.append(
                EvidenceChunk(
                    path=str(getattr(h, "path", "")),
                    start_line=int(getattr(h, "start_line", 1)),
                    end_line=int(getattr(h, "end_line", 1)),
                    excerpt=str(getattr(h, "chunk_text", ""))[:900],
                    score=getattr(h, "score", None),
                )
            )

        # Deduplicate by (path,start,end)
        seen = set()
        deduped: list[EvidenceChunk] = []
        for c in hits:
            key = (c.path, c.start_line, c.end_line)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        # Choose primary = same-file best if present, else first
        primary: EvidenceChunk | None = None
        for c in deduped:
            if path_hint and c.path == path_hint:
                primary = c
                break
        if primary is None and deduped:
            primary = deduped[0]
        if primary:
            primary_by_fp[fp] = primary

        # Build evidence file summary block the task LLM will see
        chunks_block: list[dict[str, Any]] = []
        for c in deduped[: max(1, chunks_per_finding)]:
            chunks_block.append(
                {
                    "path": c.path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "excerpt": c.excerpt[:700],
                    "score": c.score,
                }
            )

        packs.append(
            {
                "finding_fingerprint": fp,
                "finding": {
                    "path": f.path,
                    "line": f.line,
                    "category": f.category,
                    "severity": _coerce_severity(f.severity),
                    "title": f.title,
                    "evidence": f.evidence,
                    "recommendation": f.recommendation,
                    "acceptance": getattr(f, "acceptance", None),
                },
                "retrieval": {
                    "mode_a": str(mode_used_a),
                    "mode_b": str(mode_used_b),
                    "primary": {
                        "path": primary.path if primary else f.path,
                        "start_line": primary.start_line if primary else (f.line or 1),
                        "end_line": primary.end_line if primary else ((f.line or 1) + 5),
                    },
                    "chunks": chunks_block,
                },
            }
        )

    # Feed packs into repo_tasks prompt as extra_evidence; keep file_summaries minimal
    # (We use packs for grounding; file_summaries can be empty without breaking anything.)
    task_json = await generate_repo_tasks_json(
        repo_name=snap.repo,
        branch=snap.branch,
        commit_sha=snap.commit_sha,
        snapshot_id=snapshot_id,
        signal_counts={},
        file_summaries=[],
        extra_evidence=packs,
    )

    tasks = (task_json or {}).get("tasks") or []
    if not isinstance(tasks, list):
        tasks = []

    created = 0
    skipped = 0

    for t in tasks:
        if not isinstance(t, dict):
            continue

        # Determine which finding it belongs to (best-effort: fingerprint tag)
        tags_in = str(t.get("tags") or "")
        fp_match = re.search(r"\bfinding:([0-9a-f]{8,40})\b", tags_in, flags=re.IGNORECASE)
        fp = fp_match.group(1) if fp_match else None

        # Enforce primary chunk path/line if missing/unknown
        path = str(t.get("path") or "").strip()
        line = t.get("line")
        line_i: int | None = int(line) if isinstance(line, int) else None

        primary = primary_by_fp.get(fp or "", None)
        if (not path or path == "unknown") and primary:
            path = primary.path
        if line_i is None and primary:
            line_i = primary.start_line

        if not path:
            path = "unknown"

        # Stamp deterministic link (never rely on the model for formatting)
        link = _repo_link(snapshot_id, path, line_i)

        # Ensure evidence citation exists in notes; if missing, inject from primary
        notes = str(t.get("notes") or "").strip()[:4000]
        if "[EVIDENCE " not in notes and primary:
            injected = f"[EVIDENCE {primary.path}:L{primary.start_line}-{primary.end_line}] {primary.excerpt[:220]}"
            notes = (notes + ("\n" if notes else "") + injected)[:4000]

        # Ensure tags include repo/autogen and finding fingerprint tag if we have it
        base_tags = "repo,autogen"
        fp_tag = f"finding:{fp}" if fp else ""
        tags = _merge_tags(base_tags, tags_in, fp_tag)

        # Compute priority: floor by signal tags + clamp based on specificity score
        try:
            pr = int(t.get("priority", 3))
        except Exception:
            pr = 3
        pr = max(1, min(5, pr))

        floor = _priority_floor_from_tags(tags)
        pr = max(pr, floor)

        spec = _specificity_score({"path": path, "line": line_i, "notes": notes, "dod": t.get("dod")})
        if spec < 2:
            pr = min(pr, 3)
            tags = _merge_tags(tags, "needs_triage")

        # Enforce DoD must be runnable (pytest/curl)
        dod = str(t.get("dod") or "").strip()[:1000]
        if "pytest" not in dod.lower() and "curl" not in dod.lower():
            dod = (dod + " Verify with: pytest -q (or a curl reproduction).").strip()[:1000]

        # De-dupe tasks by (project + title + link)
        existing = (
            db.query(Task)
            .filter(Task.project == project)
            .filter(Task.title == str(t.get("title") or "").strip()[:240])
            .filter(Task.link == link)
            .first()
        )
        if existing:
            skipped += 1
            continue

        try:
            est = int(t.get("estimated_minutes", 60))
        except Exception:
            est = 60
        est = max(15, min(240, est))

        blocks_me = bool(t.get("blocks_me", False))
        if pr >= 4:
            blocks_me = True

        row = Task(
            goal_id=None,
            title=str(t.get("title") or "Repo task").strip()[:240],
            notes=notes,
            due_date=None,
            priority=pr,
            estimated_minutes=est,
            blocks_me=blocks_me,
            completed=False,
            completed_at=None,
            created_at=datetime.utcnow(),
            project=project,
            tags=tags,
            link=link,
            starter=str(t.get("starter") or "").strip()[:1000] or "Open the linked code location and reproduce (5 min).",
            dod=dod,
            impact_score=None,
            confidence=None,
            energy=None,
            parent_task_id=None,
        )
        db.add(row)
        created += 1

    db.commit()
    return {"snapshot_id": snapshot_id, "created": created, "skipped": skipped, "tasks_in": len(tasks)}
