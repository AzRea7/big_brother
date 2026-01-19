# backend/app/services/ops_gaps.py
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..models import RepoFile, RepoFinding


@dataclass(frozen=True)
class GapRule:
    key: str
    category: str
    title: str
    severity: int
    pattern: re.Pattern[str]
    recommendation: str


def _fingerprint(snapshot_id: int, path: str, line: int, category: str, title: str) -> str:
    raw = f"{snapshot_id}|{path}|{line}|{category}|{title}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def _insert_gap(
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
            evidence=evidence[:4000],
            recommendation=recommendation[:4000],
            fingerprint=fp,
        )
    )
    return True


def _line_of_match(text: str, start_idx: int) -> int:
    return text.count("\n", 0, start_idx) + 1


def run_ops_gap_scan(db: Session, snapshot_id: int) -> dict:
    """
    Deterministic heuristic checks for common operational foot-guns.

    This is deliberately "cheap but real":
    - finds bad patterns reliably
    - doesn't hallucinate
    - output is explainable
    """
    rules: list[GapRule] = [
        GapRule(
            key="httpx_no_timeout",
            category="ops/reliability",
            title="HTTPX client created without explicit timeout",
            severity=3,
            pattern=re.compile(r"httpx\.(AsyncClient|Client)\(\s*(?![^)]*timeout\s*=)", re.MULTILINE),
            recommendation="Pass timeout= (and ideally limits= and retries) to httpx client construction.",
        ),
        GapRule(
            key="requests_no_timeout",
            category="ops/reliability",
            title="requests.* called without timeout",
            severity=3,
            pattern=re.compile(r"requests\.(get|post|put|delete|patch)\(\s*(?![^)]*timeout\s*=)", re.MULTILINE),
            recommendation="Always set timeout= in requests calls to avoid hanging workers.",
        ),
        GapRule(
            key="print_debug",
            category="ops/quality",
            title="print() used (suggest structured logging)",
            severity=2,
            pattern=re.compile(r"^\s*print\(", re.MULTILINE),
            recommendation="Replace print() with logging (prefer JSON logs + request_id correlation).",
        ),
        GapRule(
            key="no_exception_reporting_hint",
            category="ops/observability",
            title="No obvious exception reporting integration detected",
            severity=2,
            pattern=re.compile(r"(sentry_sdk|opentelemetry|OTEL_EXPORTER|rollbar|datadog)", re.IGNORECASE),
            recommendation="Integrate error reporting (Sentry) or OpenTelemetry exporter for exceptions/traces.",
        ),
    ]

    created = 0

    files = (
        db.query(RepoFile)
        .filter(RepoFile.snapshot_id == snapshot_id)
        .all()
    )

    for rf in files:
        text = rf.content_text or ""
        if not text.strip():
            continue

        # rule 4 is "absence" type: we handle it after scanning all files
        for rule in rules[:3]:
            for m in rule.pattern.finditer(text):
                line_no = _line_of_match(text, m.start())
                evidence = text[m.start() : min(len(text), m.start() + 240)].replace("\n", "\\n")
                if _insert_gap(
                    db,
                    snapshot_id=snapshot_id,
                    path=rf.path,
                    line=line_no,
                    category=rule.category,
                    severity=rule.severity,
                    title=rule.title,
                    evidence=evidence,
                    recommendation=rule.recommendation,
                ):
                    created += 1

    # “absence” check: if none of the repo files contain a known integration marker
    has_obs = False
    obs_rule = rules[3]
    for rf in files:
        if obs_rule.pattern.search(rf.content_text or ""):
            has_obs = True
            break

    if not has_obs:
        # Store once, anchored to "repo root" pseudo-path
        if _insert_gap(
            db,
            snapshot_id=snapshot_id,
            path="(repo)",
            line=1,
            category=obs_rule.category,
            severity=obs_rule.severity,
            title=obs_rule.title,
            evidence="No matches for sentry/opentelemetry/rollbar/datadog markers across snapshot files.",
            recommendation=obs_rule.recommendation,
        ):
            created += 1

    db.commit()
    return {"snapshot_id": snapshot_id, "created": created}
