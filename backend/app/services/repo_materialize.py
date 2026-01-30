# backend/app/services/repo_materialize.py
from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from ..models import RepoFile


def materialize_snapshot_to_disk(db: Session, snapshot_id: int, dest_root: str | None = None) -> str:
    """
    Write RepoFile content from a snapshot into a folder so static tools can run.

    Returns the folder path.
    """
    if not dest_root:
        dest_root = f"./data/snapshots/{snapshot_id}"

    root = Path(dest_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    files = (
        db.query(RepoFile)
        .filter(RepoFile.snapshot_id == snapshot_id)
        .all()
    )

    for rf in files:
        if not rf.content_text:
            continue
        rel = rf.path.lstrip("/").replace("\\", "/")
        out_path = root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Best-effort: keep text files only
        out_path.write_text(rf.content_text, encoding="utf-8", errors="ignore")

    return str(root)
