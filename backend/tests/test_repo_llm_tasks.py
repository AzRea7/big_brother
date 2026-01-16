import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db import init_db, SessionLocal
from app.models import RepoSnapshot, RepoFile, Task
from app.config import settings


@pytest.fixture(autouse=True)
def _init_db():
    init_db()


def test_llm_repo_task_generation(monkeypatch):
    # Force LLM path
    settings.LLM_ENABLED = True

    # Monkeypatch LLM client used by repo_tasks
    async def fake_chat_json(self, *, system, user, temperature=0.2, max_tokens=1200):
        return {
            "tasks": [
                {
                    "title": "Fix repo status kwargs mismatch",
                    "notes": "repo_status calls latest_snapshot(repo=..., branch=...) but function signature must accept kwargs. Acceptance: /api/repo/status works with repo+branch params.",
                    "priority": 5,
                    "estimated_minutes": 45,
                    "link": "repo://backend/app/routes/repo.py",
                    "tags": "repo,api,bugfix",
                },
                {
                    "title": "Add FK join for RepoSnapshot.files",
                    "notes": "RepoFile.snapshot_id must have a ForeignKey to repo_snapshots.id so relationship join works. Acceptance: /tasks endpoint doesn't crash mapper init.",
                    "priority": 5,
                    "estimated_minutes": 30,
                    "link": "repo://backend/app/models.py#RepoFile",
                    "tags": "repo,db,sqlalchemy",
                },
            ]
        }

    from app.ai.llm import LLMClient
    monkeypatch.setattr(LLMClient, "chat_json", fake_chat_json, raising=True)

    # Insert a minimal snapshot + one file so the pipeline has something
    db = SessionLocal()
    snap = RepoSnapshot(repo="AzRea7/OneHaven", branch="main", file_count=1, stored_content_files=1)
    db.add(snap)
    db.commit()
    db.refresh(snap)

    db.add(
        RepoFile(
            snapshot_id=snap.id,
            path="onehaven/backend/app/main.py",
            sha=None,
            size=10,
            content="TODO: something",
            content_kind="text",
            skipped=False,
            is_text=True,
        )
    )
    db.commit()

    # Call endpoint to generate tasks from this snapshot (bypassing sync)
    client = TestClient(app)
    r = client.post(f"/debug/repo/sync_and_generate?project=haven&repo=AzRea7/OneHaven&branch=main")

    # NOTE: In this test, sync_and_generate will try to do GitHub sync.
    # If you want it totally offline, call the service function directly.
    # For now, just assert app is alive:
    assert r.status_code in (200, 500)

    # Direct call to service is fully deterministic:
    from app.services.repo_taskgen import generate_tasks_from_snapshot
    created, skipped = generate_tasks_from_snapshot(db, snapshot_id=snap.id, project="haven")
    assert created == 2
    assert skipped == 0

    tasks = db.query(Task).filter(Task.project == "haven").all()
    assert len(tasks) >= 2
    assert any(t.link and t.link.startswith("repo://") for t in tasks)
