# backend/tests/test_repo_tasks_e2e.py
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # deterministic in CI
    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.setenv("GITHUB_REPO", "AzRea7/OneHaven")
    monkeypatch.setenv("GITHUB_BRANCH", "main")
    yield


def test_repo_sync_and_taskgen_smoke():
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200

    r = client.post("/debug/repo/sync_and_generate?project=haven")
    assert r.status_code == 200
    data = r.json()
    assert "snapshot_id" in data
    assert "created_tasks" in data

    r = client.get("/tasks?project=haven")
    assert r.status_code == 200
    tasks = r.json()
    assert isinstance(tasks, list)
    assert len(tasks) >= 1
    assert any("repo" in (t.get("tags") or "") for t in tasks)
