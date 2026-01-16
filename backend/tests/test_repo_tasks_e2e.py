# backend/tests/test_repo_tasks_e2e.py
import os
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # Force deterministic test mode: LLM off unless explicitly set in CI secrets
    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.setenv("GITHUB_DEFAULT_REPO", "AzRea7/OneHaven")
    monkeypatch.setenv("GITHUB_DEFAULT_BRANCH", "main")
    yield


def test_repo_sync_and_seed_taskgen_smoke():
    client = TestClient(app)

    # 1) health
    r = client.get("/health")
    assert r.status_code == 200

    # 2) sync+generate (LLM disabled -> seed tasks)
    r = client.post("/debug/repo/sync_and_generate?project=haven")
    assert r.status_code == 200
    data = r.json()
    assert "snapshot_id" in data

    # 3) tasks exist
    r = client.get("/tasks?project=haven")
    assert r.status_code == 200
    tasks = r.json()
    assert isinstance(tasks, list)
    assert len(tasks) >= 1
    assert any("repo" in (t.get("tags") or "") for t in tasks)

    # 4) complete one if endpoint exists
    tid = tasks[0]["id"]
    r = client.post(f"/tasks/{tid}/complete")
    assert r.status_code in (200, 204)
