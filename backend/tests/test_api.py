from __future__ import annotations

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_create_goal_and_task():
    rg = client.post("/goals", json={"title": "Test Goal", "why": "because", "target_date": "2026-02-01"})
    assert rg.status_code == 200
    gid = rg.json()["id"]

    rt = client.post("/tasks", json={"title": "Test Task", "goal_id": gid, "priority": 4, "estimated_minutes": 30})
    assert rt.status_code == 200
    assert rt.json()["goal_id"] == gid
