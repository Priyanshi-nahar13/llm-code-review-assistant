"""tests/test_api.py"""

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    # Patch model loading so tests don't need GPU
    with patch("src.model.inference.get_engine") as mock_engine:
        mock_result = MagicMock()
        mock_result.actionable_findings = []
        mock_result.latency_ms = 100.0
        mock_engine.return_value.analyze_hunks.return_value = [mock_result]

        from src.api.server import app
        return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_empty_diff(client):
    resp = client.post("/analyze", json={"diff": "", "pr_number": 0, "repo": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert "findings" in data
    assert data["total_hunks"] == 0


SAMPLE_DIFF = """\
diff --git a/auth/session.py b/auth/session.py
index abc..def 100644
--- a/auth/session.py
+++ b/auth/session.py
@@ -38,3 +38,2 @@ class SessionManager:
   def validate_token(self, token):
-      query = f\"SELECT * FROM users WHERE token = '{token}'\"
+      query = \"SELECT id FROM users WHERE token = %s\"
"""


def test_analyze_returns_structure(client):
    resp = client.post("/analyze", json={
        "diff": SAMPLE_DIFF,
        "pr_number": 42,
        "repo": "owner/repo",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["pr_number"] == 42
    assert data["repo"] == "owner/repo"
    assert isinstance(data["findings"], list)
    assert "latency_ms" in data


def test_webhook_wrong_event(client):
    resp = client.post(
        "/webhook",
        content=b"{}",
        headers={"X-GitHub-Event": "push"},
    )
    assert resp.status_code == 200
    assert "Ignored" in resp.json()["message"]


def test_webhook_wrong_action(client):
    payload = json.dumps({"action": "closed", "pull_request": {}, "repository": {}}).encode()
    resp = client.post(
        "/webhook",
        content=payload,
        headers={"X-GitHub-Event": "pull_request"},
    )
    assert resp.status_code == 200


def test_findings_not_found(client):
    resp = client.get("/findings/owner/repo/99999")
    assert resp.status_code == 404
