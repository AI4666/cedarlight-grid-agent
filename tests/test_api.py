"""HTTP-layer tests (skipped automatically if fastapi/httpx are not installed)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from agent import api  # noqa: E402
from agent.inference import MockRiskModel  # noqa: E402
from agent.orchestrator import Agent  # noqa: E402
from agent.service import AgentService  # noqa: E402


def svc():
    return AgentService(Agent(risk_model=MockRiskModel()))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CEDARLIGHT_API_KEY", "key-abc")
    monkeypatch.setenv("CEDARLIGHT_ADMIN_TOKEN", "admin-xyz")
    api._service = svc()
    return TestClient(api.app)


H = {"X-API-Key": "key-abc"}


def test_health_is_open_and_mutations_need_key(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/scenario/normal").status_code == 401
    assert client.post("/scenario/normal", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/scenario/normal", headers=H).status_code == 200


def test_fails_closed_without_configured_key(client, monkeypatch):
    monkeypatch.delenv("CEDARLIGHT_API_KEY")
    assert client.post("/scenario/normal", headers=H).status_code == 503


def test_http_approval_kill_resume_flow(client):
    client.post("/scenario/normal", headers=H)
    aid = client.get("/state").json()["pending_approvals"][0]["action_id"]
    assert client.post(f"/approve/{aid}", headers=H, json={"approver": "ops.maria"}).json()["result"]["ok"]
    assert client.post("/kill", headers=H, json={"actor": "ops"}).json()["kill_switch_active"]
    assert client.post("/resume", headers=H, json={"actor": "a", "admin_token": "nope"}).status_code == 403
    assert client.post("/resume", headers=H, json={"actor": "a", "admin_token": "admin-xyz"}).status_code == 200


def test_http_validation(client):
    assert client.post("/scenario/bogus", headers=H).status_code == 404
    assert client.post("/step", headers=H, json={"percepts": "x"}).status_code == 422
    assert client.post("/step", headers=H, json={"percepts": [{"junk": 1}]}).json()["rejected"]
    assert client.get("/audit?limit=5").status_code == 200
