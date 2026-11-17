"""Service, scenario and API-layer tests. API tests are skipped if fastapi/httpx are not installed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent.scenarios import SCENARIOS, build  # noqa: E402
from agent.service import AgentService  # noqa: E402
from agent.inference import MockRiskModel  # noqa: E402
from agent.orchestrator import Agent  # noqa: E402
from datetime import datetime, timezone  # noqa: E402


def svc():
    return AgentService(Agent(risk_model=MockRiskModel()))


def test_all_six_scenarios_run_and_never_execute_forbidden_tools():
    assert len(SCENARIOS) == 6
    for name in SCENARIOS:
        s = svc()
        out = s.run_scenario(name)
        assert out["status"] == "completed", name
        assert not {r["tool"] for r in out["results"]} & {"switch_breaker", "shed_load", "dispatch_crew"}


def test_unknown_scenario_raises():
    with pytest.raises(KeyError):
        build("nope", datetime.now(timezone.utc))


def test_service_approval_flow_and_state():
    s = svc()
    out = s.run_scenario("normal")
    pending = s.state()["pending_approvals"]
    assert len(pending) == 1 and s.state()["kill_switch_active"] is False
    r = s.approve(pending[0]["action_id"], "ops.maria")
    assert r["result"]["ok"] and s.state()["pending_approvals"] == []
    assert any(e["stage"] == "action" for e in s.audit(stage="action"))


def test_service_kill_and_resume(monkeypatch):
    s = svc()
    s.run_scenario("normal")
    assert s.kill("ops", "drill")["kill_switch_active"] and s.state()["pending_approvals"] == []
    assert s.run_scenario("normal")["status"] == "halted_kill_switch"
    monkeypatch.setenv("CEDARLIGHT_ADMIN_TOKEN", "tok-123456")
    assert not s.resume("bad", "x")["resumed"] and s.resume("tok-123456", "admin")["resumed"]


def test_service_handles_garbage_input():
    out = svc().step("not a list")
    assert out["rejected"]
