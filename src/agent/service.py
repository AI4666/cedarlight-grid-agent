"""Framework-free service wrapper around the Agent (thread-safe). The FastAPI app and CLI are thin layers over this."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from .config import config_hash
from .orchestrator import Agent
from .scenarios import SCENARIOS, build


class AgentService:
    def __init__(self, agent: Agent | None = None):
        self.agent = agent or Agent()
        self._lock = threading.Lock()

    def step(self, batch: Any, now: datetime | None = None) -> dict:
        with self._lock:
            res = self.agent.run(batch, now or datetime.now(timezone.utc))
            return res.model_dump(mode="json")

    def run_scenario(self, name: str, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        return self.step(build(name, now), now)

    def scenarios(self) -> dict[str, str]:
        return dict(SCENARIOS)

    def state(self) -> dict:
        a, now = self.agent, datetime.now(timezone.utc)
        with self._lock:
            return {"run_id": a.run_id, "config_hash": config_hash(a.cfg), "kill_switch_active": a.kill.active,
                    "model": {"name": a.model.name, "version": a.model.version, "source": a.model.source},
                    "knowledge": a.kb.snapshot(now),
                    "pending_approvals": [p.model_dump(mode="json") for p in a.approvals.pending.values()],
                    "approved": dict(a.approvals.approved), "rejected": dict(a.approvals.rejected)}

    def audit(self, limit: int = 200, stage: str | None = None) -> list[dict]:
        with self._lock:
            ev = self.agent.audit.by_stage(stage) if stage else self.agent.audit.events
            return [e.model_dump(mode="json") for e in ev[-max(1, min(limit, 1000)):]]

    def approve(self, action_id: str, approver: str) -> dict:
        with self._lock:
            d, r = self.agent.approve(action_id, approver)
            return {"decision": d.model_dump(mode="json"), "result": r.model_dump(mode="json") if r else None}

    def reject(self, action_id: str, approver: str) -> dict:
        with self._lock:
            return {"rejected": self.agent.reject(action_id, approver)}

    def kill(self, actor: str, reason: str) -> dict:
        with self._lock:
            self.agent.kill_switch(actor, reason)
            return {"kill_switch_active": self.agent.kill.active}

    def resume(self, token: str | None, actor: str) -> dict:
        with self._lock:
            ok = self.agent.resume(token, actor)
            return {"resumed": ok, "kill_switch_active": self.agent.kill.active}

    def health(self) -> dict:
        return {"status": "ok", "kill_switch_active": self.agent.kill.active, "model_source": self.agent.model.source}
