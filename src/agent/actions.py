"""STAGE 6 (execution side) - ACTION. Only allow-listed tools exist. Forbidden tools (switch_breaker, shed_load,
dispatch_crew) are deliberately NOT implemented anywhere, so even a bug in the policy layer cannot execute them
(least privilege / defence in depth). Every call is argument-validated, time-limited, idempotent and logged."""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .schemas import ActionProposal, ApprovalState, ToolResult


class _A(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryKB(_A):
    predicate: str = Field(pattern=r"^[A-Za-z_]\w{0,40}$")
    args: list[str] = Field(default_factory=list, max_length=4)


class WriteForecast(_A):
    feeder: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,32}$")
    horizon_hours: int = Field(ge=1, le=168)
    points: list[dict[str, Any]] = Field(max_length=200)
    n_obs: int = 0
    n_missing_hours: int = 0


class RankInspections(_A):
    ranking: list[dict[str, Any]] = Field(max_length=100)
    route: list[str] = Field(default_factory=list, max_length=100)


class WriteReport(_A):
    summary: str = Field(max_length=1000)
    refs: list[str] = Field(default_factory=list, max_length=50)


class ProposeSchedule(_A):
    jobs: list[dict[str, Any]] = Field(max_length=50)
    assignment: dict[str, dict[str, Any]] = Field(max_length=50)


class RequestApproval(_A):
    action_id: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,40}$")
    summary: str = Field(max_length=500)


class Escalate(_A):
    reason: str = Field(max_length=500)
    refs: list[str] = Field(default_factory=list, max_length=50)


ARG_MODELS = {"query_kb": QueryKB, "write_forecast": WriteForecast, "rank_inspections": RankInspections,
              "write_report": WriteReport, "propose_schedule": ProposeSchedule, "request_approval": RequestApproval,
              "escalate_to_human": Escalate}


def idempotency_key(tool: str, args: dict[str, Any]) -> str:
    return hashlib.sha256((tool + "|" + json.dumps(args, sort_keys=True, default=str)).encode()).hexdigest()[:16]


class ApprovalStore:
    """Human approvals. The agent can only PROPOSE tier-2 actions; only this store can mark them approved."""

    def __init__(self):
        self.pending: dict[str, ActionProposal] = {}
        self.approved: dict[str, str] = {}      # action_id -> approver
        self.rejected: dict[str, str] = {}

    def add(self, p: ActionProposal) -> None:
        self.pending[p.action_id] = p

    def approve(self, action_id: str, approver: str) -> ActionProposal | None:
        p = self.pending.pop(action_id, None)
        if p is not None:
            self.approved[action_id] = approver
        return p

    def reject(self, action_id: str, approver: str) -> bool:
        if self.pending.pop(action_id, None) is None:
            return False
        self.rejected[action_id] = approver
        return True

    def is_approved(self, action_id: str) -> bool:
        return action_id in self.approved

    def clear(self) -> list[str]:
        ids = list(self.pending)
        self.pending.clear()
        return ids


class ToolRegistry:
    def __init__(self, cfg: dict, kb, outbox: Path | None = None):
        self.cfg, self.kb, self.outbox = cfg, kb, outbox
        self.timeout = cfg["policy"]["tool_timeout_seconds"]
        self.store: dict[str, Any] = {"forecasts": {}, "rankings": [], "reports": [], "schedules": [], "approval_requests": [], "escalations": []}
        self.results: dict[str, ToolResult] = {}          # idempotency key -> first result
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._impl = {"query_kb": self._query_kb, "write_forecast": self._write_forecast, "rank_inspections": self._rank,
                      "write_report": self._report, "propose_schedule": self._schedule, "request_approval": self._request_approval,
                      "escalate_to_human": self._escalate}
        self._now: datetime | None = None

    # ---- introspection ----
    def implemented(self) -> set[str]:
        return set(self._impl)

    def seen(self, key: str) -> ToolResult | None:
        return self.results.get(key)

    def validate_args(self, tool: str, args: dict[str, Any]) -> tuple[bool, str]:
        model = ARG_MODELS.get(tool)
        if model is None:
            return False, "no argument schema (tool not registered)"
        try:
            model.model_validate(args)
            return True, ""
        except ValidationError as e:
            return False, f"invalid arguments: {e.errors()[0]['loc']} {e.errors()[0]['type']}"

    # ---- execution ----
    def execute(self, p: ActionProposal, now: datetime) -> ToolResult:
        self._now = now
        if p.idempotency_key in self.results:
            return self.results[p.idempotency_key]
        t0 = time.perf_counter()
        fn = self._impl.get(p.tool)
        if fn is None:
            res = ToolResult(action_id=p.action_id, tool=p.tool, ok=False, error="tool not registered (least privilege)")
        else:
            try:
                out = self._pool.submit(fn, p).result(timeout=self.timeout)
                res = ToolResult(action_id=p.action_id, tool=p.tool, ok=True, output=out,
                                 duration_ms=(time.perf_counter() - t0) * 1000)
            except FutTimeout:
                res = ToolResult(action_id=p.action_id, tool=p.tool, ok=False, error="timeout")
            except Exception as e:  # noqa: BLE001 - tool failures are reported, never raised into the loop
                res = ToolResult(action_id=p.action_id, tool=p.tool, ok=False, error=f"{type(e).__name__}: {str(e)[:150]}")
        self.results[p.idempotency_key] = res
        return res

    def _persist(self, p: ActionProposal) -> str | None:
        if not self.outbox:
            return None
        self.outbox.mkdir(parents=True, exist_ok=True)
        name = f"{p.tool}_{p.idempotency_key}.json"      # file name derived from tool + hash only (no user text)
        (self.outbox / name).write_text(json.dumps(p.arguments, indent=1, default=str), encoding="utf-8")
        return name

    # ---- tools (all simulated; no real grid system is touched) ----
    def _query_kb(self, p):
        a = QueryKB.model_validate(p.arguments)
        prefix = a.predicate + "("
        facts = sorted(x for x in self.kb.trusted_atoms(self._now) if x.lstrip("~").startswith(prefix)
                       and all(arg in x for arg in a.args))
        return {"matches": facts[:50], "count": len(facts)}

    def _write_forecast(self, p):
        a = WriteForecast.model_validate(p.arguments)
        self.store["forecasts"][a.feeder] = a.model_dump()
        return {"stored": a.feeder, "file": self._persist(p)}

    def _rank(self, p):
        a = RankInspections.model_validate(p.arguments)
        self.store["rankings"].append(a.model_dump())
        return {"ranked": len(a.ranking), "file": self._persist(p)}

    def _report(self, p):
        a = WriteReport.model_validate(p.arguments)
        self.store["reports"].append(a.model_dump())
        return {"report_chars": len(a.summary), "file": self._persist(p)}

    def _schedule(self, p):
        a = ProposeSchedule.model_validate(p.arguments)
        self.store["schedules"].append(a.model_dump())       # "published" to the (simulated) crew planning board
        return {"published_jobs": len(a.jobs), "file": self._persist(p)}

    def _request_approval(self, p):
        a = RequestApproval.model_validate(p.arguments)
        self.store["approval_requests"].append(a.model_dump())
        return {"requested": a.action_id}

    def _escalate(self, p):
        a = Escalate.model_validate(p.arguments)
        self.store["escalations"].append(a.model_dump())
        return {"escalated": True}
