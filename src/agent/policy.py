"""Action policy: tiers, allow-list, argument validation, entailment preconditions, hard constraints, rate limits,
idempotency, human approval and the administrative KILL SWITCH. The policy is independent of any model or text:
nothing a percept says can change what it allows."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from .actions import ApprovalStore, ToolRegistry
from .audit import AuditLog
from .config import admin_token_ok, resolve
from .schemas import ActionProposal, ActionTier, Entailment, PolicyDecision

REQUIRES_ENTAILED = {"propose_schedule"}      # these tools need every attached precondition to be ENTAILED


class KillSwitch:
    """Fail-safe: anyone can halt the agent; only the admin token can resume it (and a file flag can force a halt)."""

    def __init__(self, cfg: dict, audit: AuditLog, approvals: ApprovalStore):
        self._flag, self.audit, self.approvals = False, audit, approvals
        self.file = Path(resolve(cfg, cfg["kill_switch"]["file_flag"]))

    @property
    def active(self) -> bool:
        return self._flag or self.file.exists()

    def activate(self, actor: str, reason: str) -> None:
        self._flag = True
        cancelled = self.approvals.clear()
        self.audit.emit("kill_switch", actor, f"KILL SWITCH ACTIVATED: {reason[:200]}; cancelled {len(cancelled)} pending approvals", cancelled)

    def reset(self, token: str | None, actor: str) -> bool:
        if not admin_token_ok(token):
            self.audit.emit("kill_switch", actor, "reset REFUSED (admin token missing or wrong)")
            return False
        if self.file.exists():
            self.audit.emit("kill_switch", actor, "reset refused: file flag still present")
            return False
        self._flag = False
        self.audit.emit("kill_switch", actor, "kill switch reset by admin")
        return True


class PolicyEngine:
    def __init__(self, cfg: dict, registry: ToolRegistry, approvals: ApprovalStore, kill: KillSwitch, audit: AuditLog,
                 schedule_checker: Callable[[dict, datetime], list[str]] | None = None):
        self.cfg, self.reg, self.approvals, self.kill, self.audit = cfg, registry, approvals, kill, audit
        self.pol = cfg["policy"]
        self.schedule_checker = schedule_checker
        self._exec_times: deque[datetime] = deque()

    def authoritative_tier(self, tool: str) -> ActionTier:
        if tool in self.pol["forbidden_tools"] or tool not in self.pol["allowed_tools"] or tool not in self.pol["tool_tiers"]:
            return ActionTier.FORBIDDEN
        return ActionTier(self.pol["tool_tiers"][tool])

    def record_execution(self, now: datetime) -> None:
        self._exec_times.append(now)

    def evaluate(self, p: ActionProposal, now: datetime) -> PolicyDecision:
        d = PolicyDecision(action_id=p.action_id, tool=p.tool, allowed=False)
        tier = self.authoritative_tier(p.tool)

        def block(reason: str, escalate: bool = True) -> PolicyDecision:
            d.blocked, d.allowed, d.escalate = True, False, escalate
            d.reasons.append(reason)
            self.audit.emit("policy", "policy", f"BLOCKED {p.tool} ({p.action_id}): {reason}", [p.action_id])
            return d

        if self.kill.active:
            return block("kill switch is active", escalate=False)
        if tier == ActionTier.FORBIDDEN:
            return block("tool is forbidden or not on the allow-list (tier 3)")
        if p.tier.value < tier.value:
            return block(f"proposal claims tier {p.tier.value} but policy tier is {tier.value}")
        if (dup := self.reg.seen(p.idempotency_key)) is not None:
            d.duplicate = True
            d.reasons.append("duplicate idempotency key; original result reused")
            return d
        ok, err = self.reg.validate_args(p.tool, p.arguments)
        if not ok:
            return block(err)
        limit = self.pol["rate_limit_per_minute"]
        while self._exec_times and self._exec_times[0] < now - timedelta(minutes=1):
            self._exec_times.popleft()
        if len(self._exec_times) >= limit:
            return block(f"rate limit {limit}/min reached", escalate=False)

        if p.tool in REQUIRES_ENTAILED:
            bad = [e for e in p.entailment if e.status != Entailment.ENTAILED]
            if not p.entailment:
                return block("no entailment evidence attached")
            if bad:
                return block("preconditions not entailed: " + "; ".join(f"{e.query}={e.status.value}" for e in bad[:4]))
            if self.schedule_checker:
                violations = self.schedule_checker(p.arguments, now)
                if violations:
                    return block("hard constraint violated: " + "; ".join(violations[:3]))

        if tier == ActionTier.APPROVAL_REQUIRED and not self.approvals.is_approved(p.action_id):
            d.requires_approval = True
            d.reasons.append("tier 2: human approval required before execution")
            self.audit.emit("policy", "policy", f"APPROVAL REQUIRED for {p.tool} ({p.action_id})", [p.action_id])
            return d

        d.allowed = True
        d.reasons.append(f"allowed at tier {tier.value}")
        self.audit.emit("policy", "policy", f"ALLOWED {p.tool} ({p.action_id}) tier {tier.value}", [p.action_id])
        return d
