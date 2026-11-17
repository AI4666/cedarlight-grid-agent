"""The agent loop:  observe -> update knowledge -> reason -> entailment check -> plan -> approval/policy check
-> act -> observe result.  Bounded, idempotent, fully audited, and interruptible by the kill switch."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import entailment, perceive, proposition
from .actions import ApprovalStore, ToolRegistry, idempotency_key
from .audit import AuditLog
from .config import config_hash, load_config, resolve
from .inference import (Inferences, Reasoner, build_schedule, load_risk_model, plan_route, preferred_schedule,
                        schedule_violations, slot_windows, solve_schedule, AgentSched)
from .knowledge import KnowledgeBase
from .policy import KillSwitch, PolicyEngine
from .schemas import (ActionProposal, ActionTier, ApprovalState, CycleResult, Entailment, EntailmentResult,
                      PolicyDecision, ToolResult, ValidationStatus)


def _parse_args(atom: str) -> tuple[str, ...]:
    from logic_engine import parse
    return parse(atom)[2]


class Agent:
    def __init__(self, cfg: dict | None = None, seed: int | None = None, risk_model=None, audit_path: Path | None = None,
                 outbox: Path | None = None, topology_path: Path | None = None):
        self.cfg = cfg or load_config()
        self.seed = self.cfg["run"]["seed"] if seed is None else seed
        self.run_id = f"run_{uuid.uuid4().hex[:10]}"
        self.audit = AuditLog(self.run_id, config_hash(self.cfg), self.seed, audit_path)
        self.kb = KnowledgeBase(self.cfg, topology_path)
        if risk_model is None:
            risk_model, note = load_risk_model(self.cfg)
        else:
            note = f"injected model {risk_model.name}"
        self.model = risk_model
        self.kb.register_model("failure_classifier", risk_model.version, risk_model.source)
        self.audit.emit("knowledge_base", "agent", f"model registered: {risk_model.name} v{risk_model.version} ({note})")
        self.reasoner = Reasoner(self.kb, self.cfg, risk_model)
        self.approvals = ApprovalStore()
        self.registry = ToolRegistry(self.cfg, self.kb, outbox)
        self.kill = KillSwitch(self.cfg, self.audit, self.approvals)
        self.policy = PolicyEngine(self.cfg, self.registry, self.approvals, self.kill, self.audit, self._check_schedule)
        self.cycle = 0
        self._req_seen = 0
        self._run_keys: set[str] = set()
        self._run_recorded: set[tuple[str, str]] = set()
        self._run_escalated: set[str] = set()
        self.last: CycleResult | None = None

    # ------------------------------------------------------------------ controls
    def kill_switch(self, actor: str, reason: str) -> None:
        self.kill.activate(actor, reason)

    def resume(self, token: str | None, actor: str) -> bool:
        return self.kill.reset(token, actor)

    # ------------------------------------------------------------------ helpers
    def _mk(self, tool: str, args: dict[str, Any], justification: str, entail: list[EntailmentResult] | None = None,
            utility: float | None = None, risk: float = 0.0) -> ActionProposal:
        tier = self.policy.authoritative_tier(tool)
        return ActionProposal(idempotency_key=idempotency_key(tool, args), tool=tool, arguments=args, tier=tier, utility=utility,
                              risk=max(0.0, min(1.0, risk)), entailment=entail or [], justification=justification[:500])

    def _check_schedule(self, args: dict, now: datetime) -> list[str]:
        """Re-validate a schedule's hard constraints against CURRENT knowledge (used at proposal and at approval time)."""
        crews = self.cfg["scheduling"]["crews"]
        cid = {c["id"]: i for i, c in enumerate(crews)}
        jobs = args["jobs"]
        available = {s for s, w in slot_windows(self.kb).items()
                     if entailment.check(f"SafeToSchedule({w})", self.kb, now).status == Entailment.ENTAILED}
        P = AgentSched(jobs, crews, len(self.kb.topology["windows"]), set(range(len(self.kb.topology["windows"]))) - available)
        try:
            assign = {i: (cid[args["assignment"][j["asset"]]["crew"]], int(args["assignment"][j["asset"]]["slot"])) for i, j in enumerate(jobs)}
        except (KeyError, ValueError, TypeError):
            return ["assignment does not match jobs"]
        return schedule_violations(P, assign, jobs)

    def _dispatch(self, p: ActionProposal, now: datetime, res: CycleResult) -> ToolResult | None:
        if p.idempotency_key in self._run_keys:          # same proposal already handled in this run (feedback iterations)
            return None
        self._run_keys.add(p.idempotency_key)
        res.proposals.append(p)
        if any(q.idempotency_key == p.idempotency_key for q in self.approvals.pending.values()):
            d = PolicyDecision(action_id=p.action_id, tool=p.tool, allowed=False, duplicate=True, reasons=["identical proposal already awaiting approval"])
            res.decisions.append(d)
            p.approval = ApprovalState.PENDING
            return None
        d = self.policy.evaluate(p, now)
        res.decisions.append(d)
        self.audit.emit("action", "agent", f"proposal {p.tool} ({p.action_id}) tier {p.tier.value}: {p.justification}"[:500], [p.action_id])
        if d.duplicate:
            return None
        if d.blocked:
            p.approval = ApprovalState.BLOCKED
            if d.escalate:
                self._escalate(f"{p.tool} blocked: {'; '.join(d.reasons)}"[:480], [p.action_id], now, res)
            return None
        if d.requires_approval:
            p.approval = ApprovalState.PENDING
            self.approvals.add(p)
            summary = f"{p.tool}: {p.justification}"[:480]
            self._dispatch(self._mk("request_approval", {"action_id": p.action_id, "summary": summary}, "notify approver"), now, res)
            return None
        result = self.registry.execute(p, now)
        self.policy.record_execution(now)
        res.results.append(result)
        self.audit.emit("action", p.tool, f"{'OK' if result.ok else 'FAILED'} {p.tool} ({p.action_id})" + (f": {result.error}" if result.error else ""), [p.action_id])
        return result

    def _escalate(self, reason: str, refs: list[str], now: datetime, res: CycleResult) -> None:
        if reason in self._run_escalated:
            return
        self._run_escalated.add(reason)
        res.escalations.append(reason)
        p = self._mk("escalate_to_human", {"reason": reason[:500], "refs": refs[:20]}, "human review required")
        d = self.policy.evaluate(p, now)
        res.decisions.append(d)
        res.proposals.append(p)
        if d.allowed:
            r = self.registry.execute(p, now)
            self.policy.record_execution(now)
            res.results.append(r)
            self.audit.emit("action", "agent", f"ESCALATED: {reason}"[:500], refs[:10])

    # ------------------------------------------------------------------ the loop
    def run(self, raw_percepts: Any, now: datetime | None = None) -> CycleResult:
        now = now or datetime.now(timezone.utc)
        self.cycle += 1
        self._run_keys, self._run_recorded, self._run_escalated = set(), set(), set()
        start = len(self.audit.events)
        res = CycleResult(run_id=self.run_id, cycle=self.cycle, iterations=0, status="completed")
        limit = self.cfg["run"]["max_loop_iterations"]
        batch = raw_percepts
        if self.kill.active:
            res.status = "halted_kill_switch"
            self.audit.emit("kill_switch", "agent", "cycle refused: kill switch active")
        else:
            for it in range(limit):
                if self.kill.active:
                    res.status = "halted_kill_switch"
                    break
                res.iterations = it + 1
                new = self._iteration(batch, now, res)
                if not new:
                    res.status = "completed"
                    break
                batch = [{"source": "tool_result", "entity_id": r.tool, "timestamp": now.isoformat(),
                          "values": {"ok": r.ok}, "quality": 1.0} for r in new]     # feed results back as percepts
                res.status = "max_iterations" if it == limit - 1 else "completed"
        res.conflicts = self.kb.conflicts(now)
        res.audit_ids = [e.event_id for e in self.audit.events[start:]]
        self.last = res
        return res

    def _iteration(self, batch: Any, now: datetime, res: CycleResult) -> list[ToolResult]:
        # 1. PERCEPTS
        percepts, rejected = perceive.ingest(batch, self.cfg, now)
        res.percepts += percepts
        res.rejected += rejected
        counts = {s: sum(p.validation_status.value == s for p in percepts) for s in ("valid", "suspect", "rejected")}
        self.audit.emit("percepts", "perceive", f"{len(percepts)} percepts {counts}, {len(rejected)} malformed", [p.percept_id for p in percepts])

        # 2. PROPOSITIONS
        known = set(self.cfg["policy"]["allowed_tools"]) | set(self.cfg["policy"]["forbidden_tools"])
        by_id = {p.percept_id: p for p in percepts}
        props = []
        for p in percepts:
            self.kb.observe(p)
            props += proposition.to_propositions(p, self.cfg, known)
        pairs = proposition.find_contradictions(props)
        res.propositions += props
        self.audit.emit("propositions", "proposition", f"{len(props)} propositions, {len(pairs)} in-batch contradictions", [x.prop_id for x in props])

        # 3. KNOWLEDGE BASE
        updates = []
        for pr in props:
            r = self.kb.tell(pr, by_id[pr.derived_from[0]])
            updates.append(f"{r['status']}: {r['atom']}")
        res.kb_updates += updates
        conflicts = self.kb.conflicts(now)
        self.audit.emit("knowledge_base", "knowledge", f"{len(updates)} updates; {len(conflicts)} conflicts: {[c.atom for c in conflicts][:5]}")

        # 4. INFERENCE
        inf = self.reasoner.infer(now)
        res.traces = inf.traces
        self.audit.emit("inference", "inference", f"{len(inf.traces)} traces; risk for {sorted(inf.risk)}; forecasts for {sorted(inf.forecasts)}; notes {inf.notes[:3]}")

        # 5-6. ENTAILMENT -> PLAN -> POLICY -> ACT
        return self._plan_and_act(percepts, inf, conflicts, now, res)

    # ------------------------------------------------------------------ plan + act
    def _plan_and_act(self, percepts, inf: Inferences, conflicts, now: datetime, res: CycleResult) -> list[ToolResult]:
        kb, cfg = self.kb, self.cfg
        dc = cfg["decision"]
        active = kb.active_dynamic(now)
        mentioned = {a for f in active for a in f.args}
        mentioned |= {a for atom in kb.uncertain_atoms() for a in _parse_args(atom)}     # low-confidence evidence is reported, not hidden
        ent: dict[str, EntailmentResult] = {}

        def ask(q: str, record: bool = True, conf: float | None = None) -> EntailmentResult:
            if q not in ent:
                ent[q] = entailment.check(q, kb, now, conf)
                if record and (q, ent[q].status.value) not in self._run_recorded:
                    self._run_recorded.add((q, ent[q].status.value))
                    res.entailments.append(ent[q])
                    self.audit.emit("entailment", "entailment", f"{q} -> {ent[q].status.value}: {ent[q].rationale}"[:500], ent[q].supporting_ids[:10])
            return ent[q]

        assets = [a for a in kb.topology["transformers"] if a in mentioned]
        for a in assets:
            ask(f"InspectUrgent({a})", conf=inf.risk.get(a, {}).get("p_fail"))
        urgent = sorted((a for a in assets if ent[f"InspectUrgent({a})"].status == Entailment.ENTAILED),
                        key=lambda a: (-inf.risk.get(a, {}).get("p_fail", 0.0), -inf.risk.get(a, {}).get("p_ml", 0.0), a))
        for a in urgent:
            ask(f"ImpactHigh({a})")
        windows = kb.topology["windows"]
        for w in windows:
            ask(f"SafeToSchedule({w})", record=bool(urgent) or w in mentioned)

        new: list[ToolResult] = []

        def go(p: ActionProposal):
            r = self._dispatch(p, now, res)
            if r is not None:
                new.append(r)

        # a) forecasts (tier 0)
        for feeder, fc in sorted(inf.forecasts.items()):
            go(self._mk("write_forecast", {"feeder": feeder, "horizon_hours": fc["horizon_hours"], "points": fc["points"],
                                           "n_obs": fc["n_obs"], "n_missing_hours": fc["n_missing_hours"]},
                        f"24h load forecast for {feeder} from {fc['n_obs']} readings ({fc['n_missing_hours']} missing hours)"))

        # b) inspection ranking (tier 0): logic-entailed urgent assets plus assets the Bayesian decision says to inspect
        bn_act = [a for a, r in inf.risk.items() if r["decision"]["status"] == "act" and r["decision"]["action"] == "rank_inspections"]
        ranked = sorted(set(urgent) | set(bn_act), key=lambda a: (-inf.risk.get(a, {}).get("p_fail", 0.0), -inf.risk.get(a, {}).get("p_ml", 0.0), a))
        if ranked:
            route = plan_route(kb, ranked)
            rows = []
            for a in ranked:
                r = inf.risk.get(a, {})
                rows.append({"asset": a, "p_fail": round(r.get("p_fail", 0.0), 4), "p_ml": None if "p_ml" not in r else round(r["p_ml"], 4),
                             "logic": ent[f"InspectUrgent({a})"].status.value, "confidence": round(r["decision"]["confidence"], 3) if r else None})
            util = sum(x["p_fail"] * dc["cost_failure"] - dc["cost_inspect"] for x in rows)
            confs = [x["confidence"] for x in rows if x["confidence"] is not None]
            go(self._mk("rank_inspections", {"ranking": rows, "route": route["order"]},
                        f"{len(rows)} assets ranked by failure risk; A* route cost {route['cost']} ({route['expanded']} states)",
                        [ent[f"InspectUrgent({a})"] for a in ranked], utility=round(util, 3), risk=1 - min(confs) if confs else 0.0))

        # c) escalations: abstention, conflicts, contradictions, injection
        abst = sorted(a for a, r in inf.risk.items() if r["decision"]["status"] == "abstain" and a not in urgent)
        if abst:
            self._escalate(f"Calibrated abstention on {abst}: evidence insufficient or too close to break-even; not ranking or scheduling them.", [], now, res)
        for c in conflicts:
            self._escalate(f"Knowledge conflict on {c.atom}: {c.note}. No conclusions drawn from it.", c.fact_ids, now, res)
        for w in windows:
            e = ent.get(f"SafeToSchedule({w})")
            if e and e.status == Entailment.CONTRADICTED and e.supporting_ids and e.conflicting_ids:   # evidence on BOTH sides
                self._escalate(f"Contradictory evidence about window {w} (safe AND not safe); window excluded.", e.conflicting_ids[:10], now, res)
        flagged = [p for p in percepts if "possible_prompt_injection" in p.validation_notes]
        reqs = kb.untrusted_requests[self._req_seen:]
        self._req_seen = len(kb.untrusted_requests)
        for q in reqs:   # text can ask for a tool, but can never authorise one
            e = ask(f"AllowedAction({q['tool']})")
            self._escalate(f"Untrusted text from {q['source']} requested tool '{q['tool']}' (policy status: {e.status.value}); request ignored, not executed.", [q["percept_id"]], now, res)
        for p in flagged:
            if not reqs:
                self._escalate(f"Possible prompt injection in {p.source}:{p.entity_id}; content treated as untrusted data.", [p.percept_id], now, res)

        # d) schedule (tier 2 -> approval)
        if urgent:
            available = {s for s, w in slot_windows(kb).items() if ent[f"SafeToSchedule({w})"].status == Entailment.ENTAILED}
            P, jobs = build_schedule(kb, urgent, available)
            pref = preferred_schedule(P, jobs)
            viol = schedule_violations(P, pref, jobs)
            note = "preferred (earliest-slot) schedule satisfies all constraints"
            chosen = pref
            if viol:
                self.audit.emit("policy", "policy", f"preferred schedule rejected: {'; '.join(viol)[:350]}")
                chosen = solve_schedule(P)
                note = f"CSP alternative; preferred schedule violated: {'; '.join(viol)[:250]}"
            if chosen is None:
                self._escalate("No feasible schedule: every candidate violates hard constraints (windows not safe, skills or capacity).", [], now, res)
            else:
                crews, sw = self.cfg["scheduling"]["crews"], slot_windows(kb)
                assignment = {jobs[i]["asset"]: {"crew": crews[c]["id"], "slot": s, "window": sw[s]} for i, (c, s) in sorted(chosen.items())}
                used = sorted({a["window"] for a in assignment.values()})
                util = sum(inf.risk.get(a, {}).get("p_fail", 0.0) * dc["cost_failure"] - dc["cost_inspect"] for a in urgent)
                go(self._mk("propose_schedule", {"jobs": jobs, "assignment": assignment}, f"{len(jobs)} urgent jobs; {note}",
                            [ent[f"InspectUrgent({a})"] for a in urgent] + [ent[f"SafeToSchedule({w})"] for w in used],
                            utility=round(util, 3), risk=0.0))

        # e) report (tier 1)
        go(self._mk("write_report", {"summary": f"urgent={urgent}; ranked={ranked}; conflicts={len(conflicts)}; abstentions={abst}", "refs": []},
                    "cycle summary"))
        return new

    # ------------------------------------------------------------------ human decisions / external requests
    def approve(self, action_id: str, approver: str, now: datetime | None = None) -> tuple[PolicyDecision, ToolResult | None]:
        now = now or datetime.now(timezone.utc)
        p = self.approvals.pending.get(action_id)
        if self.kill.active:
            self.audit.emit("kill_switch", approver, f"approval of {action_id} refused: kill switch active")
            return PolicyDecision(action_id=action_id, tool=p.tool if p else "?", allowed=False, blocked=True, reasons=["kill switch is active"]), None
        if p is None or not approver.strip():
            return PolicyDecision(action_id=action_id, tool="?", allowed=False, blocked=True, reasons=["no such pending action or missing approver"]), None
        fresh = [entailment.check(e.query, self.kb, now) for e in p.entailment]      # re-verify against CURRENT knowledge
        p2 = p.model_copy(update={"entailment": fresh})
        self.approvals.approve(action_id, approver)
        d = self.policy.evaluate(p2, now)
        if not d.allowed:
            self.approvals.approved.pop(action_id, None)
            p.approval = ApprovalState.BLOCKED
            self.audit.emit("policy", approver, f"approval of {action_id} could not be applied: {'; '.join(d.reasons)}"[:500], [action_id])
            return d, None
        p.approval = ApprovalState.APPROVED
        r = self.registry.execute(p2, now)
        self.policy.record_execution(now)
        self.audit.emit("action", approver, f"APPROVED and executed {p.tool} ({action_id}) -> {'OK' if r.ok else r.error}", [action_id])
        return d, r

    def reject(self, action_id: str, approver: str) -> bool:
        ok = self.approvals.reject(action_id, approver)
        self.audit.emit("action", approver, f"{'rejected' if ok else 'reject failed for'} {action_id}", [action_id])
        return ok

    def submit_action_request(self, tool: str, args: dict[str, Any], source: str = "planner", now: datetime | None = None) -> tuple[PolicyDecision, ToolResult | None]:
        """Entry point for any component that wants a tool run. Everything goes through the same policy gate."""
        now = now or datetime.now(timezone.utc)
        res = CycleResult(run_id=self.run_id, cycle=self.cycle, iterations=0, status="completed")
        allowed_q = entailment.check(f"AllowedAction({tool})", self.kb, now) if tool.replace("_", "").isalnum() else None
        p = self._mk(tool, args, f"requested by {source}", [allowed_q] if allowed_q else [])
        self.audit.emit("action", source, f"external request for tool '{tool[:40]}'")
        r = self._dispatch(p, now, res)
        return res.decisions[0], r
