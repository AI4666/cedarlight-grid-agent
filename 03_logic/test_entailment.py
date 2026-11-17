"""Entailment test cases for the logical agent (run: python -I -m pytest 03_logic -q)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import logic_engine as L  # noqa: E402

RULES = L.load_rules()
HORN = RULES["horn_rules"]
CNF = [c["clause"] for c in RULES["cnf_clauses"]]

STATIC = {
    "LocatedIn(T7,ZoneB)", "ConditionPoor(T7)",
    "Feeds(Sub1,F3)", "Feeds(F3,T7)", "Feeds(T7,Hosp1)", "CriticalLoad(Hosp1)",
    "WindowIn(W1,ZoneB)",
    "ToolOnAllowList(write_report)", "ToolOnAllowList(propose_schedule)",
    "Forbidden(switch_breaker)",
}


def ask(query, percepts=(), uncertain=()):
    return L.entails(query, STATIC | set(percepts), HORN, set(uncertain))


# 1. NORMAL: complete, consistent percepts -> supported conclusion
def test_normal_entailed_urgent_inspection():
    r = ask("InspectUrgent(T7)", {"LoadRatioHigh(T7)", "TempHigh(ZoneB)"})
    assert r["status"] == "entailed"
    assert any("R4" in s for s in r["support"])


# 2. UNKNOWN: nothing relevant known about T9
def test_unknown_when_no_premises():
    assert ask("InspectUrgent(T9)")["status"] == "unknown"


# 3. UNCERTAIN: partial evidence (overload seen, no heat advisory) -> do not entail
def test_uncertain_with_partial_support():
    r = ask("AtRisk(T7)", {"LoadRatioHigh(T7)"})
    assert r["status"] == "uncertain"
    assert "R3" in r["support"][0]


# 4. CONTRADICTION: one source says no storm, another says storm -> window both safe and blocked
def test_contradiction_flagged_not_entailed():
    r = ask("SafeToSchedule(W1)",
            {"WindowOpen(W1)", "CrewAvailable(W1)", "ClearanceOk(W1)", "~Storm(ZoneB)", "Storm(ZoneB)"})
    assert r["status"] == "contradicted" and r["conflict"] is True
    assert r["support"] and r["conflicting"]  # both sides reported


def test_storm_alone_makes_window_unsafe_without_conflict():
    r = ask("SafeToSchedule(W1)", {"WindowOpen(W1)", "CrewAvailable(W1)", "ClearanceOk(W1)", "Storm(ZoneB)"})
    assert r["status"] == "contradicted" and r["conflict"] is False       # blocked, but evidence is consistent


def test_safe_needs_explicit_no_storm_observation():
    base = {"WindowOpen(W1)", "CrewAvailable(W1)", "ClearanceOk(W1)"}
    assert ask("SafeToSchedule(W1)", base)["status"] == "uncertain"       # storm status unknown -> not safe
    assert ask("SafeToSchedule(W1)", base | {"~Storm(ZoneB)"})["status"] == "entailed"


# 5. FOL / TRANSITIVE: impact reaches a critical load through the supply chain
def test_transitive_impact_high():
    assert ask("ImpactHigh(Sub1)")["status"] == "entailed"
    assert ask("ImpactHigh(Hosp1)")["status"] == "unknown"  # a customer is not upstream of itself


# 6. ROBUSTNESS / INJECTION: untrusted ToolRequest cannot entail permission
def test_injection_cannot_entail_forbidden_tool():
    r = ask("AllowedAction(switch_breaker)", {"ToolRequest(switch_breaker)"})
    assert r["status"] == "contradicted"        # hard policy rule R13
    assert ask("AllowedAction(write_report)")["status"] == "entailed"
    assert ask("AllowedAction(delete_all)", {"ToolRequest(delete_all)"})["status"] == "unknown"


# 7. LOW-CONFIDENCE fact never promoted to entailment
def test_low_confidence_fact_is_only_uncertain():
    r = ask("Storm(ZoneB)", uncertain={"Storm(ZoneB)"})
    assert r["status"] == "uncertain"


# 8. BACKWARD CHAINING agrees with forward chaining
def test_backward_chaining_matches():
    facts = STATIC | {"LoadRatioHigh(T7)", "TempHigh(ZoneB)"}
    assert L.backward_chain("InspectUrgent(T7)", facts, HORN)
    assert not L.backward_chain("InspectUrgent(T9)", facts, HORN)


# 9. RESOLUTION over CNF clauses
def test_resolution_entails_and_refutes():
    kb = CNF + [["Overloaded"], ["HeatAdvisory"], ["ConditionPoor"]]
    assert L.resolution_entails(kb, "InspectUrgent")
    assert not L.resolution_entails(CNF + [["Overloaded"]], "InspectUrgent")
    assert L.resolution_entails(CNF + [["Storm"]], "~SafeToSchedule")
    assert L.resolution_entails(CNF + [["Blocked"]], "~SafeToSchedule")


# 10. Explicitly negated premise blocks a rule; absent premise merely leaves it uncertain
def test_explicit_negative_premise_blocks_partial_support():
    base = {"Feeds(S,A)", "Feeds(A,H)"}
    open_world = L.entails("ImpactHigh(S)", base, HORN)
    assert open_world["status"] == "uncertain"                       # CriticalLoad(H) simply unknown
    blocked = L.entails("ImpactHigh(S)", base | {"~CriticalLoad(H)", "~CriticalLoad(A)"}, HORN)
    assert blocked["status"] == "unknown" and "blocked" in blocked["rationale"]
    assert L.entails("ImpactHigh(S)", base | {"CriticalLoad(H)"}, HORN)["status"] == "entailed"
