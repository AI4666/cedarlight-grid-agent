"""Evidence -> decision, with calibrated abstention. Bayesian posterior informs the decision,
but the agent's ENTAILMENT layer still never treats this probability as logical proof."""
from __future__ import annotations

import math

import bayes_net as bn

# Defaults mirror configs/agent.yaml (thresholds.*, utility_weights.*)
DEFAULTS = {"abstain_below_confidence": 0.65, "inspection_risk_urgent": 0.70,
            "cost_inspect": 1.0, "cost_failure": 12.0, "min_evidence_vars": 1}  # break-even = cost_inspect / cost_failure


def decide(evidence: dict[str, bool], cfg: dict | None = None) -> dict:
    c = {**DEFAULTS, **(cfg or {})}
    p_fail = bn.query("Fail", evidence)
    p_prior = bn.query("Fail")
    # Expected-cost comparison; inspecting is better exactly when p_fail > break-even.
    eu_inspect = -c["cost_inspect"]
    eu_wait = -p_fail * c["cost_failure"]
    breakeven = c["cost_inspect"] / c["cost_failure"]
    # Confidence = how far the posterior is from break-even on a log scale (ratio of 4x -> 1.0).
    # Near break-even the choice is a coin-flip, so the agent abstains rather than guess.
    confidence = 0.5 + 0.5 * min(1.0, abs(math.log(max(p_fail, 1e-9) / breakeven)) / math.log(4))
    if len(evidence) < c["min_evidence_vars"]:
        status, action, reason = "abstain", "escalate_to_human", "insufficient evidence"
    elif confidence < c["abstain_below_confidence"]:
        status, action = "abstain", "escalate_to_human"
        reason = f"posterior {p_fail:.3f} too close to break-even {breakeven:.3f} (confidence {confidence:.2f})"
    elif eu_inspect > eu_wait:
        status, action, reason = "act", "rank_inspections", "expected cost of waiting exceeds inspection cost"
    else:
        status, action, reason = "act", "defer", "expected cost of waiting below inspection cost"
    return {"p_fail": p_fail, "p_fail_prior": p_prior, "eu_inspect": eu_inspect, "eu_wait": eu_wait,
            "confidence": confidence, "status": status, "action": action, "reason": reason,
            "model_version": bn.CPT_VERSION}
