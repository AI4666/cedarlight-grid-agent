"""STAGE 2 - PROPOSITIONS. Translate percepts into testable symbolic statements and model-ready features,
and identify contradictions within a batch.

Free text is NEVER interpreted as an instruction. If text mentions a tool, it becomes an UNTRUSTED
`ToolRequest(tool)` proposition that no rule uses, so it can never entail permission to act."""
from __future__ import annotations

import re

from .schemas import Percept, Proposition, ValidationStatus

TOOL_MENTION = re.compile(r"\b(?:call|execute|run|invoke|use)\s+(?:the\s+)?([a-z][a-z0-9_]{2,40})", re.I)


def _p(pred, args, truth, conf, pid, feats=None):
    return Proposition(predicate=pred, args=list(args), truth=truth, confidence=round(conf, 4),
                       features=feats or {}, derived_from=[pid])


def to_propositions(p: Percept, cfg: dict, known_tools: set[str]) -> list[Proposition]:
    if p.validation_status == ValidationStatus.REJECTED:
        return []
    th, v, e = cfg["thresholds"], p.values, p.entity_id
    # Suspect percepts keep their content but lose trust: confidence is halved so they fall below the trust threshold.
    conf = p.quality * (0.5 if p.validation_status == ValidationStatus.SUSPECT else 1.0)
    out: list[Proposition] = []
    num = lambda k: float(v[k]) if isinstance(v.get(k), (int, float)) and not isinstance(v.get(k), bool) else None  # noqa: E731

    if p.source == "meter" and num("load_ratio") is not None:
        high = num("load_ratio") >= th["overload_load_ratio"]
        out.append(_p("LoadRatioHigh", [e], high, conf, p.percept_id, {"load_ratio": num("load_ratio")}))
    if p.source == "weather":
        if num("temp_c") is not None:
            out.append(_p("TempHigh", [e], num("temp_c") >= th["heat_advisory_temp_c"], conf, p.percept_id, {"temp_c": num("temp_c")}))
        if isinstance(v.get("storm"), bool):
            out.append(_p("Storm", [e], v["storm"], conf, p.percept_id))
    if p.source == "asset_condition":
        if num("condition") is not None:
            out.append(_p("ConditionPoor", [e], num("condition") < th["condition_poor_below"], conf, p.percept_id,
                          {"condition": num("condition")}))
        if isinstance(v.get("past_due"), bool):
            out.append(_p("PastDue", [e], v["past_due"], conf, p.percept_id))
    if p.source == "maintenance_window":
        for key, pred in (("open", "WindowOpen"), ("crew_available", "CrewAvailable"), ("clearance_ok", "ClearanceOk")):
            if isinstance(v.get(key), bool):
                out.append(_p(pred, [e], v[key], conf, p.percept_id))

    if p.source == "tool_result" and isinstance(v.get("ok"), bool):   # outcome of our own action, fed back into memory
        out.append(_p("ToolSucceeded", [e], v["ok"], conf, p.percept_id))

    # Untrusted text -> ToolRequest propositions (never trusted; never used by rules)
    for val in v.values():
        if isinstance(val, str):
            for m in TOOL_MENTION.finditer(val):
                out.append(_p("ToolRequest", [m.group(1).lower()], True, 0.0, p.percept_id))
            for t in known_tools:
                if t in val.lower() and not any(o.predicate == "ToolRequest" and o.args == [t] for o in out):
                    out.append(_p("ToolRequest", [t], True, 0.0, p.percept_id))
    return out


def find_contradictions(props: list[Proposition]) -> list[tuple[str, str]]:
    """Pairs of propositions in the same batch with the same atom but opposite truth values."""
    seen: dict[tuple, Proposition] = {}
    pairs = []
    for pr in props:
        key = (pr.predicate, tuple(pr.args))
        if key in seen and seen[key].truth != pr.truth:
            seen[key].contradicts.append(pr.prop_id)
            pr.contradicts.append(seen[key].prop_id)
            pairs.append((seen[key].prop_id, pr.prop_id))
        else:
            seen.setdefault(key, pr)
    return pairs
