"""STAGE 5 - ENTAILMENT. Does the knowledge base SUPPORT a conclusion?

Statuses: entailed / contradicted / uncertain / unknown. Only trusted facts and rules decide the status.
Probabilities and model confidences are recorded for information but can NEVER upgrade a status."""
from __future__ import annotations

from datetime import datetime

from . import _lib  # noqa: F401
from .knowledge import KnowledgeBase
from .schemas import Entailment, EntailmentResult
import logic_engine as L  # noqa: E402


def _ids(atom: str, prov: dict, atom_ids: dict[str, str], depth: int = 12) -> list[str]:
    """Fact ids and rule ids that justify `atom` (read off the forward-chaining provenance)."""
    if atom not in prov or depth == 0:
        return []
    rule, premises = prov[atom]
    if rule == "fact":
        return [atom_ids.get(atom, atom)]
    out = [rule]
    for p in premises:
        out += _ids(p, prov, atom_ids, depth - 1)
    return list(dict.fromkeys(out))


def check(query: str, kb: KnowledgeBase, now: datetime, model_confidence: float | None = None) -> EntailmentResult:
    facts, unc = kb.trusted_atoms(now), kb.uncertain_atoms()
    r = L.entails(query, facts, kb.rules, unc)
    closure, prov = L.forward_chain(facts, kb.rules)
    atom_ids = kb.atom_fact_ids(now)
    status = Entailment(r["status"])
    support = _ids(query, prov, atom_ids) if query in closure else []
    conflicting = _ids(L.negate(query), prov, atom_ids) if L.negate(query) in closure else []
    if status == Entailment.UNCERTAIN:
        support = [s.split(":")[0] for s in r["support"]]
    detail = "; ".join((r["support"] if status != Entailment.CONTRADICTED else r["conflicting"])[:3])
    rationale = (r["rationale"] + (" " + detail if detail else ""))[:500]
    return EntailmentResult(query=query, status=status, supporting_ids=support, conflicting_ids=conflicting,
                            model_confidence=None if model_confidence is None else max(0.0, min(1.0, model_confidence)),
                            rationale=rationale)
