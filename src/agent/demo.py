"""CLI demo: python -m agent.demo [scenario|all]   (run from repo root with PYTHONPATH=src)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone

from .orchestrator import Agent
from .scenarios import SCENARIOS, build


def show(name: str) -> None:
    now = datetime.now(timezone.utc)
    a = Agent()
    r = a.run(build(name, now), now)
    print(f"\n=== {name.upper()}: {SCENARIOS[name]}")
    print(f"status={r.status} iterations={r.iterations} model={a.model.source}")
    for e in r.entailments:
        print(f"  ENTAIL  {e.query:<28} {e.status.value:<12} support={len(e.supporting_ids)} conflict={len(e.conflicting_ids)}")
    for p in r.proposals:
        print(f"  ACTION  {p.tool:<18} tier={p.tier.value} approval={p.approval.value}")
    for m in r.escalations:
        print(f"  ESCALATE {m[:110]}")
    for p in r.proposals:
        if p.tool == "propose_schedule":
            print("  Pending human approval:", p.action_id)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    for n in (SCENARIOS if arg == "all" else [arg]):
        show(n)
