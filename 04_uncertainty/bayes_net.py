"""Discrete Bayesian network for asset failure risk with exact inference by enumeration.

Structure (all variables boolean):
    Old ──► Poor ──┐
                   ├──► Fail
    Heat ─► Overload ┘
               └────► ReadingHigh   (noisy sensor of overload)

CPTs are synthetic, documented in README.md, and versioned in CPT_VERSION.
"""
from __future__ import annotations

from itertools import product

CPT_VERSION = "0.1.0"

# P(X=True | parents) keyed by parent-value tuples (order given in PARENTS).
PARENTS: dict[str, tuple[str, ...]] = {
    "Old": (),
    "Heat": (),
    "Poor": ("Old",),
    "Overload": ("Heat",),
    "ReadingHigh": ("Overload",),
    "Fail": ("Poor", "Overload"),
}

CPT: dict[str, dict[tuple[bool, ...], float]] = {
    "Old": {(): 0.40},
    "Heat": {(): 0.20},
    "Poor": {(True,): 0.35, (False,): 0.08},
    "Overload": {(True,): 0.50, (False,): 0.15},
    "ReadingHigh": {(True,): 0.90, (False,): 0.10},          # sensor sensitivity / false-alarm rate
    "Fail": {(True, True): 0.40, (True, False): 0.12,
             (False, True): 0.10, (False, False): 0.01},
}

ORDER = ["Old", "Heat", "Poor", "Overload", "ReadingHigh", "Fail"]  # topological


def _p(var: str, value: bool, assignment: dict[str, bool]) -> float:
    key = tuple(assignment[p] for p in PARENTS[var])
    p_true = CPT[var][key]
    return p_true if value else 1.0 - p_true


def joint(assignment: dict[str, bool]) -> float:
    prob = 1.0
    for v in ORDER:
        prob *= _p(v, assignment[v], assignment)
    return prob


def query(var: str, evidence: dict[str, bool] | None = None) -> float:
    """Return P(var=True | evidence) by full enumeration (6 binary vars -> 64 worlds)."""
    evidence = evidence or {}
    for k in evidence:
        if k not in CPT:
            raise KeyError(f"unknown variable {k!r}")
    num = den = 0.0
    for values in product([True, False], repeat=len(ORDER)):
        world = dict(zip(ORDER, values))
        if any(world[k] != v for k, v in evidence.items()):
            continue
        p = joint(world)
        den += p
        if world[var]:
            num += p
    if den == 0.0:
        raise ValueError("evidence has zero probability")
    return num / den


def validate() -> None:
    """Sanity checks: every CPT entry in [0,1], all parent combinations covered, joint sums to 1."""
    for v, table in CPT.items():
        need = set(product([True, False], repeat=len(PARENTS[v])))
        assert set(table) == need, f"{v}: incomplete CPT"
        assert all(0.0 <= p <= 1.0 for p in table.values()), f"{v}: probability out of range"
    total = sum(joint(dict(zip(ORDER, vals))) for vals in product([True, False], repeat=len(ORDER)))
    assert abs(total - 1.0) < 1e-9, f"joint sums to {total}"
