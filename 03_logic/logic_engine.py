"""Minimal logical-agent engine: FOL Horn forward chaining (with unification),
backward-chaining proof check, propositional resolution, and four-valued entailment.

Entailment statuses (never derived from model confidence):
  ENTAILED / CONTRADICTED / UNCERTAIN / UNKNOWN
"""
from __future__ import annotations

import re
from itertools import count
from pathlib import Path

import yaml

HERE = Path(__file__).parent
ATOM_RE = re.compile(r"^(~?)([A-Za-z_]\w*)\((.*)\)$")


# ---------- atoms ----------
def parse(atom: str) -> tuple[bool, str, tuple[str, ...]]:
    m = ATOM_RE.match(atom.strip())
    if not m:
        raise ValueError(f"bad atom: {atom!r}")
    neg, pred, args = m.groups()
    arg_t = tuple(a.strip() for a in args.split(",")) if args.strip() else ()
    return bool(neg), pred, arg_t


def fmt(neg: bool, pred: str, args: tuple[str, ...]) -> str:
    return f"{'~' if neg else ''}{pred}({','.join(args)})"


def negate(atom: str) -> str:
    neg, p, a = parse(atom)
    return fmt(not neg, p, a)


def is_var(t: str) -> bool:
    return t.startswith("?")


def unify(pattern: tuple[str, ...], ground: tuple[str, ...], s: dict) -> dict | None:
    if len(pattern) != len(ground):
        return None
    s = dict(s)
    for p, g in zip(pattern, ground):
        if is_var(p):
            if s.get(p, g) != g:
                return None
            s[p] = g
        elif p != g:
            return None
    return s


def subst(atom: str, s: dict) -> str:
    neg, p, a = parse(atom)
    return fmt(neg, p, tuple(s.get(t, t) for t in a))


# ---------- rule loading ----------
def load_rules(path: Path | None = None) -> dict:
    with open(path or HERE / "rules.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- forward chaining ----------
def _match_body(body: list[str], facts: set[str], s: dict):
    if not body:
        yield s
        return
    neg, pred, args = parse(body[0])
    for fact in facts:
        fneg, fpred, fargs = parse(fact)
        if fneg == neg and fpred == pred:
            s2 = unify(args, fargs, s)
            if s2 is not None:
                yield from _match_body(body[1:], facts, s2)


def forward_chain(facts: set[str], rules: list[dict]) -> tuple[set[str], dict[str, tuple[str, list[str]]]]:
    """Return (closure, provenance). provenance[atom] = (rule_id, premises) or ('fact', [])."""
    closure = set(facts)
    prov: dict[str, tuple[str, list[str]]] = {f: ("fact", []) for f in facts}
    changed = True
    while changed:
        changed = False
        for r in rules:
            for s in list(_match_body(r["body"], closure, {})):
                head = subst(r["head"], s)
                if head not in closure:
                    closure.add(head)
                    prov[head] = (r["id"], [subst(b, s) for b in r["body"]])
                    changed = True
    return closure, prov


def explain(atom: str, prov: dict) -> list[str]:
    """Concise derivation trace (list of 'atom <- rule(premises)'), no free-form reasoning."""
    out, seen = [], set()

    def walk(a):
        if a in seen or a not in prov:
            return
        seen.add(a)
        rule, premises = prov[a]
        if rule != "fact":
            for p in premises:
                walk(p)
            out.append(f"{a} <- {rule}({', '.join(premises)})")

    walk(atom)
    return out


# ---------- backward chaining (goal-directed proof) ----------
_rename_counter = count()


def _rename(rule: dict) -> tuple[str, list[str]]:
    """Standardize variables apart so recursive rule uses do not clash."""
    n = next(_rename_counter)
    sub = {v: f"{v}_{n}" for v in set(re.findall(r"\?\w+", rule["head"] + " ".join(rule["body"])))}
    return subst(rule["head"], sub), [subst(b, sub) for b in rule["body"]]


def _walk(t: str, s: dict) -> str:
    while is_var(t) and t in s:
        t = s[t]
    return t


def _unify_terms(a: tuple[str, ...], b: tuple[str, ...], s: dict) -> dict | None:
    if len(a) != len(b):
        return None
    s = dict(s)
    for x, y in zip(a, b):
        x, y = _walk(x, s), _walk(y, s)
        if x == y:
            continue
        if is_var(x):
            s[x] = y
        elif is_var(y):
            s[y] = x
        else:
            return None
    return s


def _prove(goals: list[str], facts: set[str], rules: list[dict], s: dict, depth: int):
    if not goals:
        yield s
        return
    if depth == 0:
        return
    g, rest = goals[0], goals[1:]
    gneg, gpred, gargs = parse(g)
    for f in facts:  # match known facts
        fneg, fpred, fargs = parse(f)
        if (fneg, fpred) == (gneg, gpred):
            s2 = _unify_terms(gargs, fargs, s)
            if s2 is not None:
                yield from _prove(rest, facts, rules, s2, depth)
    for r in rules:  # resolve against rule heads
        hneg, hpred, _ = parse(r["head"])
        if (hneg, hpred) != (gneg, gpred):
            continue
        head, body = _rename(r)
        s2 = _unify_terms(gargs, parse(head)[2], s)
        if s2 is not None:
            yield from _prove(body + rest, facts, rules, s2, depth - 1)


def backward_chain(goal: str, facts: set[str], rules: list[dict], depth: int = 8) -> bool:
    """True iff `goal` (ground or with variables) is provable by SLD-style backward chaining."""
    return next(_prove([goal], facts, rules, {}, depth), None) is not None


# ---------- four-valued entailment ----------
def entails(query: str, facts: set[str], rules: list[dict], uncertain_facts: set[str] | None = None) -> dict:
    """Decide ENTAILED / CONTRADICTED / UNCERTAIN / UNKNOWN for a ground query atom."""
    uncertain_facts = uncertain_facts or set()
    closure, prov = forward_chain(facts, rules)
    q, nq = query, negate(query)
    in_q, in_nq = q in closure, nq in closure

    if in_q and in_nq:
        return {"status": "contradicted", "conflict": True,
                "support": explain(q, prov), "conflicting": explain(nq, prov),
                "rationale": "Both the query and its negation are derivable: KB conflict, no entailment granted."}
    if in_q:
        return {"status": "entailed", "conflict": False, "support": explain(q, prov), "conflicting": [],
                "rationale": "Derivable from trusted facts and rules."}
    if in_nq:
        return {"status": "contradicted", "conflict": False, "support": [], "conflicting": explain(nq, prov),
                "rationale": "Negation is derivable from trusted facts and rules."}

    # partial support: a rule for q has at least one satisfied, query-anchored body literal, AND at least one joint binding
    # of those literals is not ruled out by an explicitly NEGATED remaining premise (open world: absence is not falsity,
    # but an explicit ~premise does block the rule).
    qneg, qpred, qargs = parse(q)
    partial, blocked_rules = [], []
    for r in rules:
        hneg, hpred, hargs = parse(r["head"])
        if (hneg, hpred) != (qneg, qpred):
            continue
        s = unify(hargs, qargs, {})
        if s is None:
            continue
        anchored = [subst(b, s) for b in r["body"]]
        sat = [b for b in anchored if any(not is_var(t) for t in parse(b)[2]) and any(True for _ in _match_body([b], closure, {}))]
        if not sat:
            continue
        rest = [b for b in anchored if b not in sat]
        unblocked = [s2 for s2 in _match_body(sat, closure, {})
                     if not any(not any(is_var(t) for t in parse(subst(x, s2))[2]) and negate(subst(x, s2)) in closure for x in rest)]
        if unblocked:
            partial.append(f"{r['id']}: {len(sat)}/{len(r['body'])} premises hold ({', '.join(sat)})")
        else:
            blocked_rules.append(r["id"])
    if q in uncertain_facts:
        partial.append(f"low-confidence fact asserts {q}")
    if partial:
        return {"status": "uncertain", "conflict": False, "support": partial, "conflicting": [],
                "rationale": "Partial support only; not enough to entail."}
    if blocked_rules:
        return {"status": "unknown", "conflict": False, "support": [], "conflicting": [],
                "rationale": f"Rules {blocked_rules} are blocked by explicitly negated premises; no remaining support."}
    return {"status": "unknown", "conflict": False, "support": [], "conflicting": [],
            "rationale": "No relevant premises in the KB."}


# ---------- propositional resolution (CNF) ----------
def _neg_lit(l: str) -> str:
    return l[1:] if l.startswith("~") else "~" + l


def resolution_entails(clauses: list[list[str]], query: str, max_clauses: int = 5000) -> bool:
    """KB |= query iff KB AND ~query is unsatisfiable (refutation by resolution)."""
    kb = {frozenset(c) for c in clauses} | {frozenset([_neg_lit(query)])}
    kb = {c for c in kb if not any(_neg_lit(l) in c for l in c)}  # drop tautologies
    while True:
        new = set()
        lst = list(kb)
        for i, c1 in enumerate(lst):
            for c2 in lst[i + 1:]:
                for lit in c1:
                    if _neg_lit(lit) in c2:
                        res = (c1 - {lit}) | (c2 - {_neg_lit(lit)})
                        if not res:
                            return True
                        if not any(_neg_lit(l) in res for l in res):
                            new.add(frozenset(res))
        if new <= kb or len(kb) > max_clauses:
            return False
        kb |= new
