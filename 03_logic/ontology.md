# 03_logic/ontology.md — First-Order Logic & Knowledge Representation

## 1. Taxonomy (is-a)
```
Asset
├── Substation
├── Feeder
└── Transformer
Customer
└── CriticalCustomer  (hospital, water plant)
Window            (maintenance window)
Zone
Tool
├── AllowedTool   (tier 0-2)
└── ForbiddenTool (tier 3)
```
FOL form: `∀x Substation(x) → Asset(x)`, `∀x Feeder(x) → Asset(x)`, `∀x Transformer(x) → Asset(x)`.

## 2. Knowledge-graph relations
| Relation | Domain → Range | Properties |
|---|---|---|
| `Feeds(a, b)` | Asset → Asset ∪ Customer | direct supply edge, acyclic |
| `Upstream(a, b)` | Asset → Asset ∪ Customer | transitive closure of `Feeds` |
| `LocatedIn(a, z)` | Asset → Zone | functional (each asset in exactly one zone) |
| `WindowIn(w, z)` | Window → Zone | functional |
| `CriticalLoad(c)` | Customer | unary |

Example graph (used by tests): `Sub1 → F3 → T7 → Hosp1`, with `LocatedIn(T7, ZoneB)`.

## 3. Quantified axioms (FOL)
1. Overload: `∀t LoadRatioHigh(t) → Overloaded(t)`
2. Heat: `∀z TempHigh(z) → HeatAdvisory(z)`
3. Risk: `∀t ∀z Overloaded(t) ∧ LocatedIn(t,z) ∧ HeatAdvisory(z) → AtRisk(t)`
4. Urgency: `∀a AtRisk(a) ∧ (ConditionPoor(a) ∨ PastDue(a)) → InspectUrgent(a)`
5. Blocking: `∀w ∀z WindowIn(w,z) ∧ Storm(z) → Blocked(w)`
6. Safety (hard): `∀w Blocked(w) → ¬SafeToSchedule(w)`
7. Transitivity: `∀a∀b∀c Feeds(a,b) ∧ Upstream(b,c) → Upstream(a,c)`
8. Impact: `∀a ∀c Upstream(a,c) ∧ CriticalLoad(c) → ImpactHigh(a)`
9. Tool policy (hard): `∀t Forbidden(t) → ¬AllowedAction(t)`
10. Existential example: `∃t Transformer(t) ∧ InspectUrgent(t)` — "some transformer needs urgent inspection" (witness found by search over KB constants).

## 4. Unification and inference rules
- **Unification:** `unify(LocatedIn(?t,?z), LocatedIn(T7,ZoneB))` → `{?t:T7, ?z:ZoneB}`; implemented in `logic_engine.unify`.
- **Generalized Modus Ponens:** used by forward chaining over Horn rules (`rules.yaml`).
- **SLD backward chaining:** goal-directed proof with variable standardization (`backward_chain`).
- **Resolution:** refutation over CNF clauses (`resolution_entails`).
- All Horn rules are definite clauses, so forward chaining is sound and complete for the positive fragment. Negative heads (`~SafeToSchedule`, `~AllowedAction`) are explicit exclusion rules; if `p` and `~p` both become derivable the engine reports a **conflict** rather than entailing either.

## 5. Closed vs open world
The KB is **open-world**: absence of a fact means *unknown*, not false. `~p` holds only if explicitly derived. This is why the engine has a distinct `unknown` status and never uses negation-as-failure.

## 6. Trust boundary
Facts carry provenance. Only facts from validated percepts (status `valid`, quality ≥ `min_percept_quality`) or static KB entries enter the trusted fact set. `ToolRequest(...)` atoms originate from untrusted text and **never** appear in rule bodies, so a request cannot, by itself, entail permission to act.
