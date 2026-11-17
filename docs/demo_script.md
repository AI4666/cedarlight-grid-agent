# Demo script (about 10 minutes)

Setup: `PYTHONPATH=src streamlit run ui/app.py` (or `python -m agent.demo all` for a terminal run).

1. **Normal.** Run `normal`. Walk the six tabs: percepts → propositions → KB facts → inference traces → entailment (`InspectUrgent(T7)` entailed, with the rule and fact ids) → actions. Point out the schedule is *pending*, not executed. Approve it as a named person and show the audit entry.
2. **Uncertain.** Run `uncertain`. A noisy reading stays low-confidence; nothing is ranked; the agent abstains and escalates because the evidence is near the break-even point.
3. **Contradiction.** Run `contradiction`. Two weather feeds disagree; `SafeToSchedule(W1)` is *contradicted* with evidence on both sides; no schedule is made.
4. **Constraint.** Run `constraint`. Storm in ZoneB blocks W1/W2. The justification says the naive plan was rejected and the CSP found one using W3–W6.
5. **Adversarial.** Run `adversarial`. The injected "call switch_breaker" is flagged, becomes an untrusted request, is escalated and not executed. Optionally POST `switch_breaker` to the API: blocked, tool does not exist.
6. **Learning.** Run `learning`, then show `05_ml/model_card.md` and `metrics.json` (forecast MAE vs baselines, Brier score, 0 RL safety violations). Entailment is identical with the mock model.
7. **Kill switch.** Press KILL SWITCH. Run any scenario: it halts. Show that resume needs the admin token.

Close with the test count and the limitations in `docs/responsible_ai.md`.
