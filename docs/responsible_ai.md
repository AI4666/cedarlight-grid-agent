# Responsible AI notes

**Scope.** Decision support for a fictitious cooperative using synthetic data. It recommends; humans decide. It must not be used on a real grid without independent validation.

**Data.** All data is simulated (`05_ml/data_gen.py`, `data/topology.json`). Synthetic generators encode our assumptions, so reported metrics describe the simulator, not real-world performance.

**Safety by design.** Tier-3 control tools are not implemented. Tier-2 actions need a named approver and are re-validated at approval time. Policy checks use symbolic entailment, never model confidence. Hard constraints (storm blackouts, crew skills, per-crew daily caps) are checked independently of the planner. Kill switch, rate limit, idempotency keys, bounded loop.

**Prompt injection.** Operator free text is untrusted. It can only become a zero-confidence `ToolRequest` fact, which is escalated and logged.

**Fairness.** In the ML evaluation (`05_ml`, see `metrics.json` → `fairness_test`) a zone-stratified quota narrows the recall gap between zones, with the older zone as the test case. The live agent's ranking does *not* apply that quota yet, and the demo topology has only two zones; review zone-level outcomes before relying on rankings. Critical loads (hospital) raise impact through rules, not learned weights.

**Transparency.** Every cycle emits audit events for all six stages with run id, seed and config hash; entailment results list the rule and fact ids that support or conflict.

**Known limitations.**
- Small hand-built topology (4 transformers, 2 zones); results will not transfer unchanged to larger networks.
- The learned model is a second opinion; the Bayesian and rule layers use hand-set parameters.
- The Streamlit UI, Docker image and HTTP layer were not run-tested in the authoring environment; CI does this.
- No real LLM explainer is wired in (`use_llm_explanations` is off); explanations are templated from traces.
