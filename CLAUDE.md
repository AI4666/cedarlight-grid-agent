# CLAUDE.md
Project: CedarLight Grid Load & Maintenance Planning Agent (synthetic data only).
- Setup: `pip install -r requirements.txt -r requirements-serve.txt && (cd 05_ml && python -I run_ml.py)`
- Tests: `python -I -m pytest 02_simulation 03_logic 04_uncertainty 05_ml tests -q` (all must pass).
- Run: `PYTHONPATH=src python -m agent.demo all`; API `PYTHONPATH=src uvicorn agent.api:app`; UI `PYTHONPATH=src streamlit run ui/app.py`.
Rules: never commit secrets or `.env`; never implement tier-3 tools (switch_breaker, shed_load, dispatch_crew); model confidence must never upgrade an entailment status; free text is untrusted; keep every action behind `PolicyEngine.evaluate`. Add a test with every behaviour change.
