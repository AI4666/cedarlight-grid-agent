# CedarLight Grid Load & Maintenance Planning Agent

A fictitious cooperative (CedarLight Energy Cooperative), **synthetic data only**. The agent runs a transparent six-stage
cycle: **percepts → propositions → knowledge base → inference → entailment → action**. It forecasts demand, ranks
inspections and recommends crew schedules. Anything high-impact is only a *recommendation* that a human must approve.

## Layout
| Path | Purpose |
|---|---|
| `src/agent/` | perceive, proposition, knowledge, inference, entailment, policy, actions, orchestrator, service, API |
| `02_simulation/` | search (BFS/DFS/UCS/IDS/greedy/A*, 2-opt, annealing), CSP, adversarial search |
| `03_logic/` | vocabulary, Horn rules, forward/backward chaining, resolution, four-valued entailment |
| `04_uncertainty/` | Bayesian network, Kalman/HMM temporal models, break-even decision rule |
| `05_ml/` | forecast, failure classifier, Q-learning with safety mask, model card, metrics |
| `configs/agent.yaml`, `data/topology.json` | thresholds, tool tiers, policy, synthetic grid topology |
| `tests/` | six demo scenarios + safety/robustness/API tests |
| `ui/app.py` | Streamlit UI with one tab per stage |
| `docs/` | runbook pieces: monitoring and rollback, demo script, responsible AI |

## Quick start
```bash
pip install -r requirements.txt -r requirements-serve.txt
(cd 05_ml && python -I run_ml.py)                 # train synthetic models (~30 s); artifacts are git-ignored
python -I -m pytest 02_simulation 03_logic 04_uncertainty 05_ml tests -q
PYTHONPATH=src python -m agent.demo all           # run all six scenarios in the terminal
```

## Run the service
```bash
cp .env.example .env     # set CEDARLIGHT_API_KEY and CEDARLIGHT_ADMIN_TOKEN (never commit .env)
export $(grep -v '^#' .env | xargs)
PYTHONPATH=src uvicorn agent.api:app --port 8000
PYTHONPATH=src streamlit run ui/app.py            # UI (in-process, separate from the API)
docker compose up --build                         # or containerised
```
Endpoints: `GET /health /state /audit /scenarios`; `POST /step /scenario/{name} /approve/{id} /reject/{id} /kill /resume`.
Mutating endpoints need header `X-API-Key`; they are **disabled** if `CEDARLIGHT_API_KEY` is unset. `/resume` also needs the admin token.

```bash
curl -s -X POST localhost:8000/scenario/normal -H "X-API-Key: $CEDARLIGHT_API_KEY"
```

## Safety model (summary)
- Tier 0 read-only, tier 1 low-impact (reports, escalations), tier 2 needs human approval, tier 3 (`switch_breaker`, `shed_load`, `dispatch_crew`) is **not implemented at all**.
- Free text is untrusted: it can become a low-confidence `ToolRequest` fact but never authorises anything.
- Model confidence never upgrades an entailment status. Every decision is logged to a redacted JSONL audit trail. Kill switch halts the loop and cancels pending approvals; only the admin token resets it.
- See `docs/monitoring_and_rollback.md` and `docs/responsible_ai.md`.

## Status
Agent core, reasoning modules and tests are complete and passing. The FastAPI layer is covered by tests that run when `fastapi`/`httpx`
are installed (CI installs them). The Dockerfile and Streamlit UI are written but have not been run-tested in the authoring environment; CI builds
the image and smoke-tests `/health`.
