# CedarLight Grid Load & Maintenance Planning Agent

Fictitious CedarLight Energy Cooperative. Synthetic data only.
Transparent six-stage cycle: percepts -> propositions -> knowledge base -> inference -> entailment -> action.

## Run the tests
    pip install -r requirements.txt
    (cd 05_ml && python -I run_ml.py)   # regenerates model artifacts (~30 s)
    python -I -m pytest 02_simulation 03_logic 04_uncertainty 05_ml tests -q

Status: agent core and tests complete (179 passing). API, UI, Dockerfile, CI and runbook still to come.
