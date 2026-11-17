# Monitoring and rollback plan

## What to watch
| Signal | Source | Alert when |
|---|---|---|
| Service health | `GET /health`, Docker HEALTHCHECK | non-200 for 3 checks |
| Kill switch | `/health`, `/state` | ever active (page the on-call) |
| Escalation rate | audit events, stage `action`, tool `escalate_to_human` | > 3x the weekly baseline |
| Conflict rate | `/state` → `knowledge.conflicts` | any conflict that persists > 1 cycle |
| Policy blocks | audit stage `policy` containing "blocked" | spike, or any block of a tier-3 tool (possible attack) |
| Rejected inputs | `rejected` in `/step` responses | sustained rise (sensor or feed fault) |
| Model health | `model.source` in `/state` | `mock` in production (artifact failed hash check, fallback active) |
| Forecast error | `05_ml/artifacts/metrics.json` vs live residuals | MAE worse than the persistence baseline |
| Approval backlog | `pending_approvals` | any item older than one shift |

Audit logs are append-only JSONL with secrets redacted; ship them to your log store and retain per policy.

## Rollback
1. **Stop the agent acting:** `POST /kill` (any key holder). This halts the loop and cancels pending approvals.
2. **Roll back the release:** redeploy the previous image tag (`docker compose` with the prior tag, or `git revert` and rebuild). Each audit event carries a `config_hash` and `seed`, so a run can be reproduced against the old config.
3. **Roll back a model:** the loader only accepts artifacts whose SHA-256 matches `05_ml/artifacts/metrics.json`; restore the previous artifact and metrics pair, or set `feature_flags.use_ml_risk_second_opinion: false`. Decisions do not depend on the model (it is a second opinion only).
4. **Resume:** `POST /resume` with the admin token after the cause is understood. Pending items were cancelled, so re-run the cycle to regenerate proposals.
5. **Post-incident:** keep the audit file, record the cause, add a regression test.

## Operating rules
- Nothing tier-2 runs without a named human approver. Approvals are re-validated against current knowledge, so a stale approval is blocked.
- Rotate `CEDARLIGHT_API_KEY` and `CEDARLIGHT_ADMIN_TOKEN` by changing env values and restarting; no code change.
