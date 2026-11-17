# 04_uncertainty — Bayesian & Temporal Reasoning

Files: `bayes_net.py` (exact-inference BN), `temporal.py` (Kalman filter + HMM), `decision.py` (posterior -> action with abstention), `test_uncertainty.py` (19 tests).
All parameters are **synthetic** (CPT_VERSION 0.1.0); replace with values fitted in `05_ml/` once the simulator exists.

## 1. Bayesian network (course topic 8)
```
Old ──► Poor ──┐
               ├──► Fail
Heat ─► Overload ┘
           └────► ReadingHigh  (noisy sensor)
```
| Node | Parents | P(True) |
|---|---|---|
| Old | – | 0.40 |
| Heat | – | 0.20 |
| Poor | Old=T / F | 0.35 / 0.08 |
| Overload | Heat=T / F | 0.50 / 0.15 |
| ReadingHigh | Overload=T / F | 0.90 / 0.10 |
| Fail | (Poor,Overload) = TT / TF / FT / FF | 0.40 / 0.12 / 0.10 / 0.01 |

Inference: exact enumeration over 64 worlds. Verified by hand: P(Overload)=0.22; P(Overload | ReadingHigh)=0.198/0.276≈0.717.

**Example posteriors for P(Fail):** prior 0.058 · ReadingHigh only 0.121 · +Heat 0.144 · +Poor 0.372 · ReadingHigh=False 0.035.

## 2. Temporal reasoning (course topic 9)
- **Kalman filter (local linear trend):** tracks load level and trend; missing readings skip the update so variance grows; `forecast(24)` gives mean and ~90% interval per hour. Tests: beats raw observations, ≥80% interval coverage over 40 seeded runs (nominal 90%, model is slightly mismatched), uncertainty widens under missing data, forecast is side-effect free.
- **HMM (asset health Good/Degraded/Failing, readings Low/Mid/High):** forward filtering (handles missing readings), Viterbi decoding, multi-step prediction. Failing is absorbing (no self-repair). Tests: belief is a distribution, high readings raise P(Failing), Viterbi accuracy >0.60 vs ~0.33 chance, missing observations equal pure prediction.

## 3. Decision and calibrated abstention
Inspecting costs `cost_inspect`; a failure costs `cost_failure` (defaults 1 and 12), so break-even P(Fail) ≈ 0.083.
`confidence = 0.5 + 0.5·min(1, |ln(p/break-even)| / ln 4)`. If confidence < `abstain_below_confidence` (0.65) or there is no evidence, the agent **abstains and escalates** instead of guessing.

| Evidence | P(Fail) | Confidence | Outcome |
|---|---|---|---|
| none | 0.058 | 0.63 | abstain |
| ReadingHigh | 0.121 | 0.63 | abstain (noisy sensor alone) |
| ReadingHigh + Heat | 0.144 | 0.70 | inspect |
| ReadingHigh + Heat + Poor | 0.372 | 1.00 | inspect |
| ReadingHigh=False | 0.035 | 0.82 | defer |

## 4. Boundaries
- Posterior probabilities feed ACTION selection and appear in `InferenceTrace.probability`. They are **never** used to upgrade an entailment status to `entailed`.
- "Calibrated" here means the abstention rule is tied to decision margins; true calibration of fitted models (reliability curves, Brier score) is evaluated in `05_ml/`.
- Known limitations: independent-parameter assumptions, synthetic CPTs, a Gaussian linear Kalman model that ignores daily load seasonality (add seasonal terms or per-hour models as an extension).
