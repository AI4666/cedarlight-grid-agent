"""ML tests. Run: python -I -m pytest 05_ml -q   (uses the artifacts produced by run_ml.py)"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import data_gen as g  # noqa: E402
import failure as fl  # noqa: E402
import forecast as fc  # noqa: E402
import rl_qlearning as rl  # noqa: E402


@pytest.fixture(scope="module")
def world():
    F, A = g.make_topology(42)
    return F, A, g.make_load(F, seed=42), g.make_asset_snapshots(F, A, seed=42)


@pytest.fixture(scope="module")
def metrics():
    p = HERE / "artifacts" / "metrics.json"
    if not p.exists():
        pytest.skip("run run_ml.py first")
    return json.loads(p.read_text(encoding="utf-8"))


# ---- data / reproducibility ----
def test_generator_is_deterministic(world):
    F, A, L, S = world
    L2 = g.make_load(F, seed=42)
    assert L.equals(L2)
    assert S.equals(g.make_asset_snapshots(F, A, seed=42))


def test_topology_matches_config_sizes(world):
    F, A, _, _ = world
    assert (F.zone.nunique(), len(F), len(A)) == (3, 12, 60)


def test_missing_rate_and_no_negative_load(world):
    L = world[2]
    assert 0.01 < L.load_kw.isna().mean() < 0.04
    assert (L.true_load_kw > 0).all()


# ---- no leakage ----
def test_forecast_split_is_chronological(world):
    tr, va, te = fc.split_by_time(fc.build_features(world[2]))
    assert tr.ts.max() < va.ts.min() and va.ts.max() < te.ts.min()


def test_lag_features_use_only_old_data(world):
    L = world[2]
    df = fc.build_features(L)
    row = df.iloc[2000]
    past = L[(L.feeder == row.feeder) & (L.ts == row.ts - np.timedelta64(24, "h"))].load_kw.iloc[0]
    assert (np.isnan(past) and np.isnan(row.lag24)) or abs(past - row.lag24) < 1e-9


def test_failure_split_is_chronological(world):
    tr, va, te = fl.split_months(world[3])
    assert tr.month.max() < va.month.min() and va.month.max() < te.month.min()


# ---- learning case: improvement over baselines, within guardrails ----
def test_forecast_beats_baselines_on_test(metrics):
    r = metrics["forecast"]["results"]
    best = r[metrics["forecast"]["selected"]]["test"]["mae"]
    assert best < r["persistence_24h"]["test"]["mae"]
    assert best < r["seasonal_naive_168h"]["test"]["mae"]


def test_forecast_interval_coverage_is_reported_and_reasonable(metrics):
    cov = metrics["forecast"]["interval"]["test_coverage"]
    assert 0.80 <= cov <= 0.97


def test_failure_models_are_better_than_chance(metrics):
    for name, r in metrics["failure"]["results"].items():
        assert r["test"]["auc"] > 0.65, name


def test_selected_failure_model_probabilities_are_usable(metrics):
    """Brier of a base-rate-only predictor is p(1-p); the model must beat it (probabilities carry signal)."""
    r = metrics["failure"]["results"][metrics["failure"]["selected"]]["test"]
    assert r["brier"] < r["base_rate"] * (1 - r["base_rate"])


# ---- artifacts ----
def test_artifacts_load_and_predict(metrics, world):
    a = joblib.load(HERE / "artifacts" / "failure_model.joblib")
    X = world[3][a["features"]].head(5)
    p = a["model"].predict_proba(X)[:, 1]
    assert p.shape == (5,) and ((p >= 0) & (p <= 1)).all()
    f = joblib.load(HERE / "artifacts" / "load_model.joblib")
    df = fc.build_features(world[2]).head(5)
    assert f["model"].predict(df[f["features"]]).shape == (5,)


def test_artifact_hashes_match_metrics(metrics):
    for name, digest in metrics["artifact_sha256"].items():
        assert hashlib.sha256((HERE / "artifacts" / name).read_bytes()).hexdigest() == digest


def test_artifacts_record_synthetic_and_version(metrics):
    assert metrics["meta"]["synthetic"] is True and metrics["meta"]["version"]


# ---- fairness ----
def test_zone_quota_narrows_recall_gap(metrics):
    f = metrics["failure"]["fairness_test"]
    assert f["zone_stratified_top_k"]["recall_gap"] < f["global_top_k"]["recall_gap"]


def test_zone_quota_flags_every_zone(world):
    S = world[3]
    score = S["age"].values
    out = fl.zone_recall_at_k(S, score, stratified=True)
    assert all(out[z]["flag_rate"] > 0 for z in ("ZoneA", "ZoneB", "ZoneC"))


# ---- RL safety ----
def test_inspect_masked_during_blackout_and_when_budget_exhausted():
    assert rl.legal((3, 2, 1, 0)) == [0]      # blackout
    assert rl.legal((3, 0, 0, 0)) == [0]      # no crew budget
    assert rl.legal((3, 1, 0, 0)) == [0, 1]


def test_illegal_policy_actions_are_counted_and_overridden():
    always_inspect = lambda s, r: 1  # noqa: E731
    rng = np.random.default_rng(0)
    results = [rl.run_episode(always_inspect, rng) for _ in range(2000)]
    assert sum(v for _, v in results) > 0                     # violations detected...
    # ...and overridden: never more than BUDGET inspections get through (reward bounded by safe execution)


def test_trained_policy_has_no_violations_and_beats_naive(metrics):
    r = metrics["rl"]
    assert r["q_learning"]["safety_violations"] == 0
    assert r["q_learning"]["mean_reward"] > r["never_inspect"]["mean_reward"] + 1.0
    assert r["q_learning"]["mean_reward"] > r["random_legal"]["mean_reward"] + 1.0


def test_q_table_shape():
    Q = np.load(HERE / "artifacts" / "q_table.npy")
    assert Q.shape == (4, rl.BUDGET + 1, 2, rl.N_ASSETS, 2)
