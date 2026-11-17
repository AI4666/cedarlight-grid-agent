"""Tests for Bayesian, temporal and decision modules.  Run: python -I -m pytest 04_uncertainty -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bayes_net as bn  # noqa: E402
import decision as dec  # noqa: E402
import temporal as tp  # noqa: E402


# ---------------- Bayesian network ----------------
def test_cpts_valid_and_joint_normalised():
    bn.validate()


def test_hand_checked_marginal():
    # P(Overload) = P(H)*0.5 + P(~H)*0.15 = 0.2*0.5 + 0.8*0.15 = 0.22
    assert abs(bn.query("Overload") - 0.22) < 1e-12


def test_bayes_rule_posterior_overload_given_reading():
    # P(O|R) = 0.9*0.22 / (0.9*0.22 + 0.1*0.78) = 0.198 / 0.276
    assert abs(bn.query("Overload", {"ReadingHigh": True}) - 0.198 / 0.276) < 1e-12


def test_evidence_moves_posterior_in_right_direction():
    base = bn.query("Fail")
    assert bn.query("Fail", {"ReadingHigh": True}) > base
    assert bn.query("Fail", {"ReadingHigh": False}) < base
    assert bn.query("Fail", {"ReadingHigh": True, "Heat": True, "Poor": True}) > \
           bn.query("Fail", {"ReadingHigh": True, "Heat": True})


def test_conditional_independence_structure():
    # Fail is independent of Old and Heat given its parents Poor and Overload.
    a = bn.query("Fail", {"Poor": True, "Overload": True})
    b = bn.query("Fail", {"Poor": True, "Overload": True, "Old": False, "Heat": False})
    assert abs(a - b) < 1e-12


def test_unknown_variable_rejected():
    import pytest
    with pytest.raises(KeyError):
        bn.query("Fail", {"Nonsense": True})


# ---------------- Kalman filter ----------------
def _ramp(n, slope=2.0, base=100.0, sd=5.0, seed=0):
    rng = np.random.default_rng(seed)
    truth = base + slope * np.arange(n)
    return truth, truth + rng.normal(0, sd, n)


def test_kalman_tracks_better_than_raw_observations():
    truth, obs = _ramp(60)
    kf = tp.LoadKalman(level0=obs[0])
    est = np.array([kf.step(z)[0] for z in obs])
    rmse_filter = np.sqrt(np.mean((est[10:] - truth[10:]) ** 2))
    rmse_raw = np.sqrt(np.mean((obs[10:] - truth[10:]) ** 2))
    assert rmse_filter < rmse_raw


def test_kalman_forecast_interval_coverage():
    hits = total = 0
    for seed in range(40):
        truth, obs = _ramp(70, seed=seed)
        kf = tp.LoadKalman(level0=obs[0])
        for z in obs[:60]:
            kf.step(z)
        for f, y in zip(kf.forecast(5), obs[60:65]):  # compare with noisy future observations
            hits += f["lo"] <= y <= f["hi"]
            total += 1
    assert hits / total >= 0.80  # nominal 90%; allow slack for model mismatch


def test_missing_data_widens_uncertainty():
    _, obs = _ramp(40)
    kf = tp.LoadKalman(level0=obs[0])
    for z in obs:
        _, var_before = kf.step(z)
    for _ in range(6):
        _, var_after = kf.step(None)
    assert var_after > var_before


def test_forecast_does_not_mutate_filter():
    kf = tp.LoadKalman(level0=100.0)
    kf.step(101.0)
    x0, P0 = kf.x.copy(), kf.P.copy()
    kf.forecast(24)
    assert np.allclose(kf.x, x0) and np.allclose(kf.P, P0)


# ---------------- HMM ----------------
def _simulate_hmm(T, seed):
    rng = np.random.default_rng(seed)
    s = rng.choice(3, p=tp.PI)
    states, obs = [], []
    for _ in range(T):
        states.append(int(s))
        obs.append(int(rng.choice(3, p=tp.B[s])))
        s = rng.choice(3, p=tp.A[s])
    return states, obs


def test_hmm_filter_is_a_distribution():
    _, obs = _simulate_hmm(30, 1)
    b = tp.hmm_filter(obs)
    assert np.allclose(b.sum(axis=1), 1.0) and (b >= 0).all()


def test_hmm_high_readings_raise_failing_belief():
    low = tp.hmm_filter([0, 0, 0, 0])[-1]
    high = tp.hmm_filter([2, 2, 2, 2])[-1]
    assert high[2] > low[2] and high[0] < low[0]


def test_hmm_viterbi_beats_chance():
    accs = []
    for seed in range(30):
        states, obs = _simulate_hmm(40, seed)
        accs.append(np.mean(np.array(tp.hmm_viterbi(obs)) == np.array(states)))
    assert np.mean(accs) > 0.60  # chance is ~0.33; Good is the dominant state


def test_hmm_missing_observations_equal_pure_prediction():
    b = tp.hmm_filter([2, 2, None, None, None])
    base = tp.hmm_filter([2, 2])[-1]
    assert np.allclose(b[-1], tp.hmm_predict(base, 3))


def test_failing_state_is_absorbing_in_prediction():
    far = tp.hmm_predict(np.array([0.0, 0.0, 1.0]), 50)
    assert np.allclose(far, [0, 0, 1])


# ---------------- Decision + calibrated abstention ----------------
def test_normal_case_acts_on_strong_evidence():
    r = dec.decide({"ReadingHigh": True, "Heat": True, "Poor": True})
    assert r["status"] == "act" and r["action"] == "rank_inspections"


def test_clearly_low_risk_defers_not_escalates():
    r = dec.decide({"ReadingHigh": False})
    assert r["status"] == "act" and r["action"] == "defer"


def test_uncertain_case_abstains():
    assert dec.decide({})["status"] == "abstain"                       # no evidence
    assert dec.decide({"ReadingHigh": True})["status"] == "abstain"   # noisy sensor only, near break-even


def test_abstention_threshold_is_configurable():
    strict = dec.decide({"ReadingHigh": True, "Heat": True}, {"abstain_below_confidence": 0.9})
    assert strict["status"] == "abstain"
