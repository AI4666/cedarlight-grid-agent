"""Temporal reasoning: a Kalman filter for load tracking/forecasting and an HMM for asset health.

Both are deterministic given inputs (the caller supplies any random seed for simulation).
"""
from __future__ import annotations

import numpy as np


# =====================================================================
# Kalman filter: local linear trend  state = [level, trend]
# =====================================================================
class LoadKalman:
    """Tracks load (kW) with a constant-velocity model. Missing observations (None/NaN) skip the update,
    so uncertainty grows -- the basis for calibrated abstention under missing data."""

    def __init__(self, level0: float, q_level: float = 4.0, q_trend: float = 0.05,
                 r_obs: float = 25.0, p0: float = 100.0):
        self.x = np.array([level0, 0.0], dtype=float)
        self.P = np.eye(2) * p0
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.diag([q_level, q_trend])
        self.R = np.array([[r_obs]])

    def step(self, z: float | None) -> tuple[float, float]:
        # predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # update (skipped when the observation is missing)
        if z is not None and not np.isnan(z):
            y = np.array([z]) - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + (K @ y)
            self.P = (np.eye(2) - K @ self.H) @ self.P
        return float(self.x[0]), float(self.P[0, 0])

    def forecast(self, horizon: int, z90: float = 1.645) -> list[dict]:
        """k-step-ahead mean and ~90% interval from the current belief (does not mutate the filter)."""
        x, P, out = self.x.copy(), self.P.copy(), []
        for k in range(1, horizon + 1):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            var = float(P[0, 0] + self.R[0, 0])  # predictive variance of a new observation
            sd = var ** 0.5
            out.append({"k": k, "mean": float(x[0]), "lo": float(x[0] - z90 * sd), "hi": float(x[0] + z90 * sd)})
        return out


# =====================================================================
# HMM: hidden asset health  {Good, Degraded, Failing}
#      observed condition reading {Low, Mid, High}  (High reading = worse)
# =====================================================================
STATES = ["Good", "Degraded", "Failing"]
OBS = ["Low", "Mid", "High"]
PI = np.array([0.80, 0.18, 0.02])
A = np.array([[0.95, 0.04, 0.01],     # transition (per inspection interval); no self-repair
              [0.00, 0.90, 0.10],
              [0.00, 0.00, 1.00]])
B = np.array([[0.80, 0.18, 0.02],     # emission P(obs | state)
              [0.25, 0.55, 0.20],
              [0.05, 0.25, 0.70]])


def hmm_filter(obs: list[int | None]) -> np.ndarray:
    """Forward filtering. obs[t] is an index into OBS or None (missing). Returns belief per step (T x 3)."""
    belief = PI.copy()
    out = []
    for t, o in enumerate(obs):
        if t > 0:
            belief = belief @ A
        if o is not None:
            belief = belief * B[:, o]
        belief = belief / belief.sum()
        out.append(belief.copy())
    return np.array(out)


def hmm_viterbi(obs: list[int]) -> list[int]:
    """Most likely hidden state sequence (all observations present)."""
    T = len(obs)
    with np.errstate(divide="ignore"):
        logA, logB, logPI = np.log(A), np.log(B), np.log(PI)
    delta = np.zeros((T, 3))
    psi = np.zeros((T, 3), dtype=int)
    delta[0] = logPI + logB[:, obs[0]]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + logA
        psi[t] = scores.argmax(axis=0)
        delta[t] = scores.max(axis=0) + logB[:, obs[t]]
    path = [int(delta[-1].argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(psi[t][path[-1]]))
    return path[::-1]


def hmm_predict(belief: np.ndarray, steps: int) -> np.ndarray:
    """Propagate a belief forward `steps` intervals with no new observations."""
    for _ in range(steps):
        belief = belief @ A
    return belief
