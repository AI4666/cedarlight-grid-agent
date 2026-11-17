"""Tabular Q-learning for a toy inspection-selection MDP, with a hard action mask for safe exploration.

Episode: a crew-day. 8 candidate assets arrive one at a time with a failure-risk bucket (from the risk model).
State  = (risk_bucket 0..3, inspections_left 0..3, blackout 0/1, position 0..7)  # position keeps the MDP Markov
Action = 0 skip, 1 inspect
Reward = -inspect_cost when inspecting (and the failure is reduced to `residual`), else a realised failure costs
         -failure_cost with probability p. All numbers are synthetic.
SAFETY: 'inspect' is MASKED when blackout=1 (storm) or inspections_left=0, in training AND at execution.
"""
from __future__ import annotations

import numpy as np

BUCKET_P = np.array([0.02, 0.06, 0.15, 0.40])   # representative failure probability per bucket
BUCKET_W = np.array([0.40, 0.30, 0.20, 0.10])   # how often each bucket occurs
N_ASSETS, BUDGET = 8, 3
COST_INSPECT, COST_FAIL, RESIDUAL = 1.0, 12.0, 0.15


def legal(state) -> list[int]:
    _, left, blackout, _ = state
    return [0, 1] if (left > 0 and not blackout) else [0]


def new_episode(rng):
    buckets = rng.choice(4, size=N_ASSETS, p=BUCKET_W)
    blackout = rng.random(N_ASSETS) < 0.15
    return buckets, blackout


def step_reward(bucket, action, rng):
    p = BUCKET_P[bucket]
    if action == 1:
        return -COST_INSPECT - (COST_FAIL if rng.random() < p * RESIDUAL else 0.0)
    return -(COST_FAIL if rng.random() < p else 0.0)


def run_episode(policy, rng, learn=None):
    buckets, blackout = new_episode(rng)
    left, total, violations = BUDGET, 0.0, 0
    for i in range(N_ASSETS):
        s = (int(buckets[i]), left, int(blackout[i]), i)
        a = policy(s, rng)
        if a not in legal(s):
            violations += 1
            a = 0
        r = step_reward(buckets[i], a, rng)
        total += r
        nl = left - (1 if a == 1 else 0)
        if learn:
            s2 = (int(buckets[i + 1]), nl, int(blackout[i + 1]), i + 1) if i + 1 < N_ASSETS else None
            learn(s, a, r, s2)
        left = nl
    return total, violations


def train_q(episodes=300000, gamma=1.0, eps0=0.5, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((4, BUDGET + 1, 2, N_ASSETS, 2))
    N = np.zeros_like(Q)                       # visit counts -> decaying step size 1/N^0.7

    def learn(s, a, r, s2):
        target = r if s2 is None else r + gamma * max(Q[s2][x] for x in legal(s2))
        N[s][a] += 1
        Q[s][a] += max(0.002, 1.0 / N[s][a] ** 0.7) * (target - Q[s][a])

    for ep in range(episodes):
        eps = max(0.02, eps0 * (1 - ep / episodes))
        def pol(s, r):
            ls = legal(s)                                    # exploration restricted to legal actions
            return int(r.choice(ls)) if r.random() < eps else max(ls, key=lambda x: Q[s][x])
        run_episode(pol, rng, learn)
    return Q


def evaluate_policies(Q, episodes=20000, seed=123):
    breakeven_bucket = int(np.argmax(BUCKET_P * (1 - RESIDUAL) * COST_FAIL > COST_INSPECT))
    pols = {
        "random_legal": lambda s, r: int(r.choice(legal(s))),
        "never_inspect": lambda s, r: 0,
        "greedy_threshold": lambda s, r: 1 if (s[0] >= breakeven_bucket and 1 in legal(s)) else 0,
        "q_learning": lambda s, r: max(legal(s), key=lambda x: Q[s][x]),
    }
    out = {}
    for name, pol in pols.items():
        rng = np.random.default_rng(seed)             # identical episodes for every policy (paired comparison)
        res = [run_episode(pol, rng) for _ in range(episodes)]
        costs = np.array([c for c, _ in res])
        out[name] = {"mean_reward": float(costs.mean()), "se": float(costs.std() / np.sqrt(episodes)),
                     "safety_violations": int(sum(v for _, v in res))}
    return out
