"""Tests for simulation algorithms.  Run: python -I -m pytest 02_simulation -q"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import sim_algos as S  # noqa: E402

SEEDS = range(10)


# ---------------- A. search ----------------
@pytest.mark.parametrize("seed", range(5))
def test_heuristic_admissible_and_consistent(seed):
    r = S.check_heuristic(S.RouteProblem(seed))
    assert r["admissible"] and r["consistent"]


@pytest.mark.parametrize("seed", SEEDS)
def test_ucs_and_astar_are_optimal(seed):
    P = S.RouteProblem(seed)
    opt = S.brute_force_optimum(P)
    u, a = S.ucs(P), S.astar(P)
    assert abs(u["cost"] - opt) < 1e-9 and abs(a["cost"] - opt) < 1e-9
    assert a["expanded"] <= u["expanded"]


@pytest.mark.parametrize("seed", SEEDS)
def test_bfs_and_ids_minimise_moves(seed):
    P = S.RouteProblem(seed)
    b, i = S.bfs(P), S.ids(P)
    assert b["moves"] == i["moves"]
    assert b["moves"] <= S.ucs(P)["moves"]


@pytest.mark.parametrize("seed", SEEDS)
def test_all_searches_find_valid_solutions_not_better_than_optimal(seed):
    P = S.RouteProblem(seed)
    opt = S.brute_force_optimum(P)
    for f in (S.bfs, S.dfs, S.greedy, S.ids):
        r = f(P)
        assert r["found"] and r["cost"] >= opt - 1e-9


def test_local_search_never_beats_optimum_and_is_deterministic():
    P = S.RouteProblem(3)
    opt = S.brute_force_optimum(P)
    assert S.two_opt(P, seed=1)["cost"] >= opt - 1e-9
    assert S.anneal(P, seed=1)["cost"] >= opt - 1e-9
    assert S.two_opt(P, seed=1)["cost"] == S.two_opt(P, seed=1)["cost"]


def test_graph_is_connected():
    P = S.RouteProblem(11)
    assert np.isfinite(P.D).all()


# ---------------- B. CSP ----------------
CONFIGS = [dict(), dict(mrv=True), dict(mrv=True, fc=True), dict(mrv=True, lcv=True, fc=True),
           dict(mrv=True, lcv=True, fc=True, use_ac3=True)]


@pytest.mark.parametrize("seed", range(15))
def test_complete_solvers_agree_and_solutions_are_valid(seed):
    P = S.SchedProblem(seed, n_jobs=8, n_blackout=1)
    results = [S.solve_csp(P, **kw) for kw in CONFIGS]
    assert not any(r["timeout"] for r in results)
    assert len({r["solved"] for r in results}) == 1
    for r in results:
        if r["solved"]:
            assert P.valid(r["assign"])


def test_instance_family_has_both_sat_and_unsat():
    sat = [S.solve_csp(S.SchedProblem(s, n_jobs=8, n_blackout=1), mrv=True, fc=True)["solved"] for s in range(30)]
    assert 0.2 < sum(sat) / len(sat) < 0.8


def test_mrv_fc_reduces_effort_overall():
    plain = mrvfc = 0
    for s in range(30):
        P = S.SchedProblem(s, n_jobs=8, n_blackout=1)
        plain += S.solve_csp(P)["nodes"]
        mrvfc += S.solve_csp(P, mrv=True, fc=True)["nodes"]
    assert mrvfc < plain


def test_ac3_detects_wipeout_and_prunes():
    P = S.SchedProblem(0, n_jobs=8, n_blackout=1)
    doms = {v: list(P.domain[v]) for v in P.vars}
    before = sum(map(len, doms.values()))
    ok = S.ac3(P, doms)
    assert (not ok) or sum(map(len, doms.values())) <= before
    P2 = S.SchedProblem(1, n_jobs=8, n_blackout=1)
    P2.domain[0] = []           # impossible job -> wipe-out
    doms2 = {v: list(P2.domain[v]) for v in P2.vars}
    doms2[0] = []
    assert S.ac3(P2, doms2) is False


def test_min_conflicts_solutions_are_valid_and_never_claim_unsat_success():
    solved = 0
    for s in range(30):
        P = S.SchedProblem(s, n_jobs=8, n_blackout=1)
        truth = S.solve_csp(P, mrv=True, fc=True)["solved"]
        m = S.min_conflicts(P, seed=s)
        if m["solved"]:
            solved += 1
            assert truth and P.valid(m["assign"])
        if not truth:
            assert not m["solved"]
    assert solved > 0


def test_cap_constraint_enforced():
    P = S.SchedProblem(0, n_jobs=8, n_blackout=1)
    assign = {0: (0, 0), 1: (0, 1)}                  # crew 0 already has 2 jobs on day 0 (slots 0 and 1)
    assert S.CAP_PER_DAY == 2
    assert not P.cap_ok(assign, 2, (0, 0))           # third job same crew, same day -> rejected
    assert not P.cap_ok(assign, 2, (0, 1))
    assert P.cap_ok(assign, 2, (0, 2))               # next day is fine
    assert P.cap_ok(assign, 2, (1, 0))               # other crew is fine


def test_invalid_assignment_rejected():
    P = S.SchedProblem(0, n_jobs=8, n_blackout=1)
    assert not P.valid({})                          # incomplete
    assert not P.valid({j: (0, 0) for j in P.vars})  # everyone in the same cell


# ---------------- C. game ----------------
@pytest.mark.parametrize("H", [2, 3, 4])
def test_alphabeta_matches_minimax_and_prunes(H):
    c1, c2 = [0], [0]
    mm, ab = S.minimax(0, -1, H, c1), S.alphabeta(0, -1, H, c2)
    assert mm[0] == ab[0] and mm[1] == ab[1]
    assert c2[0] < c1[0]


def test_alphabeta_prunes_majority_at_depth():
    c1, c2 = [0], [0]
    S.minimax(0, -1, 5, c1)
    S.alphabeta(0, -1, 5, c2)
    assert c2[0] < 0.5 * c1[0]


def test_expectimax_value_equals_exact_expected_cost_of_its_policy():
    H = 3
    value = S.expectimax(0, -1, H, [0])[0]
    exact = 0.0
    for seq in itertools.product(range(3), repeat=H):
        p = np.prod([S.SPIKE_P[s] for s in seq])
        exact += p * S.play(S.pol_expectimax, H, seq)
    assert abs(exact - value) < 1e-9


def test_minimax_value_equals_worst_case_of_its_policy():
    H = 3
    value = S.minimax(0, -1, H, [0])[0]
    worst = max(S.play(S.pol_minimax, H, seq) for seq in itertools.product(range(3), repeat=H))
    assert abs(worst - value) < 1e-9


def test_risk_tradeoff_between_policies():
    g = S.evaluate_game(H=4, n_random=3000, seed=0)
    assert g["expectimax"]["expected_cost"] < g["minimax"]["expected_cost"] < g["never_reserve"]["expected_cost"]
    assert g["minimax"]["worst_case_cost"] < g["expectimax"]["worst_case_cost"] < g["never_reserve"]["worst_case_cost"]


# ---------------- determinism ----------------
def test_everything_is_seed_deterministic():
    a = (S.ucs(S.RouteProblem(5))["expanded"], S.solve_csp(S.SchedProblem(5, n_jobs=8, n_blackout=1), mrv=True)["nodes"],
         S.evaluate_game(H=3, n_random=200, seed=2))
    b = (S.ucs(S.RouteProblem(5))["expanded"], S.solve_csp(S.SchedProblem(5, n_jobs=8, n_blackout=1), mrv=True)["nodes"],
         S.evaluate_game(H=3, n_random=200, seed=2))
    assert a == b


def test_astar_returns_a_valid_path_with_matching_cost():
    for seed in range(5):
        P = S.RouteProblem(seed)
        r = S.astar(P)
        path = r["path"]
        assert path[0] == 0 and path[-1] == 0 and set(P.required) <= set(path)
        cost = 0.0
        rem = set(P.required)
        for a, b in zip(path, path[1:]):
            cost += dict(P.adj[a])[b] + (P.inspect_cost if b in rem else 0.0)
            rem.discard(b)
        assert abs(cost - r["cost"]) < 1e-9 and not rem
