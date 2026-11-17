"""Core simulation algorithms for the CedarLight agent (course topics 2-5).

  A. Inspection routing  : BFS / DFS / UCS / IDS / greedy / A* / local search (2-opt, annealing)
  B. Crew scheduling CSP : backtracking, +MRV, +LCV, +forward checking, +AC-3, min-conflicts
  C. Adversarial demand  : minimax, alpha-beta, expectimax on a reserve-vs-spike game

Everything is deterministic given the seed.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from collections import deque

import numpy as np

# =====================================================================
# A. Inspection routing (state-space search)
# =====================================================================


class RouteProblem:
    """Crew leaves depot (node 0), must inspect every node in `required`, and return to the depot.
    State = (location, frozenset(remaining)).  Moving along an edge costs its length; arriving at a node that
    still needs inspection adds `inspect_cost` and removes it from `remaining`."""

    def __init__(self, seed: int = 0, n_nodes: int = 12, k_required: int = 4, inspect_cost: float = 1.0):
        rng = np.random.default_rng(seed)
        self.n, self.inspect_cost = n_nodes, inspect_cost
        self.pts = rng.uniform(0, 10, (n_nodes, 2))
        d = np.linalg.norm(self.pts[:, None] - self.pts[None], axis=2)
        edges = set()
        for i in range(n_nodes):
            for j in np.argsort(d[i])[1:4]:                 # 3 nearest neighbours
                edges.add((min(i, int(j)), max(i, int(j))))
        parent = list(range(n_nodes))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in edges:
            parent[find(a)] = find(b)
        while len({find(i) for i in range(n_nodes)}) > 1:   # stitch components with the closest cross pair
            best = min(((d[i, j], i, j) for i in range(n_nodes) for j in range(n_nodes) if find(i) != find(j)))
            _, i, j = best
            edges.add((min(i, j), max(i, j)))
            parent[find(i)] = find(j)
        self.adj = {i: [] for i in range(n_nodes)}
        for a, b in edges:
            self.adj[a].append((b, float(d[a, b])))
            self.adj[b].append((a, float(d[a, b])))
        self.required = frozenset(int(x) for x in rng.choice(np.arange(1, n_nodes), k_required, replace=False))
        # all-pairs shortest paths (Floyd-Warshall) for heuristics / local search
        D = np.full((n_nodes, n_nodes), np.inf)
        np.fill_diagonal(D, 0)
        for a in self.adj:
            for b, w in self.adj[a]:
                D[a, b] = w
        for k in range(n_nodes):
            D = np.minimum(D, D[:, [k]] + D[[k], :])
        self.D = D
        self.start = (0, self.required)

    def is_goal(self, s):
        return not s[1] and s[0] == 0

    def successors(self, s):
        loc, rem = s
        for nb, w in self.adj[loc]:
            if nb in rem:
                yield (nb, rem - {nb}), w + self.inspect_cost
            else:
                yield (nb, rem), w

    def h(self, s):
        """Admissible: must still reach the farthest remaining site and get home, and pay each inspection."""
        loc, rem = s
        if not rem:
            return float(self.D[loc, 0])
        return max(self.D[loc, r] + self.D[r, 0] for r in rem) + self.inspect_cost * len(rem)


def _result(found, cost, expanded, moves=None, t0=0.0, path=None):
    return {"found": found, "cost": cost, "expanded": expanded, "moves": moves, "path": path,
            "ms": (time.perf_counter() - t0) * 1000}


def bfs(P: RouteProblem, limit=200000):
    t0, frontier, seen, exp = time.perf_counter(), deque([(P.start, 0.0, 0)]), {P.start}, 0
    while frontier:
        s, g, m = frontier.popleft()
        exp += 1
        if P.is_goal(s):
            return _result(True, g, exp, m, t0)
        for s2, c in P.successors(s):
            if s2 not in seen:
                seen.add(s2)
                frontier.append((s2, g + c, m + 1))
        if exp > limit:
            break
    return _result(False, math.inf, exp, None, t0)


def dfs(P: RouteProblem, limit=200000):
    t0, stack, seen, exp = time.perf_counter(), [(P.start, 0.0, 0)], set(), 0
    while stack:
        s, g, m = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        exp += 1
        if P.is_goal(s):
            return _result(True, g, exp, m, t0)
        for s2, c in P.successors(s):
            if s2 not in seen:
                stack.append((s2, g + c, m + 1))
        if exp > limit:
            break
    return _result(False, math.inf, exp, None, t0)


def ucs(P: RouteProblem, limit=200000):
    return _best_first(P, lambda g, s: g, limit)


def astar(P: RouteProblem, limit=200000, h=None):
    h = h or P.h
    return _best_first(P, lambda g, s: g + h(s), limit)


def greedy(P: RouteProblem, limit=200000):
    return _best_first(P, lambda g, s: P.h(s), limit)


def _best_first(P, f, limit):
    t0, tie = time.perf_counter(), itertools.count()
    pq = [(f(0.0, P.start), next(tie), 0.0, P.start, 0)]
    best_g, parent, exp = {P.start: 0.0}, {P.start: None}, 0
    while pq:
        _, _, g, s, m = heapq.heappop(pq)
        if g > best_g.get(s, math.inf):
            continue
        exp += 1
        if P.is_goal(s):
            chain, cur = [], s
            while cur is not None:
                chain.append(cur[0])
                cur = parent[cur]
            return _result(True, g, exp, m, t0, path=chain[::-1])   # node sequence from the depot back to the depot
        for s2, c in P.successors(s):
            g2 = g + c
            if g2 < best_g.get(s2, math.inf):
                best_g[s2] = g2
                parent[s2] = s
                heapq.heappush(pq, (f(g2, s2), next(tie), g2, s2, m + 1))
        if exp > limit:
            break
    return _result(False, math.inf, exp, None, t0)


def ids(P: RouteProblem, max_depth=40):
    """Iterative deepening with a depth-aware transposition table (a state is re-expanded only if reached
    at a shallower depth). Finds the solution with the FEWEST moves, like BFS, with low memory."""
    t0, exp = time.perf_counter(), 0
    for limit in range(max_depth + 1):
        seen: dict = {}
        found = []

        def dls(s, g, depth):
            nonlocal exp
            if P.is_goal(s):
                found.append((g, depth))
                return True
            if depth == limit:
                return False
            if seen.get(s, 10 ** 9) <= depth:
                return False
            seen[s] = depth
            exp += 1
            return any(dls(s2, g + c, depth + 1) for s2, c in P.successors(s))

        if dls(P.start, 0.0, 0):
            return _result(True, found[0][0], exp, found[0][1], t0)
    return _result(False, math.inf, exp, None, t0)


def tour_cost(P: RouteProblem, order):
    path = [0, *order, 0]
    return float(sum(P.D[a, b] for a, b in zip(path, path[1:])) + P.inspect_cost * len(order))


def brute_force_optimum(P: RouteProblem):
    return min(tour_cost(P, o) for o in itertools.permutations(sorted(P.required)))


def two_opt(P: RouteProblem, seed=0, restarts=5):
    """Hill-climbing local search on the visiting order (2-opt moves) with random restarts."""
    rng, evals, best = np.random.default_rng(seed), 0, math.inf
    t0 = time.perf_counter()
    for _ in range(restarts):
        order = list(rng.permutation(sorted(P.required)))
        cur, evals = tour_cost(P, order), evals + 1
        improved = True
        while improved:
            improved = False
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                    c, evals = tour_cost(P, cand), evals + 1
                    if c < cur - 1e-12:
                        order, cur, improved = cand, c, True
        best = min(best, cur)
    return {"cost": best, "evals": evals, "ms": (time.perf_counter() - t0) * 1000}


def anneal(P: RouteProblem, seed=0, steps=400, t_start=3.0, t_end=0.05):
    rng, t0 = np.random.default_rng(seed), time.perf_counter()
    order = list(rng.permutation(sorted(P.required)))
    cur = best = tour_cost(P, order)
    for k in range(steps):
        T = t_start * (t_end / t_start) ** (k / steps)
        i, j = sorted(rng.choice(len(order), 2, replace=False))
        cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
        c = tour_cost(P, cand)
        if c < cur or rng.random() < math.exp((cur - c) / T):
            order, cur = cand, c
            best = min(best, cur)
    return {"cost": best, "evals": steps + 1, "ms": (time.perf_counter() - t0) * 1000}


def check_heuristic(P: RouteProblem):
    """Exhaustively verify admissibility (h <= true cost-to-go) and consistency over the reachable state space."""
    states, stack = {P.start}, [P.start]
    while stack:
        s = stack.pop()
        for s2, _ in P.successors(s):
            if s2 not in states:
                states.add(s2)
                stack.append(s2)
    # true cost-to-go by Dijkstra on the reversed graph
    rev = {s: [] for s in states}
    for s in states:
        for s2, c in P.successors(s):
            rev[s2].append((s, c))
    ctg = {s: math.inf for s in states}
    pq = []
    for s in states:
        if P.is_goal(s):
            ctg[s] = 0.0
            pq.append((0.0, id(s), s))
    heapq.heapify(pq)
    while pq:
        d, _, s = heapq.heappop(pq)
        if d > ctg[s]:
            continue
        for p, c in rev[s]:
            if d + c < ctg[p]:
                ctg[p] = d + c
                heapq.heappush(pq, (ctg[p], id(p), p))
    admissible = all(P.h(s) <= ctg[s] + 1e-9 for s in states if ctg[s] < math.inf)
    consistent = all(P.h(s) <= c + P.h(s2) + 1e-9 for s in states for s2, c in P.successors(s))
    return {"states": len(states), "admissible": admissible, "consistent": consistent}


# =====================================================================
# B. Crew scheduling as a CSP
# =====================================================================
SKILLS = ["A", "B", "C"]
CAP_PER_DAY = 2          # max jobs per crew per day (a day = 2 slots)


class SchedProblem:
    """Variables = jobs. Domain = (crew, slot). Constraints:
       unary  : crew must have the skill; no outdoor work in blackout (storm) slots
       binary : two jobs cannot share a (crew, slot); jobs on the same feeder need >= 2 slots between them
       global : each crew does at most CAP_PER_DAY jobs per day."""

    def __init__(self, seed=0, n_jobs=10, n_crews=3, n_slots=6, n_blackout=2):
        rng = np.random.default_rng(seed)
        self.n_jobs, self.n_crews, self.n_slots = n_jobs, n_crews, n_slots
        self.feeder = {j: int(rng.integers(0, 4)) for j in range(n_jobs)}
        self.skill = {j: SKILLS[int(rng.integers(0, 3))] for j in range(n_jobs)}
        self.crew_skills = [set(rng.choice(SKILLS, 2, replace=False)) for _ in range(n_crews)]
        self.blackout = set(int(x) for x in rng.choice(n_slots, n_blackout, replace=False))
        self.domain = {j: [(c, s) for c in range(n_crews) for s in range(n_slots)
                           if self.skill[j] in self.crew_skills[c] and s not in self.blackout]
                       for j in range(n_jobs)}
        self.vars = list(range(n_jobs))

    def binary_ok(self, j1, v1, j2, v2):
        if v1 == v2:
            return False
        if self.feeder[j1] == self.feeder[j2] and abs(v1[1] - v2[1]) < 2:
            return False
        return True

    def cap_ok(self, assign, var, val):
        c, s = val
        n = sum(1 for j, (c2, s2) in assign.items() if j != var and c2 == c and s2 // 2 == s // 2)
        return n < CAP_PER_DAY

    def valid(self, assign):
        if set(assign) != set(self.vars):
            return False
        for j, v in assign.items():
            if v not in self.domain[j] or not self.cap_ok({k: x for k, x in assign.items() if k != j}, j, v):
                return False
        return all(self.binary_ok(a, assign[a], b, assign[b]) for a, b in itertools.combinations(self.vars, 2))


def ac3(P: SchedProblem, domains):
    """Arc consistency. Returns False if some domain is wiped out (problem unsatisfiable)."""
    queue = deque((a, b) for a in P.vars for b in P.vars if a != b)
    while queue:
        a, b = queue.popleft()
        removed = [va for va in domains[a] if not any(P.binary_ok(a, va, b, vb) for vb in domains[b])]
        if removed:
            domains[a] = [v for v in domains[a] if v not in removed]
            if not domains[a]:
                return False
            queue.extend((c, a) for c in P.vars if c not in (a, b))
    return True


def solve_csp(P: SchedProblem, mrv=False, lcv=False, fc=False, use_ac3=False, node_limit=200000):
    t0 = time.perf_counter()
    domains = {v: list(P.domain[v]) for v in P.vars}
    stats = {"nodes": 0, "backtracks": 0, "timeout": False}
    if use_ac3 and not ac3(P, domains):
        return {"solved": False, **stats, "assign": None, "ms": (time.perf_counter() - t0) * 1000}

    def legal(var, assign, doms):
        return [v for v in doms[var] if all(P.binary_ok(var, v, o, assign[o]) for o in assign) and P.cap_ok(assign, var, v)]

    def select(assign, doms):
        free = [v for v in P.vars if v not in assign]
        if not mrv:
            return free[0]
        return min(free, key=lambda v: (len(doms[v]), v))   # smallest *current* domain (pruned if fc is on)

    def order(var, assign, doms):
        vals = legal(var, assign, doms)
        if not lcv:
            return vals
        free = [o for o in P.vars if o not in assign and o != var]
        return sorted(vals, key=lambda v: sum(1 for o in free for vo in doms[o] if not P.binary_ok(var, v, o, vo)))

    def bt(assign, doms):
        if len(assign) == len(P.vars):
            return dict(assign)
        var = select(assign, doms)
        for val in order(var, assign, doms):
            stats["nodes"] += 1
            if stats["nodes"] > node_limit:
                stats["timeout"] = True
                return None
            assign[var] = val
            new = doms
            if fc:   # forward checking: prune neighbours' domains, fail early on wipe-out
                new = {k: list(d) for k, d in doms.items()}
                wiped = False
                for o in P.vars:
                    if o not in assign:
                        new[o] = [x for x in new[o] if P.binary_ok(var, val, o, x)]
                        if not new[o]:
                            wiped = True
                            break
                if wiped:
                    del assign[var]
                    stats["backtracks"] += 1
                    continue
            r = bt(assign, new)
            if r is not None or stats["timeout"]:
                return r
            del assign[var]
            stats["backtracks"] += 1
        return None

    sol = bt({}, domains)
    return {"solved": sol is not None, **stats, "assign": sol, "ms": (time.perf_counter() - t0) * 1000}


def min_conflicts(P: SchedProblem, seed=0, max_steps=3000):
    rng, t0 = np.random.default_rng(seed), time.perf_counter()
    if any(not P.domain[v] for v in P.vars):
        return {"solved": False, "steps": 0, "assign": None, "ms": 0.0}
    assign = {v: P.domain[v][int(rng.integers(len(P.domain[v])))] for v in P.vars}

    def conflicts(var, val):
        n = sum(1 for o in P.vars if o != var and not P.binary_ok(var, val, o, assign[o]))
        return n + (0 if P.cap_ok(assign, var, val) else 1)

    for step in range(max_steps):
        bad = [v for v in P.vars if conflicts(v, assign[v]) > 0]
        if not bad:
            return {"solved": P.valid(assign), "steps": step, "assign": assign, "ms": (time.perf_counter() - t0) * 1000}
        var = bad[int(rng.integers(len(bad)))]
        scored = [(conflicts(var, v), v) for v in P.domain[var]]
        m = min(c for c, _ in scored)
        assign[var] = [v for c, v in scored if c == m][int(rng.integers(sum(1 for c, _ in scored if c == m)))]
    return {"solved": False, "steps": max_steps, "assign": assign, "ms": (time.perf_counter() - t0) * 1000}


# =====================================================================
# C. Adversarial demand-spike game
# =====================================================================
RESERVE_COST = [0.0, 2.0, 5.0]       # operator commits a reserve level each hour
RESERVE_CAP = [0.0, 4.0, 8.0]        # MW covered by that reserve level
SPIKE = [0.0, 3.0, 7.0]              # adversary / nature chooses the demand spike
SPIKE_P = [0.70, 0.25, 0.05]         # probability model for nature (expectimax)
PENALTY = 6.0                        # cost per MW of unserved demand
FATIGUE_CAP = 5.0                    # high reserve used two hours in a row only delivers this much


def step_cost(last, a, s):
    cap = RESERVE_CAP[a] if not (a == 2 and last == 2) else FATIGUE_CAP
    return RESERVE_COST[a] + PENALTY * max(0.0, SPIKE[s] - cap)


def minimax(t, last, H, counter):
    counter[0] += 1
    if t == H:
        return 0.0, None
    best, best_a = math.inf, None
    for a in range(3):
        worst = -math.inf
        for s in range(3):
            counter[0] += 1
            v = step_cost(last, a, s) + minimax(t + 1, a, H, counter)[0]
            worst = max(worst, v)
        if worst < best:
            best, best_a = worst, a
    return best, best_a


def alphabeta(t, last, H, counter):
    """Alpha-beta over the same tree as `minimax`. Bounds apply to the TOTAL path cost, so the accumulated
    cost is carried down explicitly. Operator = MIN node, adversary = MAX node (worst spikes tried first)."""
    best = {"a": None}

    def min_node(t, last, acc, alpha, beta, root=False):
        counter[0] += 1
        if t == H:
            return acc
        v = math.inf
        for a in range(3):
            w = max_node(t, last, a, acc, alpha, min(beta, v))
            if w < v:
                v = w
                if root:
                    best["a"] = a
            if v <= alpha:
                return v
        return v

    def max_node(t, last, a, acc, alpha, beta):
        counter[0] += 1
        v = -math.inf
        for s in (2, 1, 0):
            v = max(v, min_node(t + 1, a, acc + step_cost(last, a, s), max(alpha, v), beta))
            if v >= beta:
                return v
        return v

    val = min_node(t, last, 0.0, -math.inf, math.inf, root=True)
    return val, best["a"]


def expectimax(t, last, H, counter):
    counter[0] += 1
    if t == H:
        return 0.0, None
    best, best_a = math.inf, None
    for a in range(3):
        counter[0] += 1
        v = sum(SPIKE_P[s] * (step_cost(last, a, s) + expectimax(t + 1, a, H, counter)[0]) for s in range(3))
        if v < best:
            best, best_a = v, a
    return best, best_a


def play(policy, H, spikes):
    """Run a receding-horizon policy against a fixed spike sequence; return total cost."""
    last, total = -1, 0.0
    for t in range(H):
        a = policy(t, last, H - t)
        total += step_cost(last, a, spikes[t])
        last = a
    return total


def pol_minimax(t, last, h):
    return minimax(0, last, h, [0])[1]


def pol_expectimax(t, last, h):
    return expectimax(0, last, h, [0])[1]


def pol_never_reserve(t, last, h):
    return 0


def evaluate_game(H=4, n_random=2000, seed=0):
    rng = np.random.default_rng(seed)
    seqs = rng.choice(3, size=(n_random, H), p=SPIKE_P)
    out = {}
    for name, pol in (("never_reserve", pol_never_reserve), ("minimax", pol_minimax), ("expectimax", pol_expectimax)):
        costs = [play(pol, H, sq) for sq in seqs]
        worst = max(play(pol, H, sq) for sq in itertools.product(range(3), repeat=H))
        out[name] = {"expected_cost": float(np.mean(costs)), "worst_case_cost": float(worst)}
    return out
