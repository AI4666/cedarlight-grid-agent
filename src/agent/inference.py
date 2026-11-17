"""STAGE 4 - INFERENCE. Logical, probabilistic, temporal, learned and search/CSP reasoning, each producing an
inspectable trace (concise evidence records - never hidden chain-of-thought)."""
from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import _lib  # noqa: F401
from .config import ROOT, resolve
from .knowledge import KnowledgeBase
from .schemas import InferenceMethod, InferenceStep, InferenceTrace
import bayes_net as bn  # noqa: E402
import decision as dec  # noqa: E402
import logic_engine as L  # noqa: E402
import sim_algos as S  # noqa: E402
import temporal as tp  # noqa: E402

LOGIC_CONCLUSIONS = {"InspectUrgent", "AtRisk", "ImpactHigh", "SafeToSchedule", "Blocked"}
ML_FEATURES = ["age", "condition", "load_ratio", "heat_hours", "prior_faults"]


# =====================================================================
# Learned model behind an interface (validated I/O, timeout, hash-verified artifact, deterministic double)
# =====================================================================
class ModelError(RuntimeError):
    pass


class RiskModel(Protocol):
    name: str
    version: str
    source: str

    def predict(self, features: dict[str, float]) -> float: ...


class MockRiskModel:
    """Deterministic test double (also the safe fallback when the real artifact is missing or fails verification)."""
    name, version, source = "mock_logistic", "1.0", "mock"

    def predict(self, f: dict[str, float]) -> float:
        z = -6.2 + 0.05 * f["age"] + 0.03 * (100 - f["condition"]) + 3.0 * (f["load_ratio"] - 0.6) + 0.012 * f["heat_hours"] + 0.25 * f["prior_faults"]
        return float(1 / (1 + math.exp(-z)))


class JoblibRiskModel:
    """Loads the trained classifier ONLY if its SHA-256 matches the hash recorded by the training run
    (unpickling untrusted files is code execution, so unverified artifacts are refused)."""

    def __init__(self, artifact: Path, metrics_path: Path, timeout_s: float):
        import joblib
        if not artifact.exists():
            raise ModelError("artifact missing")
        try:
            expected = json.loads(metrics_path.read_text(encoding="utf-8"))["artifact_sha256"][artifact.name]
        except Exception as e:
            raise ModelError("no recorded hash for artifact") from e
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise ModelError("artifact hash mismatch - refusing to load")
        blob = joblib.load(artifact)
        self._m, self._features, self._timeout = blob["model"], blob["features"], timeout_s
        self.name, self.version, self.source = "failure_classifier", blob["meta"]["version"], "joblib"
        self._pool = ThreadPoolExecutor(max_workers=1)

    def predict(self, f: dict[str, float]) -> float:
        import pandas as pd
        try:
            row = pd.DataFrame([{k: float(f[k]) for k in self._features}])
            p = float(self._pool.submit(lambda: self._m.predict_proba(row)[0, 1]).result(timeout=self._timeout))
        except FutTimeout as e:
            raise ModelError("model timeout") from e
        except Exception as e:
            raise ModelError(f"model failure: {type(e).__name__}") from e
        if not (0.0 <= p <= 1.0) or math.isnan(p):
            raise ModelError("model output out of range")
        return p


def load_risk_model(cfg: dict) -> tuple[RiskModel, str]:
    """Returns (model, note). Falls back to the deterministic double with an explanatory note."""
    spec = cfg["models"]["failure_classifier"]
    try:
        m = JoblibRiskModel(resolve(cfg, spec["artifact"]), ROOT / "05_ml" / "artifacts" / "metrics.json",
                            cfg["policy"]["tool_timeout_seconds"])
        return m, f"loaded {m.name} v{m.version}"
    except Exception as e:  # noqa: BLE001 - any load problem means: use the safe double
        return MockRiskModel(), f"using mock risk model ({e})"


# =====================================================================
# Planning helpers (search + CSP)
# =====================================================================
class GraphRoute(S.RouteProblem):
    """Inspection routing over the given assets: complete Euclidean graph, depot = node 0."""

    def __init__(self, depot, coords: dict[str, tuple[float, float]], assets: list[str], inspect_cost: float = 1.0):
        self.names = ["DEPOT", *assets]
        self.n, self.inspect_cost = len(self.names), inspect_cost
        self.pts = np.array([depot, *[coords[a] for a in assets]], dtype=float)
        D = np.linalg.norm(self.pts[:, None] - self.pts[None], axis=2)
        self.D = D
        self.adj = {i: [(j, float(D[i, j])) for j in range(self.n) if j != i] for i in range(self.n)}
        self.required = frozenset(range(1, self.n))
        self.start = (0, self.required)


class AgentSched(S.SchedProblem):
    def __init__(self, jobs: list[dict], crews: list[dict], n_slots: int, blocked_slots: set[int]):
        self.n_jobs, self.n_crews, self.n_slots = len(jobs), len(crews), n_slots
        self.feeder = {i: j["feeder"] for i, j in enumerate(jobs)}
        self.skill = {i: j["skill"] for i, j in enumerate(jobs)}
        self.crew_skills = [set(c["skills"]) for c in crews]
        self.blackout = set(blocked_slots)
        self.domain = {i: [(c, s) for c in range(len(crews)) for s in range(n_slots)
                           if self.skill[i] in self.crew_skills[c] and s not in self.blackout] for i in range(len(jobs))}
        self.vars = list(range(len(jobs)))


def plan_route(kb: KnowledgeBase, assets: list[str]) -> dict[str, Any]:
    coords = {a: tuple(kb.topology["transformers"][a]["xy"]) for a in assets if a in kb.topology["transformers"]}
    assets = [a for a in assets if a in coords]
    if not assets:
        return {"order": [], "cost": 0.0, "expanded": 0}
    P = GraphRoute(tuple(kb.topology["depot"]), coords, assets)
    r = S.astar(P)
    order = []
    for node in r["path"] or []:
        if node != 0 and P.names[node] not in order:
            order.append(P.names[node])
    return {"order": order, "cost": round(r["cost"], 3), "expanded": r["expanded"]}


def slot_windows(kb: KnowledgeBase) -> dict[int, str]:
    return {d["slot"]: w for w, d in kb.topology["windows"].items()}


def build_schedule(kb: KnowledgeBase, assets: list[str], available_slots: set[int]) -> tuple[AgentSched, list[dict]]:
    sc = kb.cfg["scheduling"]
    jobs = [{"asset": a, "feeder": kb.feeder_of(a) or "NA", "skill": sc["skill_by_asset_type"][kb.asset_type(a) or "transformer"]}
            for a in assets]
    n_slots = len(kb.topology["windows"])
    P = AgentSched(jobs, sc["crews"], n_slots, set(range(n_slots)) - available_slots)
    return P, jobs


def preferred_schedule(P: AgentSched, jobs: list[dict]) -> dict[int, tuple[int, int]]:
    """What a naive planner prefers: earliest slots, first crew with the skill. It ignores blackouts and capacity -
    the policy layer is what catches that."""
    out = {}
    for i in range(len(jobs)):
        crew = next((c for c in range(P.n_crews) if P.skill[i] in P.crew_skills[c]), 0)
        out[i] = (crew, i % P.n_slots)
    return out


def schedule_violations(P: AgentSched, assign: dict[int, tuple[int, int]], jobs: list[dict]) -> list[str]:
    v = []
    for i, (c, s) in sorted(assign.items()):
        if P.skill[i] not in P.crew_skills[c]:
            v.append(f"{jobs[i]['asset']}: crew{c + 1} lacks skill {P.skill[i]}")
        if s in P.blackout:
            v.append(f"{jobs[i]['asset']}: slot {s} is not safe to schedule")
    ok_pairs = [(i, a) for i, a in assign.items()]
    for (i, a), (j, b) in ((x, y) for n, x in enumerate(ok_pairs) for y in ok_pairs[n + 1:]):
        if not P.binary_ok(i, a, j, b):
            v.append(f"{jobs[i]['asset']}/{jobs[j]['asset']}: crew-slot clash or feeder clearance")
    for i, a in assign.items():
        if not P.cap_ok({k: x for k, x in assign.items() if k != i}, i, a):
            v.append(f"{jobs[i]['asset']}: crew daily capacity exceeded")
    return list(dict.fromkeys(v))


def solve_schedule(P: AgentSched) -> dict[int, tuple[int, int]] | None:
    return S.solve_csp(P, mrv=True, lcv=True, fc=True, use_ac3=True)["assign"]


# =====================================================================
# Inference over the whole KB
# =====================================================================
@dataclass
class Inferences:
    closure: set[str] = field(default_factory=set)
    prov: dict = field(default_factory=dict)
    traces: list[InferenceTrace] = field(default_factory=list)
    risk: dict[str, dict[str, Any]] = field(default_factory=dict)
    forecasts: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class Reasoner:
    def __init__(self, kb: KnowledgeBase, cfg: dict, risk_model: RiskModel):
        self.kb, self.cfg, self.model = kb, cfg, risk_model

    # ---- logic ----
    def _logic(self, now: datetime, out: Inferences) -> None:
        facts = self.kb.trusted_atoms(now)
        out.closure, out.prov = L.forward_chain(facts, self.kb.rules)
        ids = self.kb.atom_fact_ids(now)
        for atom in sorted(out.closure - facts):
            neg, pred, _ = L.parse(atom)
            if pred not in LOGIC_CONCLUSIONS:
                continue
            steps, seen = [], set()

            def walk(a):
                if a in seen or a not in out.prov or out.prov[a][0] == "fact":
                    return
                seen.add(a)
                rule, prem = out.prov[a]
                for x in prem:
                    walk(x)
                steps.append(InferenceStep(step=len(steps) + 1, method=InferenceMethod.FORWARD_CHAINING,
                                           inputs=[ids.get(x, x) for x in prem], rule_or_model=rule, output=a,
                                           note=f"{rule}: {len(prem)} premises hold"))

            walk(atom)
            if steps:
                out.traces.append(InferenceTrace(conclusion=atom, steps=steps))

    # ---- Bayesian network + decision ----
    def _evidence(self, asset: str, now: datetime) -> dict[str, bool]:
        trusted = self.kb.trusted_atoms(now)
        zone, ev = self.kb.zone_of(asset), {}

        def has(pred, ent):
            if f"{pred}({ent})" in trusted:
                return True
            if f"~{pred}({ent})" in trusted:
                return False
            return None
        for var, pred, ent in (("Heat", "TempHigh", zone), ("ReadingHigh", "LoadRatioHigh", asset), ("Poor", "ConditionPoor", asset)):
            v = has(pred, ent)
            if v is not None:
                ev[var] = v
        age = self.kb.latest.get(asset, {}).get("age")
        if age is not None:
            ev["Old"] = age >= self.cfg["thresholds"]["age_old_years"]
        return ev

    def _bayes_and_ml(self, now: datetime, out: Inferences) -> None:
        dcfg = {"abstain_below_confidence": self.cfg["thresholds"]["abstain_below_confidence"], **self.cfg["decision"]}
        for asset in self.kb.topology["transformers"]:
            ev = self._evidence(asset, now)
            if not ({"ReadingHigh", "Poor"} & set(ev)):     # zone heat / age alone is context, not an asset-specific observation
                continue
            d = dec.decide(ev, dcfg)
            rec: dict[str, Any] = {"p_fail": d["p_fail"], "decision": d, "evidence": ev}
            out.traces.append(InferenceTrace(
                conclusion=f"Fail({asset})", probability=round(d["p_fail"], 6), model_versions={"bayes_net": bn.CPT_VERSION},
                steps=[InferenceStep(step=1, method=InferenceMethod.BAYES_NET, inputs=[f"{k}={v}" for k, v in sorted(ev.items())],
                                     rule_or_model=f"bayes_net v{bn.CPT_VERSION}", output=f"P(Fail)={d['p_fail']:.3f}",
                                     note=f"{d['status']}: {d['reason']}"[:300])]))
            hist = self.kb.history.get(f"{asset}:condition", [])
            if len(hist) >= 2:   # HMM health tracking from successive condition readings
                obs = [0 if c >= 70 else 1 if c >= 40 else 2 for _, c in hist]
                rec["hmm_failing"] = float(tp.hmm_filter(obs)[-1][2])
                out.traces.append(InferenceTrace(conclusion=f"HealthState({asset})", probability=round(rec["hmm_failing"], 6),
                    steps=[InferenceStep(step=1, method=InferenceMethod.HMM, inputs=[f"{len(obs)} condition readings"],
                                         rule_or_model="hmm_filter", output=f"P(Failing)={rec['hmm_failing']:.3f}", note="forward filtering")]))
            feats = self.kb.latest.get(asset, {})
            if self.cfg["feature_flags"]["use_ml_risk_second_opinion"] and all(k in feats for k in ML_FEATURES):
                try:
                    rec["p_ml"] = self.model.predict({k: feats[k] for k in ML_FEATURES})
                    rec["ml_source"] = self.model.source
                    out.traces.append(InferenceTrace(conclusion=f"MLRisk({asset})", probability=round(rec["p_ml"], 6),
                        model_versions={self.model.name: self.model.version},
                        steps=[InferenceStep(step=1, method=InferenceMethod.ML_MODEL, inputs=[f"{k}={feats[k]:.2f}" for k in ML_FEATURES],
                                             rule_or_model=f"{self.model.name} v{self.model.version} ({self.model.source})",
                                             output=f"p_fail_30d={rec['p_ml']:.3f}", note="second opinion; never proof")]))
                except ModelError as e:
                    out.notes.append(f"ML second opinion unavailable for {asset}: {e}")
            out.risk[asset] = rec

    # ---- Kalman forecast ----
    def _forecast(self, now: datetime, out: Inferences) -> None:
        th = self.cfg["thresholds"]
        for key, series in self.kb.history.items():
            ent, kind = key.split(":")
            if kind != "load_kw" or ent not in self.kb.topology["feeders"]:
                continue
            if len(series) < th["min_history_for_forecast"]:
                out.notes.append(f"{ent}: only {len(series)} load readings; forecast withheld")
                continue
            t0 = series[0][0].replace(minute=0, second=0, microsecond=0)
            by_hour = {int((ts - t0) / timedelta(hours=1)): v for ts, v in series}
            kf = tp.LoadKalman(level0=series[0][1])
            for h in range(max(by_hour) + 1):
                kf.step(by_hour.get(h))
            pts = kf.forecast(self.cfg["simulation"]["horizon_hours"])
            miss = max(by_hour) + 1 - len(by_hour)
            out.forecasts[ent] = {"horizon_hours": len(pts), "points": [{k: round(v, 2) if k != "k" else v for k, v in p.items()} for p in pts],
                                  "n_obs": len(by_hour), "n_missing_hours": miss}
            out.traces.append(InferenceTrace(conclusion=f"Forecast({ent})", model_versions={"kalman": "local_linear_trend"},
                steps=[InferenceStep(step=1, method=InferenceMethod.KALMAN, inputs=[f"{len(by_hour)} readings, {miss} missing hours"],
                                     rule_or_model="LoadKalman", output=f"24h mean {pts[-1]['mean']:.1f} kW (90%: {pts[-1]['lo']:.1f}-{pts[-1]['hi']:.1f})",
                                     note="missing readings widen the interval")]))

    def infer(self, now: datetime) -> Inferences:
        out = Inferences()
        self._logic(now, out)
        self._bayes_and_ml(now, out)
        self._forecast(now, out)
        return out
