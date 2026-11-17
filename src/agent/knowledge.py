"""STAGE 3 - KNOWLEDGE BASE. Facts, rules, ontology relations and model metadata, all with provenance.

* Static facts (topology, tool policy) are PROTECTED: percepts can never overwrite them.
* Dynamic facts come only from trusted propositions and expire after a TTL.
* Contradictory observations are kept side by side and reported as conflicts (never silently resolved).
* Low-confidence propositions are remembered as `uncertain` and can never be used as premises."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from . import _lib  # noqa: F401
from .config import ROOT, resolve
from .schemas import ConflictRecord, KnowledgeFact, KnowledgeRule, Percept, Proposition
from logic_engine import fmt  # noqa: E402

PROTECTED = {"Forbidden", "ToolOnAllowList", "LocatedIn", "Feeds", "CriticalLoad", "WindowIn"}


def _atom(pred: str, args: list[str], truth: bool = True) -> str:
    return fmt(not truth, pred, tuple(args))


class KnowledgeBase:
    def __init__(self, cfg: dict[str, Any], topology_path: Path | None = None):
        self.cfg = cfg
        self.topology = json.loads(Path(topology_path or resolve(cfg, cfg["paths"]["topology"])).read_text(encoding="utf-8"))
        self.static: dict[str, KnowledgeFact] = {}
        self.dynamic: list[KnowledgeFact] = []
        self.meta: dict[str, dict[str, Any]] = {}          # fact_id -> {source, obs_ts, confidence}
        self.uncertain: dict[str, float] = {}               # atom -> best confidence (never a premise)
        self.untrusted_requests: list[dict[str, Any]] = []  # ToolRequest propositions from free text
        self.protected_attempts: list[ConflictRecord] = []
        self.latest: dict[str, dict[str, float]] = {}       # entity -> latest numeric observations (for ML features)
        self.history: dict[str, list[tuple[datetime, float]]] = {}  # time series (e.g. feeder load_kw, asset condition)
        self.models: dict[str, dict[str, Any]] = {}
        self.rules: list[dict[str, Any]] = yaml.safe_load((ROOT / "03_logic" / "rules.yaml").read_text(encoding="utf-8"))["horn_rules"]
        self._load_static()

    # ---------- static knowledge ----------
    def _add_static(self, pred: str, args: list[str], truth: bool = True, provenance: str = "topology"):
        f = KnowledgeFact(predicate=pred, args=args, truth=truth, provenance=provenance)
        self.static[_atom(pred, args, truth)] = f

    def _load_static(self) -> None:
        t = self.topology
        for sub, d in t["substations"].items():
            self._add_static("LocatedIn", [sub, d["zone"]])
        for fid, d in t["feeders"].items():
            self._add_static("LocatedIn", [fid, d["zone"]])
            self._add_static("Feeds", [d["from"], fid])
        for tid, d in t["transformers"].items():
            self._add_static("LocatedIn", [tid, d["zone"]])
            self._add_static("Feeds", [d["feeder"], tid])
        for cid, d in t["customers"].items():
            self._add_static("Feeds", [d["fed_by"], cid])
            # Topology is closed-world: customers not listed as critical are explicitly NOT critical.
            self._add_static("CriticalLoad", [cid], truth=bool(d["critical"]))
        for w, d in t["windows"].items():
            self._add_static("WindowIn", [w, d["zone"]])
        pol = self.cfg["policy"]
        for tool in pol["allowed_tools"]:
            self._add_static("ToolOnAllowList", [tool], provenance="config.policy.allowed_tools")
        for tool in pol["forbidden_tools"]:
            self._add_static("Forbidden", [tool], provenance="config.policy.forbidden_tools")

    # ---------- topology helpers ----------
    def zone_of(self, entity: str) -> str | None:
        t = self.topology
        for group in ("transformers", "feeders", "substations"):
            if entity in t[group]:
                return t[group][entity]["zone"]
        return None

    def asset_type(self, entity: str) -> str | None:
        for group, typ in (("transformers", "transformer"), ("feeders", "feeder"), ("substations", "substation")):
            if entity in self.topology[group]:
                return typ
        return None

    def feeder_of(self, asset: str) -> str | None:
        return self.topology["transformers"].get(asset, {}).get("feeder")

    # ---------- updates ----------
    def register_model(self, name: str, version: str, source: str, metrics: dict | None = None) -> None:
        self.models[name] = {"version": version, "source": source, "metrics": metrics or {}}

    def observe(self, p: Percept) -> None:
        """Remember numeric observations for feature construction and time-series models (trusted percepts only)."""
        if p.validation_status.value != "valid":
            return
        for k, v in p.values.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.latest.setdefault(p.entity_id, {})[k] = float(v)
                if k in ("load_kw", "condition"):
                    h = self.history.setdefault(f"{p.entity_id}:{k}", [])
                    if not any(ts == p.timestamp for ts, _ in h):
                        h.append((p.timestamp, float(v)))
                        h.sort()

    def tell(self, prop: Proposition, p: Percept) -> dict[str, str]:
        atom_pos = _atom(prop.predicate, prop.args, prop.truth)
        if prop.predicate == "ToolRequest":
            self.untrusted_requests.append({"tool": prop.args[0], "percept_id": p.percept_id, "source": p.source})
            return {"status": "untrusted_request", "atom": atom_pos}
        if prop.predicate in PROTECTED:
            opposite = _atom(prop.predicate, prop.args, not prop.truth)
            if opposite in self.static:
                self.protected_attempts.append(ConflictRecord(kind="protected_static", atom=atom_pos,
                    fact_ids=[self.static[opposite].fact_id], note="percept contradicts protected static knowledge; ignored"))
            return {"status": "rejected_protected", "atom": atom_pos}
        if prop.confidence < self.cfg["thresholds"]["min_percept_quality"]:
            self.uncertain[atom_pos] = max(self.uncertain.get(atom_pos, 0.0), prop.confidence)
            return {"status": "untrusted_low_confidence", "atom": atom_pos}

        for f in self.dynamic:
            if f.valid_to is None and f.predicate == prop.predicate and f.args == prop.args:
                m = self.meta[f.fact_id]
                if f.truth == prop.truth:
                    if p.timestamp > m["obs_ts"]:
                        m["obs_ts"] = p.timestamp
                    return {"status": "duplicate", "atom": atom_pos}
                if m["source"] == p.source and p.timestamp > m["obs_ts"]:   # newer reading from same source supersedes
                    f.valid_to = p.timestamp
        fact = KnowledgeFact(predicate=prop.predicate, args=prop.args, truth=prop.truth, valid_from=p.timestamp,
                             provenance=f"percept:{p.percept_id}|source:{p.source}|conf:{prop.confidence}")
        self.dynamic.append(fact)
        self.meta[fact.fact_id] = {"source": p.source, "obs_ts": p.timestamp, "confidence": prop.confidence}
        return {"status": "added", "atom": atom_pos, "fact_id": fact.fact_id}

    # ---------- views ----------
    def active_dynamic(self, now: datetime) -> list[KnowledgeFact]:
        ttl = timedelta(hours=self.cfg["thresholds"]["fact_ttl_hours"])
        return [f for f in self.dynamic if f.valid_to is None and self.meta[f.fact_id]["obs_ts"] >= now - ttl]

    def trusted_atoms(self, now: datetime) -> set[str]:
        return set(self.static) | {_atom(f.predicate, f.args, f.truth) for f in self.active_dynamic(now)}

    def uncertain_atoms(self) -> set[str]:
        return set(self.uncertain)

    def atom_fact_ids(self, now: datetime) -> dict[str, str]:
        out = {a: f.fact_id for a, f in self.static.items()}
        out.update({_atom(f.predicate, f.args, f.truth): f.fact_id for f in self.active_dynamic(now)})
        return out

    def conflicts(self, now: datetime) -> list[ConflictRecord]:
        act = self.active_dynamic(now)
        out = list(self.protected_attempts)
        by_atom: dict[tuple, dict[bool, KnowledgeFact]] = {}
        for f in act:
            by_atom.setdefault((f.predicate, tuple(f.args)), {})[f.truth] = f
        for (pred, args), sides in by_atom.items():
            if True in sides and False in sides:
                out.append(ConflictRecord(kind="observation", atom=_atom(pred, list(args)),
                           fact_ids=[sides[True].fact_id, sides[False].fact_id],
                           note="active facts assert both the atom and its negation"))
        return out

    def snapshot(self, now: datetime) -> dict[str, Any]:
        return {"static_facts": len(self.static), "dynamic_active": len(self.active_dynamic(now)),
                "uncertain": dict(self.uncertain), "untrusted_requests": len(self.untrusted_requests),
                "conflicts": [c.atom for c in self.conflicts(now)], "models": self.models}
