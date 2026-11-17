"""Synthetic CedarLight grid data (all fictitious). Deterministic given `seed`."""
from __future__ import annotations

import numpy as np
import pandas as pd

ZONES = ["ZoneA", "ZoneB", "ZoneC"]


def make_topology(seed: int = 42, feeders_per_zone: int = 4, assets_per_feeder: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    feeders, assets = [], []
    for zi, z in enumerate(ZONES):
        for k in range(feeders_per_zone):
            fid = f"F{zi * feeders_per_zone + k + 1:02d}"
            feeders.append({"feeder": fid, "zone": z, "base_kw": rng.uniform(250, 650),
                            "capacity_kw": 0.0, "temp_offset": [-1.0, 0.5, 2.0][zi]})
    feeders = pd.DataFrame(feeders)
    feeders["capacity_kw"] = feeders["base_kw"] * rng.uniform(1.55, 1.9, len(feeders))
    for _, f in feeders.iterrows():
        for j in range(assets_per_feeder):
            # Zone C is deliberately older -> lets us test fairness/coverage by zone.
            age_scale = {"ZoneA": 9, "ZoneB": 11, "ZoneC": 15}[f["zone"]]
            assets.append({"asset": f"T{len(assets) + 1:02d}", "feeder": f["feeder"], "zone": f["zone"],
                           "age0": float(rng.gamma(4.0, age_scale / 4.0))})
    return feeders, pd.DataFrame(assets)


def make_load(feeders: pd.DataFrame, days: int = 120, seed: int = 42, missing_rate: float = 0.02) -> pd.DataFrame:
    """Hourly feeder load with daily/weekly seasonality, temperature response, noise and missing readings."""
    rng = np.random.default_rng(seed + 1)
    ts = pd.date_range("2026-06-01", periods=days * 24, freq="h")
    hour, dow = ts.hour.values, ts.dayofweek.values
    day = np.arange(len(ts)) / 24.0
    base_temp = 27 + 4 * np.sin(2 * np.pi * day / 60) + 5 * np.sin(2 * np.pi * (hour - 9) / 24)
    storms = rng.random(days) < 0.06  # storm days
    rows = []
    for _, f in feeders.iterrows():
        temp = base_temp + f["temp_offset"] + rng.normal(0, 1.0, len(ts))
        shape = 1 + 0.22 * np.sin(2 * np.pi * (hour - 12) / 24) + 0.10 * np.exp(-((hour - 19) ** 2) / 8)
        weekend = np.where(dow >= 5, 0.92, 1.0)
        cooling = 0.018 * np.clip(temp - 26, 0, None)  # +1.8% per degC above 26C
        true = f["base_kw"] * shape * weekend * (1 + cooling) * (1 + rng.normal(0, 0.03, len(ts)))
        obs = true.copy()
        obs[rng.random(len(ts)) < missing_rate] = np.nan
        rows.append(pd.DataFrame({"ts": ts, "feeder": f["feeder"], "zone": f["zone"], "temp_c": temp,
                                  "storm": np.repeat(storms, 24), "load_kw": obs, "true_load_kw": true}))
    return pd.concat(rows, ignore_index=True).sort_values(["ts", "feeder"]).reset_index(drop=True)


def make_asset_snapshots(feeders: pd.DataFrame, assets: pd.DataFrame, months: int = 36, seed: int = 42,
                         burn_in: int = 12) -> pd.DataFrame:
    """Monthly asset snapshots. label = failure within the next 30 days (ground truth hidden from models)."""
    rng = np.random.default_rng(seed + 2)
    fcap = feeders.set_index("feeder")["capacity_kw"].to_dict()
    fbase = feeders.set_index("feeder")["base_kw"].to_dict()
    rows = []
    for _, a in assets.iterrows():
        history: list[int] = []   # months in which this asset failed (repaired afterwards)
        for m in range(-burn_in, months):   # burn-in months build fault history but are not returned
            prior_faults = sum(1 for fm in history if m - fm <= 12)   # trailing-12-month faults
            age = a["age0"] + (m + burn_in) / 12
            load_ratio = np.clip(fbase[a["feeder"]] * rng.uniform(1.2, 1.6) / fcap[a["feeder"]] + rng.normal(0, 0.04), 0.3, 1.2)
            heat_hours = max(0.0, rng.normal(60 if a["zone"] != "ZoneC" else 80, 20))
            repaired_recently = any(m - fm <= 3 for fm in history)
            cond_true = np.clip(95 - 1.6 * age - 0.05 * heat_hours - 3 * prior_faults + (10 if repaired_recently else 0) + rng.normal(0, 5), 5, 100)
            cond_obs = np.clip(cond_true + rng.normal(0, 4), 0, 100)       # inspection score is noisy
            logit = -6.2 + 0.05 * age + 0.030 * (100 - cond_true) + 3.0 * (load_ratio - 0.6) + 0.012 * heat_hours + 0.25 * prior_faults
            p = 1 / (1 + np.exp(-logit))
            fail = int(rng.random() < p)
            if m < 0:
                if fail:
                    history.append(m)
                continue
            rows.append({"month": m, "asset": a["asset"], "feeder": a["feeder"], "zone": a["zone"], "age": age,
                         "condition": cond_obs, "load_ratio": load_ratio, "heat_hours": heat_hours,
                         "prior_faults": prior_faults, "fail_30d": fail})
            if fail:
                history.append(m)
    return pd.DataFrame(rows)
