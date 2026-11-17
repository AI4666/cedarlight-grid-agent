"""Load forecasting: features, time-ordered splits, baselines vs learned model, interval calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline

FEATURES = ["hour_sin", "hour_cos", "dow", "is_weekend", "temp_c", "storm", "lag24", "lag168", "feeder_idx"]


def build_features(load: pd.DataFrame) -> pd.DataFrame:
    """Lags use only observations >= 24h old (a 24h-ahead forecast). Temperature is the (forecast) signal."""
    df = load.sort_values(["feeder", "ts"]).copy()
    g = df.groupby("feeder")["load_kw"]
    df["lag24"] = g.shift(24)
    df["lag168"] = g.shift(168)
    df["hour_sin"] = np.sin(2 * np.pi * df["ts"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["ts"].dt.hour / 24)
    df["dow"] = df["ts"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["storm"] = df["storm"].astype(int)
    df["feeder_idx"] = df["feeder"].str[1:].astype(int)
    df = df.dropna(subset=["lag168"])                 # first week has no weekly lag
    df = df.dropna(subset=["load_kw"])                # train/evaluate on observed targets only
    return df.sort_values(["ts", "feeder"]).reset_index(drop=True)


def split_by_time(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    ts = np.sort(df["ts"].unique())
    t1, t2 = ts[int(len(ts) * train_frac)], ts[int(len(ts) * (train_frac + val_frac))]
    return df[df["ts"] < t1], df[(df["ts"] >= t1) & (df["ts"] < t2)], df[df["ts"] >= t2]


def mape(y, p):
    return float(np.mean(np.abs(y - p) / np.abs(y)))


def _fill(a, b):  # baseline fallback when a lag is missing
    return np.where(np.isnan(a), b, a)


def evaluate(df, pred):
    y = df["load_kw"].values
    return {"mae": float(mean_absolute_error(y, pred)), "mape": mape(y, pred)}


def run(load: pd.DataFrame, seed: int = 42) -> dict:
    df = build_features(load)
    tr, va, te = split_by_time(df)
    assert tr["ts"].max() < va["ts"].min() <= va["ts"].max() < te["ts"].min(), "temporal leakage"

    models = {
        "linear": make_pipeline(SimpleImputer(strategy="median"), LinearRegression()),
        "hgb": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=seed),
    }
    results = {"persistence_24h": {}, "seasonal_naive_168h": {}}
    for name, d in (("val", va), ("test", te)):
        results["persistence_24h"][name] = evaluate(d, _fill(d["lag24"].values, d["lag168"].values))
        results["seasonal_naive_168h"][name] = evaluate(d, d["lag168"].values)

    cv = {}
    for name, m in models.items():
        maes = []
        for tri, vai in TimeSeriesSplit(n_splits=4).split(tr):
            m.fit(tr.iloc[tri][FEATURES], tr.iloc[tri]["load_kw"])
            maes.append(mean_absolute_error(tr.iloc[vai]["load_kw"], m.predict(tr.iloc[vai][FEATURES])))
        cv[name] = {"cv_mae_mean": float(np.mean(maes)), "cv_mae_std": float(np.std(maes))}
        m.fit(tr[FEATURES], tr["load_kw"])
        results[name] = {"val": evaluate(va, m.predict(va[FEATURES])), "test": evaluate(te, m.predict(te[FEATURES])),
                         **cv[name]}

    best = min(models, key=lambda k: results[k]["val"]["mae"])          # selection on validation only
    model = models[best]
    # Empirical 90% interval from validation residuals (relative), then check coverage on test.
    rel = (va["load_kw"].values - model.predict(va[FEATURES])) / model.predict(va[FEATURES])
    lo_q, hi_q = np.quantile(rel, [0.05, 0.95])
    p_te = model.predict(te[FEATURES])
    cover = float(np.mean((te["load_kw"].values >= p_te * (1 + lo_q)) & (te["load_kw"].values <= p_te * (1 + hi_q))))
    return {"selected": best, "results": results, "interval": {"lo_rel": float(lo_q), "hi_rel": float(hi_q),
            "test_coverage": cover, "target": 0.90}, "n": {"train": len(tr), "val": len(va), "test": len(te)},
            "model": model, "features": FEATURES}
