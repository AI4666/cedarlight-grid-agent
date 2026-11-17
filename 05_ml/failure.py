"""Asset failure-risk classification: baselines, logistic regression, gradient boosting, small neural net.
Time-ordered split by snapshot month; fairness reported as top-k recall by zone."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "condition", "load_ratio", "heat_hours", "prior_faults"]


def split_months(df, train_end=24, val_end=30):
    return df[df.month < train_end], df[(df.month >= train_end) & (df.month < val_end)], df[df.month >= val_end]


def precision_at_k(y, score, k):
    idx = np.argsort(-score)[:k]
    return float(np.mean(y[idx]))


def metrics(y, score):
    k = max(1, int(round(0.10 * len(y))))
    return {"auc": float(roc_auc_score(y, score)), "pr_auc": float(average_precision_score(y, score)),
            "brier": float(brier_score_loss(y, np.clip(score, 0, 1))), f"precision@top10%": precision_at_k(y.values, score, k),
            "base_rate": float(y.mean())}


def zone_recall_at_k(df, score, frac=0.10, stratified=False):
    """Share of true failures captured by flagging the top `frac` of assets, reported per zone.
    stratified=False: one global ranking.  stratified=True: top `frac` inside each zone (coverage quota)."""
    df = df.reset_index(drop=True)
    flagged = np.zeros(len(df), bool)
    if stratified:
        for _, idx in df.groupby("zone").indices.items():
            k = max(1, int(round(frac * len(idx))))
            flagged[idx[np.argsort(-score[idx])[:k]]] = True
    else:
        flagged[np.argsort(-score)[: max(1, int(round(frac * len(df))))]] = True
    out = {}
    for z, sub in df.assign(flag=flagged).groupby("zone"):
        pos = sub[sub.fail_30d == 1]
        out[z] = {"failures": int(len(pos)), "recall": float(pos.flag.mean()) if len(pos) else None,
                  "flag_rate": float(sub.flag.mean())}
    rec = [v["recall"] for v in out.values() if v["recall"] is not None]
    out["recall_gap"] = float(max(rec) - min(rec)) if len(rec) > 1 else 0.0
    out["overall_recall"] = float(df[df.fail_30d == 1].assign(f=flagged[df.fail_30d == 1]).f.mean())
    return out


def run(snap: pd.DataFrame, seed: int = 42) -> dict:
    tr, va, te = split_months(snap)
    assert tr.month.max() < va.month.min() and va.month.max() < te.month.min(), "temporal leakage"
    Xtr, ytr = tr[FEATURES], tr.fail_30d
    models = {
        # unweighted on purpose: class reweighting would inflate the probabilities the agent consumes
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        "hgb": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=3, random_state=seed),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier((16, 8), alpha=1e-2, max_iter=800, random_state=seed)),
    }
    res = {"age_only_baseline": {"val": metrics(va.fail_30d, va["age"].values), "test": metrics(te.fail_30d, te["age"].values)},
           "condition_only_baseline": {"val": metrics(va.fail_30d, (100 - va["condition"]).values),
                                      "test": metrics(te.fail_30d, (100 - te["condition"]).values)}}
    scores = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        sv, st = m.predict_proba(va[FEATURES])[:, 1], m.predict_proba(te[FEATURES])[:, 1]
        res[name] = {"val": metrics(va.fail_30d, sv), "test": metrics(te.fail_30d, st)}
        scores[name] = (m, st)
    # Selection on validation only, with a simplicity tie-break: among models within 0.01 PR-AUC of the best,
    # prefer the simpler / more explainable one (order: logreg < hgb < mlp).
    top = max(res[k]["val"]["pr_auc"] for k in models)
    best = next(k for k in ("logreg", "hgb", "mlp") if res[k]["val"]["pr_auc"] >= top - 0.01)
    model, st = scores[best]
    fairness = {"global_top_k": zone_recall_at_k(te, st), "zone_stratified_top_k": zone_recall_at_k(te, st, stratified=True)}
    return {"selected": best, "results": res, "fairness_test": fairness, "model": model, "features": FEATURES,
            "n": {"train": len(tr), "val": len(va), "test": len(te)}, "train_base_rate": float(ytr.mean())}
