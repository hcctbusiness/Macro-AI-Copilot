"""Walk-forward ML: predict which assets beat the equal-weight basket next month.

For each asset *i* and month-end *t* the label is
``y[i, t] = 1  if  r[i, t→t+1] > mean_j r[j, t→t+1]``.
At prediction date *t* a model is trained only on pairs (X[s], y[s]) with
``s <= t-1`` — every one of those labels is fully realised by *t* — and then
predicts from X[t]. There is no tuning on the test period: hyper-parameters are
fixed a priori in this file.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def make_model(kind: str = "logit", seed: int = config.SEED):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    if kind == "gbm":
        return GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                                          subsample=0.7, min_samples_leaf=10, random_state=seed)
    raise ValueError(f"unknown model kind {kind!r}")


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calendar month-end to month-end simple returns (indexed by month-end)."""
    m = prices.resample("ME").last()
    return m.pct_change(fill_method=None)


def relative_targets(mret: pd.DataFrame) -> pd.DataFrame:
    """Forward 1-month outperformance labels indexed by the *decision* month-end."""
    fwd = mret.shift(-1)
    rel = fwd.sub(fwd.mean(axis=1), axis=0)
    y = (rel > 0).astype(float)
    y[fwd.isna()] = np.nan
    return y


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.DataFrame,
    kind: str = "logit",
    min_train: int = config.MIN_TRAIN_MONTHS,
    retrain_every: int = 1,
) -> pd.DataFrame:
    """Out-of-sample probabilities P(asset beats basket next month), one row per month.

    ``X`` – features indexed by month-end; ``y`` – labels from :func:`relative_targets`.
    """
    idx = X.index.intersection(y.index)
    X, y = X.loc[idx], y.loc[idx]
    preds = {}
    models: dict[str, object] = {}
    for pos in range(len(idx)):
        t = idx[pos]
        train_idx = idx[:pos]  # strictly before t  (labels for s<=t-1 are known at t)
        out = {}
        for a in y.columns:
            ya = y.loc[train_idx, a].dropna()
            if len(ya) < min_train or ya.nunique() < 2:
                continue
            if a not in models or (len(ya) - min_train) % retrain_every == 0:
                m = make_model(kind)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(X.loc[ya.index].values, ya.values.astype(int))
                models[a] = m
            out[a] = float(models[a].predict_proba(X.loc[[t]].values)[0, 1])
        if len(out) == y.shape[1]:
            preds[t] = out
    return pd.DataFrame.from_dict(preds, orient="index")[list(y.columns)] if preds else pd.DataFrame(columns=y.columns)


def prediction_diagnostics(probs: pd.DataFrame, mret: pd.DataFrame) -> dict[str, float]:
    """Out-of-sample hit rate, mean rank IC and top-vs-bottom spread."""
    fwd = mret.shift(-1).reindex(probs.index)
    rel = fwd.sub(fwd.mean(axis=1), axis=0)
    ok = fwd.notna().all(axis=1)
    p, r = probs[ok], rel[ok]
    hit = float((((p > 0.5) == (r > 0)).values).mean())
    ic = p.rank(axis=1).corrwith(r.rank(axis=1), axis=1)
    top = r.values[np.arange(len(p)), p.values.argmax(axis=1)]
    bot = r.values[np.arange(len(p)), p.values.argmin(axis=1)]
    return {
        "Months": int(ok.sum()),
        "Hit rate": hit,
        "Mean rank IC": float(ic.mean()),
        "IC t-stat": float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))),
        "Top-minus-bottom (monthly)": float(np.mean(top - bot)),
    }
