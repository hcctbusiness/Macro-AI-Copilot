"""Regime detection.

1. **Growth × inflation quadrant** (rule-based, nothing fitted): direction of the
   composite growth score and of CPI inflation over the last 6 months.
2. **Gaussian mixture on standardised macro features**, fitted on an expanding
   window only. At each month *t* the scaler and the mixture are fitted on data
   up to *t* and the label for *t* is predicted; clusters are re-ordered by their
   financial-stress centroid so labels are comparable across refits.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import config

QUADRANTS = ["Goldilocks", "Reflation", "Stagflation", "Slowdown"]
STRESS_COLS = ["nfci", "log_vix", "credit_baa10y"]


def quadrant_regimes(scores: pd.DataFrame, feat: pd.DataFrame, window: int = 6) -> pd.Series:
    """Label each month Goldilocks / Reflation / Stagflation / Slowdown.

    growth rising    : composite growth score higher than ``window`` months ago
    inflation rising : CPI y/y higher than ``window`` months ago
    """
    g_up = scores["growth"].diff(window) > 0
    cpi = feat["cpi_yoy"].reindex(scores.index)
    i_up = cpi.diff(window) > 0
    lab = np.select(
        [g_up & ~i_up, g_up & i_up, ~g_up & i_up],
        ["Goldilocks", "Reflation", "Stagflation"],
        default="Slowdown",
    )
    valid = scores["growth"].diff(window).notna() & cpi.diff(window).notna()
    return pd.Series(lab, index=scores.index, name="quadrant")[valid]


def _ordered_labels(gmm: GaussianMixture, cols: list[str]) -> np.ndarray:
    """Map raw component ids → rank by stress (0 = calmest)."""
    idx = [cols.index(c) for c in STRESS_COLS if c in cols]
    stress = gmm.means_[:, idx].mean(axis=1) if idx else gmm.means_.mean(axis=1)
    order = np.argsort(stress)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap


def fit_gmm_expanding(
    feat: pd.DataFrame,
    n_clusters: int = config.N_CLUSTERS,
    min_months: int = config.REGIME_MIN_MONTHS,
    refit_every: int = 1,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Expanding-window GMM regimes.

    Returns a frame with column ``regime`` (0 = lowest stress … n-1 = highest) and
    ``p_<k>`` membership probabilities, starting at the ``min_months``-th row.
    Only rows ``<= t`` are ever used to produce the label for ``t``.
    """
    X = feat.dropna()
    cols = list(X.columns)
    rows = []
    gmm = scaler = remap = None
    for i in range(min_months - 1, len(X)):
        if gmm is None or (i - (min_months - 1)) % refit_every == 0:
            hist = X.iloc[: i + 1].values
            scaler = StandardScaler().fit(hist)
            gmm = GaussianMixture(n_clusters, covariance_type="diag", n_init=3,
                                  reg_covar=1e-3, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gmm.fit(scaler.transform(hist))
            remap = _ordered_labels(gmm, cols)
        x_t = scaler.transform(X.iloc[[i]].values)
        prob_raw = gmm.predict_proba(x_t)[0]
        prob = np.zeros(n_clusters)
        prob[remap] = prob_raw
        rows.append([int(np.argmax(prob)), *prob])
    out = pd.DataFrame(rows, index=X.index[min_months - 1:],
                       columns=["regime", *[f"p_{k}" for k in range(n_clusters)]])
    out["regime"] = out["regime"].astype(int)
    return out


def gmm_names(n_clusters: int = config.N_CLUSTERS) -> dict[int, str]:
    base = {4: ["Calm", "Normal", "Elevated stress", "High stress"],
            3: ["Calm", "Normal", "Stress"]}
    names = base.get(n_clusters, [f"Regime {k}" for k in range(n_clusters)])
    return dict(enumerate(names))
