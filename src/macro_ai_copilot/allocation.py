"""Map model output and regimes to long-only weights, and run monthly backtests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import quantcore as qc

from . import config

# Prior (hand-specified, not fitted) asset mix for each growth/inflation quadrant.
PLAYBOOK = {
    "Goldilocks":  {"SPY": .30, "EFA": .15, "EEM": .15, "IEF": .15, "TLT": .10, "GLD": .05, "DBC": .10},
    "Reflation":   {"SPY": .20, "EFA": .15, "EEM": .15, "IEF": .10, "TLT": .10, "GLD": .10, "DBC": .20},
    "Stagflation": {"SPY": .10, "EFA": .05, "EEM": .05, "IEF": .20, "TLT": .10, "GLD": .30, "DBC": .20},
    "Slowdown":    {"SPY": .15, "EFA": .05, "EEM": .05, "IEF": .30, "TLT": .30, "GLD": .15, "DBC": .00},
}


def project_capped_simplex(v: np.ndarray, lo: float = config.MIN_WEIGHT, hi: float = config.MAX_WEIGHT) -> np.ndarray:
    """Euclidean projection of ``v`` onto {w : sum w = 1, lo <= w_i <= hi}."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n * lo > 1 + 1e-12 or n * hi < 1 - 1e-12:
        raise ValueError("infeasible weight bounds")
    a, b = v.min() - hi - 1, v.max() - lo + 1
    for _ in range(200):  # bisection on the shift lambda
        lam = (a + b) / 2
        s = np.clip(v - lam, lo, hi).sum()
        if s > 1:
            a = lam
        else:
            b = lam
    w = np.clip(v - (a + b) / 2, lo, hi)
    return w / w.sum()


def tilt_weights(probs: pd.DataFrame, strength: float = config.TILT_STRENGTH,
                 lo: float = config.MIN_WEIGHT, hi: float = config.MAX_WEIGHT) -> pd.DataFrame:
    """Equal-weight base, tilted toward assets the model expects to outperform."""
    n = probs.shape[1]
    raw = 1.0 / n + strength * probs.sub(probs.mean(axis=1), axis=0)
    w = np.vstack([project_capped_simplex(row, lo, hi) for row in raw.values])
    return pd.DataFrame(w, index=probs.index, columns=probs.columns)


def playbook_weights(quadrant: pd.Series, assets: list[str] = config.ASSETS) -> pd.DataFrame:
    rows = [[PLAYBOOK[q].get(a, 0.0) for a in assets] for q in quadrant.values]
    return pd.DataFrame(rows, index=quadrant.index, columns=assets)


def static_weights(index: pd.Index, mix: dict[str, float], assets: list[str] = config.ASSETS) -> pd.DataFrame:
    return pd.DataFrame([[mix.get(a, 0.0) for a in assets]] * len(index), index=index, columns=assets)


def to_trading_days(monthly: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Move month-end decision dates onto the last trading day on/before them."""
    days = prices.index
    pos = days.searchsorted(monthly.index, side="right") - 1
    keep = pos >= 0
    out = monthly[keep].copy()
    out.index = days[pos[keep]]
    return out[~out.index.duplicated(keep="last")]


def drifted_path(rebal: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Daily weight path: reset to target on rebalance days, drift with prices in between.

    Feeding this path to :func:`quantcore.backtest` gives a genuine monthly-rebalanced
    portfolio (turnover only on rebalance days) instead of daily re-balancing to target.
    """
    cols = list(rebal.columns)
    rets = prices[cols].pct_change(fill_method=None).fillna(0.0)
    rebal = rebal.loc[rebal.index >= rets.index[0]]
    out = pd.DataFrame(np.nan, index=rets.index, columns=cols)
    dates = list(rebal.index) + [None]
    for d, nxt in zip(dates[:-1], dates[1:]):
        seg = rets.loc[d:nxt].iloc[:-1] if nxt is not None else rets.loc[d:]
        growth = (1 + seg.iloc[1:]).cumprod()
        w0 = rebal.loc[d].values
        path = np.vstack([w0, growth.values * w0]) if len(growth) else w0[None, :]
        path = path / path.sum(axis=1, keepdims=True)
        out.loc[seg.index] = path
    return out.dropna(how="all")


def run_backtest(monthly_w: pd.DataFrame, prices: pd.DataFrame, name: str,
                 start: pd.Timestamp | None = None) -> qc.BacktestResult:
    """Monthly-rebalanced, cost-aware backtest via QuantCore (signal at close, trade next day)."""
    reb = to_trading_days(monthly_w, prices)
    if start is not None:
        reb = reb.loc[reb.index >= start]
    path = drifted_path(reb, prices)
    costs = qc.CostModel(commission_bps=config.COMMISSION_BPS, slippage_bps=config.SLIPPAGE_BPS)
    res = qc.backtest(path, prices, costs, lag=1, name=name)
    first = path.index[0]
    keep = res.returns.index > first  # first day with a held book
    return qc.BacktestResult(res.returns[keep], res.gross_returns[keep], res.weights[keep],
                             res.turnover[keep], res.costs[keep], name=name)
