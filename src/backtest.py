from typing import Callable, Dict, List

import numpy as np
import pandas as pd


def backtest_regime_strategy(
    returns: pd.DataFrame,
    regimes_pred: pd.Series,
    weight_fn: Callable[[str, List[str]], np.ndarray],
) -> pd.Series:
    data = returns.join(regimes_pred, how="inner").dropna()
    assets = list(returns.columns)

    port_rets = []
    for date, row in data.iterrows():
        r = row[assets]
        reg = row["regime_pred"]
        w = weight_fn(reg, assets)
        port_rets.append((date, float(np.dot(w, r.values))))

    port = pd.Series(
        [x[1] for x in port_rets],
        index=[x[0] for x in port_rets],
        name="strategy_return",
    )
    return port


def summary_stats(rets: pd.Series, freq: int = 4) -> Dict[str, float]:
    ann_ret = (1 + rets).prod() ** (freq / len(rets)) - 1
    ann_vol = rets.std() * (freq ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    equity = (1 + rets).cumprod()
    dd = (equity.cummax() - equity) / equity
    max_dd = dd.max() if len(dd) else float("nan")

    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }
