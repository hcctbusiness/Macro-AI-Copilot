from typing import Dict, List

import numpy as np
import pandas as pd


def estimate_regime_stats(
    returns: pd.DataFrame, regimes: pd.Series
) -> Dict[str, Dict[str, pd.Series]]:
    data = returns.join(regimes, how="inner").dropna()
    stats: Dict[str, Dict[str, pd.Series]] = {}
    for r in sorted(data["regime"].unique()):
        sub = data[data["regime"] == r]
        stats[r] = {
            "mu": sub[returns.columns].mean(),
            "cov": sub[returns.columns].cov(),
        }
    return stats


def simple_regime_weights(regime: str, assets: List[str]) -> np.ndarray:
    n = len(assets)
    w = np.zeros(n)

    if regime == "expansion":
        w[:] = 1.0 / n
        w[0] += 0.2
        w[-1] -= 0.2 / (n - 1)
    elif regime == "recession":
        w[:] = 1.0 / n
        w[1] += 0.2
        w[-1] += 0.1
        w[0] -= 0.3
    else:
        w[:] = 1.0 / n

    w = np.maximum(w, 0)
    return w / w.sum()
