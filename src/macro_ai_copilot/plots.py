"""Regime timeline chart with asset performance per regime."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

QUAD_COLORS = {"Goldilocks": "#2E7D32", "Reflation": "#D9822B", "Stagflation": "#B23A48", "Slowdown": "#328CC1"}
GMM_COLORS = ["#CFE3D4", "#9FC2DE", "#F2C38B", "#D9737F"]
ASSET_COLORS = ["#0B3C5D", "#328CC1", "#8FB8DE", "#C9A227", "#7A5C3E", "#2E7D32", "#B23A48"]


def _spans(labels: pd.Series):
    """Yield (start, end, label) runs of a monthly label series."""
    lab = labels.values
    idx = labels.index
    start = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[start]:
            end = idx[i] if i < len(lab) else idx[-1] + pd.offsets.MonthEnd(1)
            yield idx[start] - pd.offsets.MonthEnd(1), end, lab[start]
            start = i


def regime_chart(prices: pd.DataFrame, quadrant: pd.Series, gmm: pd.Series, gmm_names: dict[int, str],
                 perf_quad: pd.DataFrame, perf_gmm: pd.DataFrame, path: str | Path) -> Path:
    fig = plt.figure(figsize=(14, 12), dpi=110)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.3, 0.45, 2.0], hspace=0.42, wspace=0.18, top=0.93)
    ax = fig.add_subplot(gs[0, :])
    axg = fig.add_subplot(gs[1, :], sharex=ax)
    axq, axr = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

    q = quadrant.loc[quadrant.index >= prices.index[0]]
    for s, e, lab in _spans(q):
        ax.axvspan(s, e, color=QUAD_COLORS[lab], alpha=0.16, lw=0)
    for a, c in zip(["SPY", "TLT", "GLD", "DBC"], ["#0B3C5D", "#328CC1", "#C9A227", "#7A5C3E"]):
        ax.plot(prices[a] / prices[a].iloc[0], color=c, lw=1.4, label=a)
    ax.set_yscale("log")
    ax.set_xlim(prices.index[0], prices.index[-1])
    ax.set_title("Growth × inflation quadrant (shaded) and asset prices, growth of $1 (log scale)",
                 loc="left", fontweight="bold")
    h1 = [Patch(color=QUAD_COLORS[k], alpha=0.35, label=k) for k in QUAD_COLORS]
    leg1 = ax.legend(handles=h1, loc="upper left", frameon=False, ncol=4, fontsize=9)
    ax.add_artist(leg1)
    ax.legend(loc="lower right", frameon=False, ncol=4, fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    g = gmm.loc[gmm.index >= prices.index[0]]
    for s, e, lab in _spans(g):
        axg.axvspan(s, e, color=GMM_COLORS[int(lab) % len(GMM_COLORS)], lw=0)
    axg.set_yticks([])
    axg.set_title("Unsupervised regime (Gaussian mixture, expanding-window fit, ordered by financial stress)",
                  loc="left", fontweight="bold", fontsize=10)
    h2 = [Patch(color=GMM_COLORS[k], label=v) for k, v in gmm_names.items()]
    axg.legend(handles=h2, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=len(h2), frameon=False, fontsize=9)
    plt.setp(ax.get_xticklabels(), visible=True)

    for axis, perf, title in [(axq, perf_quad, "Annualised return by quadrant"),
                              (axr, perf_gmm, "Annualised return by GMM regime")]:
        n_reg, n_ast = perf.shape
        width = 0.8 / n_ast
        x = np.arange(n_reg)
        for j, a in enumerate(perf.columns):
            axis.bar(x + (j - (n_ast - 1) / 2) * width, perf[a].values * 100, width,
                     color=ASSET_COLORS[j % len(ASSET_COLORS)], label=a)
        axis.set_xticks(x, [str(i).replace(" (", "\n(") for i in perf.index], fontsize=9)
        axis.axhline(0, color="k", lw=0.6)
        axis.set_ylabel("% per year")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axq.legend(frameon=False, ncol=7, fontsize=9, loc="upper center", bbox_to_anchor=(1.09, -0.2))
    fig.suptitle("Macro regimes and multi-asset behaviour", fontsize=15, fontweight="bold", x=0.06, ha="left")
    fig.text(0.99, 0.005, "HCCT Group · Macro-AI-Copilot (descriptive, in-sample statistics)",
             ha="right", fontsize=8, color="#7A7A7A")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
