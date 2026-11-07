"""End-to-end pipeline: data → features → regimes → ML → allocation → backtest → report."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import quantcore as qc

from . import allocation, config, data, features, model, plots, regimes, sentiment


@dataclass
class PipelineOutput:
    features: pd.DataFrame
    quadrant: pd.Series
    gmm: pd.DataFrame
    probs: dict[str, pd.DataFrame]
    weights: dict[str, pd.DataFrame]
    results: list
    stats: pd.DataFrame
    diagnostics: pd.DataFrame
    perf_quad: pd.DataFrame
    perf_gmm: pd.DataFrame
    extra: dict = field(default_factory=dict)


def regime_performance(mret: pd.DataFrame, labels: pd.Series, names: dict | None = None) -> pd.DataFrame:
    """Annualised mean next-month return of each asset, grouped by the regime known at t."""
    fwd = mret.shift(-1)
    df = fwd.join(labels.rename("lab"), how="inner").dropna()
    g = df.groupby("lab")
    perf = g[list(mret.columns)].mean() * 12
    counts = g.size()
    order = list(names) if names is not None else list(perf.index)
    order = [o for o in order if o in perf.index]
    perf, counts = perf.loc[order], counts.loc[order]
    label = {k: f"{names[k] if isinstance(names, dict) else k} (n={counts[k]})" for k in order}
    return perf.rename(index=label)


def run(docs: str | Path = "docs", news_csv: str | Path = "data/raw/news_headlines.csv",
        refresh: bool = False, verbose: bool = True) -> PipelineOutput:
    log = print if verbose else (lambda *a, **k: None)
    docs = Path(docs)
    docs.mkdir(parents=True, exist_ok=True)

    log("1/6 Loading FRED macro data and ETF prices ...")
    raw = data.load_macro(refresh=refresh)
    px = data.load_prices(refresh=refresh)

    log("2/6 Building point-in-time macro features ...")
    panel = features.align_macro(raw)
    feat = features.build_features(panel)
    scores = features.growth_inflation_scores(feat)

    log("3/6 Detecting regimes (quadrant rule + expanding-window GMM) ...")
    quad = regimes.quadrant_regimes(scores, feat)
    gmm = regimes.fit_gmm_expanding(feat)
    names = regimes.gmm_names()

    log("4/6 Walk-forward ML predictions (logistic + gradient boosting) ...")
    mret = model.monthly_returns(px)
    y = model.relative_targets(mret)
    probs = {
        "logit": model.walk_forward_predict(feat, y, kind="logit"),
        "gbm": model.walk_forward_predict(feat, y, kind="gbm", retrain_every=3),
    }

    log("5/6 Allocation and monthly-rebalanced backtests ...")
    common = probs["logit"].index.intersection(probs["gbm"].index).intersection(quad.index)
    weights = {
        "ML Tilt (Logit)": allocation.tilt_weights(probs["logit"].loc[common]),
        "ML Tilt (GBM)": allocation.tilt_weights(probs["gbm"].loc[common]),
        "Regime Playbook": allocation.playbook_weights(quad.loc[common]),
        "60/40 (SPY/IEF)": allocation.static_weights(common, {"SPY": 0.6, "IEF": 0.4}),
        "Equal Weight": allocation.static_weights(common, {a: 1 / len(config.ASSETS) for a in config.ASSETS}),
    }
    start = allocation.to_trading_days(weights["Equal Weight"], px).index[0]
    results = [allocation.run_backtest(w, px, name=n, start=start) for n, w in weights.items()]
    stats = qc.compare(results)
    bench = results[-1].returns
    for r in results:
        stats.loc["Beta vs EW", r.name] = qc.metrics.beta_alpha(r.returns, bench)[0]
    diag = pd.DataFrame({k: model.prediction_diagnostics(v.loc[common], mret) for k, v in probs.items()})

    # Robustness: same logistic model with every monthly macro series lagged 2 months.
    lag2 = features.build_features(features.align_macro(raw, monthly={k: 2 for k in config.MONTHLY_SERIES}))
    p2 = model.walk_forward_predict(lag2, y, kind="logit")
    c2 = common.intersection(p2.index)
    rob = allocation.run_backtest(allocation.tilt_weights(p2.loc[c2]), px, name="ML Tilt (Logit, 2m lag)", start=start)
    robustness = pd.concat([stats["ML Tilt (Logit)"], rob.stats()], axis=1)
    diag["logit_lag2"] = pd.Series(model.prediction_diagnostics(p2.loc[c2], mret))

    log("6/6 Charts, sentiment demo and report ...")
    tear = {r.name: r.returns for r in results if r.name in ("ML Tilt (Logit)", "60/40 (SPY/IEF)", "Equal Weight")}
    qc.report.tearsheet(tear, docs / "tearsheet.png", title="Macro-AI-Copilot — ML tilt vs benchmarks")
    qc.report.tearsheet({r.name: r.returns for r in results}, docs / "tearsheet_all.png",
                        title="Macro-AI-Copilot — all strategies")
    perf_quad = regime_performance(mret, quad, {q: q for q in regimes.QUADRANTS})
    perf_gmm = regime_performance(mret, gmm["regime"], names)
    plots.regime_chart(px, quad, gmm["regime"], names, perf_quad, perf_gmm, docs / "regimes.png")

    news = sentiment.load_headlines(news_csv)
    sent = sentiment.evaluate(news)
    extra = {
        "robustness": robustness,
        "start": results[0].returns.index[0], "end": results[0].returns.index[-1],
        "sentiment": sent, "sentiment_weekly": sentiment.weekly_index(news),
        "latest": {
            "date": common[-1], "quadrant": quad.iloc[-1], "quadrant_date": quad.index[-1],
            "gmm": names[int(gmm["regime"].iloc[-1])], "gmm_date": gmm.index[-1],
            "weights": weights["ML Tilt (Logit)"].iloc[-1],
            "probs": probs["logit"].iloc[-1],
        },
        "quad_share": quad.loc[common].value_counts(normalize=True),
        "gmm_share": gmm["regime"].loc[common.intersection(gmm.index)].map(names).value_counts(normalize=True),
        "data_end": {c: panel[c].last_valid_index() for c in panel.columns},
    }
    return PipelineOutput(feat, quad, gmm, probs, weights, results, stats, diag, perf_quad, perf_gmm, extra)
