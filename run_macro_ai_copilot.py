from pathlib import Path

import matplotlib.pyplot as plt

from src.data_loading import load_macro_data, load_market_data, load_text_data
from src.features import (
    build_macro_features,
    build_sentiment_index,
    combine_macro_and_sentiment,
)
from src.regime_model import (
    label_macro_regimes,
    train_regime_classifier,
    predict_regimes,
)
from src.allocation import simple_regime_weights
from src.backtest import backtest_regime_strategy, summary_stats


macro = load_macro_data(start="1995-01-01")

prices, returns = load_market_data(
    tickers=["SPY", "AGG", "GLD"],
    start="1995-01-01",
    freq="Q",
)

text_path = Path("data/raw/news_headlines.csv")
text_df = load_text_data(text_path)

macro_feat = build_macro_features(macro, lags=2)
sent_ts = build_sentiment_index(text_df, freq="Q")
features = combine_macro_and_sentiment(macro_feat, sent_ts)

regimes = label_macro_regimes(macro)
regime_model, metrics = train_regime_classifier(features, regimes)

print("Labels:", metrics["labels"])
print("Expansion precision:", metrics["classification_report"]["expansion"]["precision"])

regimes_pred = predict_regimes(regime_model, features)

aligned_returns = returns.loc[features.index].dropna()

strategy_rets = backtest_regime_strategy(
    aligned_returns,
    regimes_pred,
    weight_fn=simple_regime_weights,
)

ew_rets = aligned_returns.mean(axis=1)

strategy_stats = summary_stats(strategy_rets, freq=4)
ew_stats = summary_stats(ew_rets, freq=4)

print("\nRegime strategy:", strategy_stats)
print("Equal-weight:", ew_stats)

cum_strategy = (1 + strategy_rets).cumprod()
cum_ew = (1 + ew_rets).cumprod()

plt.figure(figsize=(9, 4))
plt.plot(cum_strategy, label="Regime-aware strategy")
plt.plot(cum_ew, label="Equal-weight portfolio")
plt.ylabel("Growth of $1")
plt.title("Macro-AI Copilot: Regime-aware vs Equal-weight")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
