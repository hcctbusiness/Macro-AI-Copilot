import numpy as np
import pandas as pd
import pytest

from macro_ai_copilot import allocation, config, model


@pytest.mark.parametrize("seed", range(5))
def test_projection_respects_bounds(seed):
    v = np.random.default_rng(seed).normal(0, 0.5, 7)
    w = allocation.project_capped_simplex(v, 0.0, 0.35)
    assert abs(w.sum() - 1) < 1e-9
    assert w.min() >= -1e-12 and w.max() <= 0.35 + 1e-9


def test_projection_infeasible():
    with pytest.raises(ValueError):
        allocation.project_capped_simplex(np.zeros(2), 0.0, 0.3)


def test_tilt_weights_sum_and_limits():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    probs = pd.DataFrame(rng.uniform(0, 1, (24, 7)), idx, config.ASSETS)
    w = allocation.tilt_weights(probs, strength=3.0)
    assert np.allclose(w.sum(axis=1), 1)
    assert (w.values >= -1e-12).all() and (w.values <= config.MAX_WEIGHT + 1e-9).all()
    # Higher probability ⇒ weakly higher weight.
    for t in idx:
        order = probs.loc[t].sort_values().index
        assert (np.diff(w.loc[t, order].values) >= -1e-9).all()


def test_equal_probabilities_give_equal_weight():
    probs = pd.DataFrame(0.5, index=pd.date_range("2020-01-31", periods=2, freq="ME"), columns=config.ASSETS)
    assert np.allclose(allocation.tilt_weights(probs).values, 1 / 7)


def test_playbook_weights_valid():
    for q, mix in allocation.PLAYBOOK.items():
        assert abs(sum(mix.values()) - 1) < 1e-9, q
        assert max(mix.values()) <= config.MAX_WEIGHT
        assert set(mix) == set(config.ASSETS)


def test_monthly_backtest_trades_only_on_rebalance_days(prices):
    idx = prices.resample("ME").last().index
    w = allocation.static_weights(idx, {a: 1 / 7 for a in config.ASSETS})
    res = allocation.run_backtest(w, prices, "EW")
    assert abs(res.turnover.iloc[0] - 1.0) < 1e-9  # initial purchase
    reb_exec = set(allocation.to_trading_days(w, prices).index)
    days = res.turnover.index
    prev_day = pd.Series(days, index=days).shift(1)
    non_reb = [d for d in days[1:] if prev_day[d] not in reb_exec]
    assert res.turnover.loc[non_reb].abs().max() < 1e-10
    assert np.allclose(res.weights.sum(axis=1), 1)


def test_monthly_backtest_matches_buy_and_hold_drift(prices):
    idx = prices.resample("ME").last().index
    w = allocation.static_weights(idx, {"SPY": 1.0})
    res = allocation.run_backtest(w, prices, "SPY", start=prices.index[0])
    spy = prices["SPY"].pct_change().loc[res.returns.index]
    assert np.allclose(res.gross_returns.values, spy.values)


def test_monthly_returns_index():
    px = pd.DataFrame({"A": np.linspace(100, 120, 60)}, index=pd.bdate_range("2020-01-01", periods=60))
    m = model.monthly_returns(px)
    assert m.index.is_month_end.all()
