import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradebot.config import _real_key
from tradebot.strategies import buy_and_hold, momentum, vol_target
from tradebot.walkforward import expand_grid, make_folds, walk_forward


@pytest.fixture
def prices():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=1000)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx)))), index=idx)


def fixed(close, level=1.0, **_):
    return pd.Series(level, index=close.index)


# ---------------------------------------------------------------- folds

def test_folds_are_contiguous_and_end_at_last_bar():
    folds = make_folds(1000, train_bars=500, test_bars=100)
    assert len(folds) == 5
    assert folds[-1].test_end == 1000
    for a, b in zip(folds, folds[1:]):
        assert a.test_end == b.test_start
    assert all(f.test_start - f.train_start == 500 for f in folds)


def test_too_little_data_raises():
    with pytest.raises(ValueError, match="Need at least"):
        make_folds(100, train_bars=500, test_bars=100)


def test_expand_grid():
    assert expand_grid({"a": [1, 2], "b": [3]}) == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]
    assert expand_grid({}) == [{}]


# ---------------------------------------------------------------- walk-forward

def test_strategy_never_sees_past_the_window_it_is_tested_on(prices):
    seen = []

    def spy(close, **_):
        seen.append(close.index[-1])
        return fixed(close)

    folds = make_folds(len(prices), 500, 100)
    walk_forward(prices, spy, {"x": [1, 2]}, train_bars=500, test_bars=100)
    expected = []
    for f in folds:
        expected += [prices.index[f.test_start - 1]] * 2 + [prices.index[f.test_end - 1]]
    assert seen == expected


def test_changing_later_prices_does_not_change_earlier_folds(prices):
    from tradebot.strategies import sma_crossover

    grid = {"fast": [5, 20], "slow": [50, 100]}
    base = walk_forward(prices, sma_crossover, grid, train_bars=500, test_bars=100)
    shocked = prices.copy()
    shocked.iloc[-100:] *= 2  # only the last test window
    other = walk_forward(shocked, sma_crossover, grid, train_bars=500, test_bars=100)
    for a, b in zip(base.folds[:-1], other.folds[:-1]):
        assert a["params"] == b["params"]
        assert a["test_return"] == pytest.approx(b["test_return"])


def test_picks_the_parameter_that_won_in_training(prices):
    r = walk_forward(prices, fixed, {"level": [0.0, 1.0]}, train_bars=500, test_bars=100)
    # level 0 has no volatility so its Sharpe is NaN and can't be selected
    assert all(f["params"] == {"level": 1.0} for f in r.folds)


def test_only_test_windows_are_reported(prices):
    r = walk_forward(prices, buy_and_hold, {}, train_bars=500, test_bars=100, cost_bps=0)
    folds = make_folds(len(prices), 500, 100)
    assert r.oos.equity.index[0] == prices.index[folds[0].test_start]
    assert r.oos.equity.index[-1] == prices.index[-1]
    assert r.oos.metrics["total_return"] == pytest.approx(r.benchmark.metrics["total_return"])


# ---------------------------------------------------------------- strategies

def test_momentum_compares_skip_to_lookback():
    close = pd.Series([10.0, 11, 12, 13, 5, 6], index=pd.date_range("2024-01-01", periods=6))
    # t: close[t-1] > close[t-3], flat until both exist
    # t=3: 12 > 10 long; t=4: 13 > 11 long; t=5: 5 > 12 false
    assert momentum(close, lookback=3, skip=1).tolist() == [0, 0, 0, 1, 1, 0]


def test_vol_target_scales_down_and_never_levers():
    idx = pd.bdate_range("2024-01-01", periods=120)
    calm = 100 * np.cumprod(1 + np.tile([0.001, -0.001], 30))
    wild = calm[-1] * np.cumprod(1 + np.tile([0.05, -0.05], 30))
    close = pd.Series(np.concatenate([calm, wild]), index=idx)
    t = vol_target(close, target_vol=0.15, window=20)
    assert (t.iloc[:20] == 0).all()   # warm-up
    assert t.iloc[59] == 1.0          # calm: capped, no leverage
    assert t.iloc[-1] < 0.25          # wild: cut back
    assert t.between(0, 1).all()


def test_vol_target_handles_flat_prices():
    close = pd.Series([50.0] * 40, index=pd.bdate_range("2024-01-01", periods=40))
    assert vol_target(close, window=20).between(0, 1).all()


# ---------------------------------------------------------------- keys

@pytest.mark.parametrize("value", [None, "", "your_paper_key_here",
                                   "https://paper-api.alpaca.markets/v2"])
def test_placeholders_and_urls_are_not_keys(value):
    assert _real_key(value) is None


def test_real_key_is_kept():
    assert _real_key("PKABC123") == "PKABC123"
