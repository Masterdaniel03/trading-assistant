import numpy as np
import pandas as pd
import pytest

from tradebot.strategies import rsi, rsi_reversion, sma_crossover


@pytest.fixture
def ohlcv_frame():
    index = pd.to_datetime([
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
        "2024-01-16",
    ])
    close = pd.Series(
        [100.0, 90.0, 80.0, 90.0, 100.0, 90.0, 80.0, 90.0, 100.0],
        index=index,
    )
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [100.0, 150.0, 150.0, 200.0, 0.0, 300.0, 0.0, 350.0, 400.0],
        },
        index=index,
    )


def test_sma_crossover_uses_rolling_windows_and_preserves_dates(ohlcv_frame):
    target = sma_crossover(ohlcv_frame["close"], fast=2, slow=3)

    assert target.index.equals(ohlcv_frame.index)
    assert target.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    assert ohlcv_frame.index[3] - ohlcv_frame.index[2] > pd.Timedelta(days=1)
    assert ohlcv_frame["volume"].iloc[4] == 0.0
    assert target.iloc[4] == 1.0


def test_rsi_returns_expected_values(ohlcv_frame):
    actual = rsi(ohlcv_frame["close"], period=2)
    expected = [50.0, 0.0, 0.0, 50.0, 75.0, 37.5, 18.75, 59.375, 79.6875]

    assert actual.index.equals(ohlcv_frame.index)
    assert actual.tolist() == pytest.approx(expected)


def test_rsi_reversion_emits_expected_stateful_signals(ohlcv_frame):
    target = rsi_reversion(
        ohlcv_frame["close"], period=2, buy_below=30, sell_above=55
    )

    assert target.index.equals(ohlcv_frame.index)
    assert target.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_flat_prices_produce_neutral_rsi_and_no_rsi_reversion_signal():
    close = pd.Series([42.0] * 5, index=pd.date_range("2024-02-01", periods=5))

    assert rsi(close, period=3).tolist() == [50.0] * 5
    assert rsi_reversion(close, period=3).tolist() == [0.0] * 5


def test_nan_close_recovers_after_rolling_window_and_rsi_stays_finite(ohlcv_frame):
    close = ohlcv_frame["close"].copy()
    close.iloc[4] = np.nan

    target = sma_crossover(close, fast=2, slow=3)
    rsi_values = rsi(close, period=2)

    assert target.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert target.index.equals(close.index)
    assert rsi_values.index.equals(close.index)
    assert rsi_values.notna().all()
