import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from tradebot import data, db
from tradebot.backtest import compare, format_metrics, run_backtest
from tradebot.data import Quote, drop_incomplete_bars, first_incomplete_date
from tradebot.strategies import buy_and_hold
from tradebot.walkforward import walk_forward


def utc(text):
    return pd.Timestamp(text, tz="UTC")


def bars(symbol, dates, closes=None):
    closes = closes if closes is not None else [100.0] * len(dates)
    return pd.DataFrame({"symbol": symbol, "date": dates, "open": closes, "high": closes,
                         "low": closes, "close": closes, "volume": 1.0})


# ---------------------------------------------------------------- unfinished bars

def test_crypto_bar_for_today_is_never_final():
    assert first_incomplete_date("BTC-USD", utc("2026-11-11 22:00")) == "2026-11-11"
    assert first_incomplete_date("BTC-USD", utc("2026-11-12 00:01")) == "2026-11-12"


def test_stock_bar_is_final_only_after_the_new_york_close():
    # 2026-11-11 is in winter time: New York = UTC-5
    assert first_incomplete_date("SPY", utc("2026-11-11 15:00")) == "2026-11-11"   # 10:00 NY, market open
    assert first_incomplete_date("SPY", utc("2026-11-11 21:05")) == "2026-11-11"   # 16:05 NY, too soon
    assert first_incomplete_date("SPY", utc("2026-11-11 21:20")) == "2026-11-12"   # 16:20 NY, final
    # summer time: New York = UTC-4
    assert first_incomplete_date("SPY", utc("2026-07-01 20:20")) == "2026-07-02"


def test_drop_incomplete_bars_keeps_only_final_ones():
    df = bars("SPY", ["2026-11-09", "2026-11-10", "2026-11-11"])
    during = drop_incomplete_bars(df, "SPY", utc("2026-11-11 15:00"))
    after = drop_incomplete_bars(df, "SPY", utc("2026-11-11 21:30"))
    assert during["date"].tolist() == ["2026-11-09", "2026-11-10"]
    assert after["date"].tolist() == ["2026-11-09", "2026-11-10", "2026-11-11"]
    assert drop_incomplete_bars(df.iloc[:0], "SPY").empty


# ---------------------------------------------------------------- missing closes

def test_fetch_prices_drops_rows_without_a_close(monkeypatch):
    idx = pd.DatetimeIndex(["2026-11-09", "2026-11-10", "2026-11-11"], name="Date")
    raw = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": [10.0, 11.0, np.nan],
                        "Volume": 5.0}, index=idx)
    monkeypatch.setattr(data, "_yf_history", lambda *a: raw)
    out = data.fetch_prices("SPY", "2026-11-01")
    assert out["date"].tolist() == ["2026-11-09", "2026-11-10"]
    assert out.columns.tolist() == data.PRICE_COLUMNS


def test_stored_rows_without_a_close_are_ignored(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    db.init_db(conn)
    db.upsert_prices(conn, bars("SPY", ["2026-11-09", "2026-11-10"], [10.0, 11.0]))
    conn.execute("INSERT INTO prices VALUES ('SPY','2026-11-11',1,1,1,NULL,1)")
    conn.commit()
    assert db.load_close(conn, "SPY").tolist() == [10.0, 11.0]
    assert db.last_price_date(conn, "SPY") == "2026-11-10"


# ---------------------------------------------------------------- quotes

def quote(bid, ask):
    return Quote("AAPL", bid, ask, 1, 1, datetime(2026, 11, 6, tzinfo=timezone.utc), "test")


def test_spread_needs_both_sides():
    assert quote(100.0, 100.10).spread_bps == pytest.approx(0.10 / 100.05 * 1e4)
    assert math.isnan(quote(328.06, 0.0).spread_bps)   # used to print -10000.0bp
    assert math.isnan(quote(0.0, 0.0).spread_bps)
    assert quote(328.06, 0.0).mid == 328.06


# ---------------------------------------------------------------- reports

def test_trade_count_is_shown_as_a_whole_number():
    close = pd.Series([100.0, 101, 102, 103], index=pd.bdate_range("2024-01-01", periods=4))
    r = run_backtest(close, buy_and_hold(close))
    assert format_metrics(compare([r]))["trades"].iloc[0] == "1"


def test_train_sharpe_is_nan_when_no_setting_could_be_scored():
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))),
                      index=pd.bdate_range("2020-01-01", periods=300))

    def always_flat(c, **_):
        return pd.Series(0.0, index=c.index)

    r = walk_forward(close, always_flat, {"x": [1, 2]}, train_bars=150, test_bars=50)
    assert all(math.isnan(f["train_sharpe"]) for f in r.folds)
    assert math.isnan(r.mean_train_sharpe)
