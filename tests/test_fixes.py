import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradebot import db
from tradebot.data import history_was_readjusted, incremental_start
from tradebot.rebalance import plan_orders
from tradebot.risk import RiskViolation, check_data_fresh

RISK = {"max_position_pct": 0.2, "max_total_exposure_pct": 1.0,
        "daily_loss_limit_pct": 0.03, "min_order_notional": 1.0}


# ---------------------------------------------------------------- incremental fetch

def test_incremental_start():
    assert incremental_start(None, "2015-01-01") == "2015-01-01"
    assert incremental_start("2026-11-06", "2015-01-01", overlap_days=10) == "2026-10-27"
    assert incremental_start("2015-01-03", "2015-01-01", overlap_days=10) == "2015-01-01"


def _stored(closes, start="2026-10-26"):
    return pd.Series(closes, index=pd.bdate_range(start, periods=len(closes)), dtype=float)


def _fresh(stored, closes):
    return pd.DataFrame({"date": stored.index.strftime("%Y-%m-%d"), "close": closes})


def test_unchanged_overlap_is_not_readjusted():
    s = _stored([100, 101, 102, 103])
    assert not history_was_readjusted(s, _fresh(s, [100, 101, 102, 103]))


def test_partial_last_bar_is_ignored():
    # Last stored bar was saved intraday; the final close differs. Not a re-adjustment.
    s = _stored([100, 101, 102, 103])
    assert not history_was_readjusted(s, _fresh(s, [100, 101, 102, 105]))


def test_dividend_readjustment_detected():
    s = _stored([100, 101, 102, 103])
    assert history_was_readjusted(s, _fresh(s, [99.7, 100.7, 101.7, 102.7]))


def test_empty_fresh_is_not_readjusted():
    s = _stored([100, 101])
    assert not history_was_readjusted(s, pd.DataFrame(columns=["date", "close"]))


# ---------------------------------------------------------------- stale data guard

def test_fresh_data_passes():
    check_data_fresh({"SPY": "2026-11-06"}, "2026-11-06")
    check_data_fresh({"SPY": "2026-11-09"}, "2026-11-06")      # newer than needed is fine


def test_stale_or_missing_data_blocks_trading():
    with pytest.raises(RiskViolation, match="QQQ \\(no data\\)"):
        check_data_fresh({"SPY": "2026-11-06", "QQQ": None}, "2026-11-06")
    # one session behind is already stale: this is the "fetch lagged by a day" case
    with pytest.raises(RiskViolation, match="SPY \\(has 2026-11-05, needs 2026-11-06\\)"):
        check_data_fresh({"SPY": "2026-11-05"}, "2026-11-06")
    with pytest.raises(RiskViolation, match="Could not tell"):
        check_data_fresh({"SPY": "2026-11-06"}, None)


def test_last_completed_session_follows_new_york_not_the_local_clock():
    from tradebot.data import last_completed_session
    sessions = ["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11"]   # Fri, Mon, Tue, Wed
    utc = lambda t: pd.Timestamp(t, tz="UTC")
    # Wed 15:00 New York: Wednesday's bar is not final yet
    assert last_completed_session(sessions, utc("2026-11-11 20:00")) == "2026-11-10"
    # Wed 16:20 New York: now it is
    assert last_completed_session(sessions, utc("2026-11-11 21:20")) == "2026-11-11"
    # Thu 00:30 UTC is still Wed evening in New York (and Wed afternoon in California)
    assert last_completed_session(sessions, utc("2026-11-12 00:30")) == "2026-11-11"
    # Sunday: Friday's session, whatever the calendar day is
    assert last_completed_session(sessions[:1], utc("2026-11-08 18:00")) == "2026-11-06"
    assert last_completed_session([], utc("2026-11-08 18:00")) is None


# ---------------------------------------------------------------- unlisted positions

def test_unlisted_positions_closed_only_when_asked():
    current = {"AAPL": 800.0, "TSLA": 300.0}
    kept = plan_orders({"AAPL": 1.0}, current, equity=4000, risk=RISK)
    assert all(o.symbol != "TSLA" for o in kept)

    orders = plan_orders({"AAPL": 1.0}, current, equity=4000, risk=RISK, close_unlisted=True)
    by = {o.symbol: o for o in orders}
    assert by["TSLA"].side == "close" and by["TSLA"].notional == 300.0
    assert "AAPL" not in by  # already on target


# ---------------------------------------------------------------- close check

def test_pending_buy_is_not_mistaken_for_a_close():
    from alpaca.trading.enums import OrderSide

    from tradebot.broker import AlpacaBroker

    buy = SimpleNamespace(side=OrderSide.BUY, id="b1")
    sell = SimpleNamespace(side=OrderSide.SELL, id="s1")

    class Client:
        def __init__(self, orders):
            self.orders = orders

        def get_all_positions(self):
            return [SimpleNamespace(symbol="AAPL")]

        def get_orders(self, req):
            return self.orders

    b = AlpacaBroker.__new__(AlpacaBroker)
    b.client = Client([buy])
    assert b._close_in_flight("AAPL") is None
    b.client = Client([buy, sell])
    assert b._close_in_flight("AAPL") is sell


# ---------------------------------------------------------------- signal dedupe

def test_repeated_signal_not_logged_twice(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    db.init_db(conn)
    assert db.log_signal(conn, "SPY", "sma", 1.0, "as of 2026-11-06")
    assert not db.log_signal(conn, "SPY", "sma", 1.0, "as of 2026-11-06")
    assert db.log_signal(conn, "SPY", "sma", 0.0, "as of 2026-11-09")
    assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == 2
