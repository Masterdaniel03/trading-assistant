import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
from tradebot import db
from tradebot.backtest import apply_stop_loss
from tradebot.broker import PositionInfo
from tradebot.risk import stop_loss_pct, triggered_stops

RISK = {"max_position_pct": 0.20, "max_total_exposure_pct": 1.0, "max_risk_per_position_pct": 0.015,
        "stop_cooldown_bars": 2, "daily_loss_limit_pct": 0.03, "min_order_notional": 1.0,
        "max_data_age_days": 4, "allow_live": False}


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


# ---------------------------------------------------------------- the rule

def test_full_size_stop_risks_exactly_the_budget():
    assert stop_loss_pct(RISK) == pytest.approx(0.075)
    assert RISK["max_position_pct"] * stop_loss_pct(RISK) == pytest.approx(0.015)


def test_triggered_stops():
    hit = triggered_stops({"A": -0.08, "B": -0.05, "C": 0.10, "D": -0.075}, RISK)
    assert hit == {"A": -0.08, "D": -0.075}


# ---------------------------------------------------------------- backtest stop

def test_stop_exits_then_cools_down_then_reenters():
    close = _series([100, 95, 92, 91, 90, 90, 90])
    #                entry ok  stop  blocked x2 (bars 2,3), bar 4 re-enters
    target = apply_stop_loss(close, _series([1] * 7), 0.075, cooldown_bars=2)
    assert target.tolist() == [1, 1, 0, 0, 1, 1, 1]


def test_new_entry_resets_the_reference_price():
    close = _series([100, 110, 100, 100, 93, 92])
    # flat at bar 2 clears the 100 entry; re-enters at bar 3 (100); 92 is -8% from there
    target = apply_stop_loss(close, _series([1, 1, 0, 1, 1, 1]), 0.075, cooldown_bars=2)
    assert target.tolist() == [1, 1, 0, 1, 1, 0]


def test_stop_does_not_use_future_prices():
    close = _series([100, 99, 98, 97, 96, 50])
    full = apply_stop_loss(close, _series([1] * 6), 0.075)
    head = apply_stop_loss(close.iloc[:5], _series([1] * 5), 0.075)
    assert full.iloc[:5].tolist() == head.tolist()


def test_flat_signal_never_stops():
    close = _series([100, 50, 25])
    assert apply_stop_loss(close, _series([0, 0, 0]), 0.075).tolist() == [0, 0, 0]


# ---------------------------------------------------------------- the bot

class FakeBroker:
    account_name = "alpaca_paper"
    last = {}

    def __init__(self, *a, **k):
        self.closed, self.bought = [], []
        FakeBroker.last = self

    def account(self):
        return {"equity": FakeBroker.equity, "last_equity": 100_000.0, "cash": 0.0, "status": "ACTIVE"}

    def position_details(self):
        return FakeBroker.held

    def is_market_open(self):
        return True

    def close_position(self, symbol):
        self.closed.append(symbol)
        return SimpleNamespace(id="c")

    def submit_notional(self, symbol, side, notional):
        self.bought.append(symbol)
        return SimpleNamespace(id="b")


@pytest.fixture
def bot(tmp_path, monkeypatch):
    import tradebot.broker
    monkeypatch.setattr(tradebot.broker, "AlpacaBroker", FakeBroker)
    monkeypatch.setattr(main, "load_secrets", lambda: {"alpaca_key": "k", "alpaca_secret": "s", "alpaca_paper": True})
    cfg = {"db_path": str(tmp_path / "t.db"), "risk": RISK, "strategies": {},
           "paper_bot": {"strategy": "buy_and_hold", "symbols": ["AAA", "BBB"]}}
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    FakeBroker.equity = 100_000.0
    FakeBroker.held = {}

    def store(symbol, days_old=0, n=5):
        last = date.today() - timedelta(days=days_old)
        dates = [(last - timedelta(days=i)).isoformat() for i in range(n)]
        db.upsert_prices(conn, pd.DataFrame({"symbol": symbol, "date": dates, "open": 1.0, "high": 1.0,
                                             "low": 1.0, "close": 1.0, "volume": 1.0}))

    return SimpleNamespace(cfg=cfg, conn=conn, store=store)


def run(bot, execute=True):
    main.cmd_paper(bot.cfg, argparse.Namespace(execute=execute))
    return FakeBroker.last


def test_stop_closes_position_and_is_recorded(bot):
    bot.store("AAA"), bot.store("BBB")
    FakeBroker.held = {"AAA": PositionInfo(18_000.0, -0.08), "BBB": PositionInfo(20_000.0, 0.02)}
    b = run(bot)
    assert b.closed == ["AAA"] and b.bought == []
    assert db.last_stop_date(bot.conn, "AAA") == date.today().isoformat()
    assert bot.conn.execute("SELECT notes FROM trades").fetchone()[0] == "stop-loss"


def test_dry_run_sends_and_records_nothing(bot):
    bot.store("AAA"), bot.store("BBB")
    FakeBroker.held = {"AAA": PositionInfo(18_000.0, -0.08)}
    b = run(bot, execute=False)
    assert b.closed == [] and b.bought == []
    assert db.last_stop_date(bot.conn, "AAA") is None


def test_daily_loss_blocks_buys_but_not_stops(bot):
    bot.store("AAA"), bot.store("BBB")
    FakeBroker.equity = 95_000.0  # -5% today; BBB would otherwise be bought
    FakeBroker.held = {"AAA": PositionInfo(18_000.0, -0.08)}
    b = run(bot)
    assert b.closed == ["AAA"] and b.bought == []


def test_stale_data_blocks_buys_but_not_stops(bot):
    bot.store("AAA", days_old=30), bot.store("BBB", days_old=30)
    FakeBroker.held = {"AAA": PositionInfo(18_000.0, -0.08), "BBB": PositionInfo(1_000.0, 0.01)}
    b = run(bot)
    assert b.closed == ["AAA"] and b.bought == []


def test_halt_with_no_stops_sends_nothing(bot):
    bot.store("AAA"), bot.store("BBB")
    FakeBroker.equity = 95_000.0
    with pytest.raises(SystemExit):
        run(bot)
    assert FakeBroker.last.bought == []


def test_cooldown_blocks_rebuy_until_enough_bars_pass(bot):
    bot.store("AAA"), bot.store("BBB")
    today = date.today().isoformat()
    FakeBroker.held = {"BBB": PositionInfo(20_000.0, 0.0)}

    db.log_stop(bot.conn, "AAA", today, -0.08)  # 0 bars since
    assert run(bot).bought == []

    earlier = (date.today() - timedelta(days=2)).isoformat()
    bot.conn.execute("UPDATE stop_events SET date = ?", (earlier,))  # 2 bars since = cooldown over
    bot.conn.commit()
    assert run(bot).bought == ["AAA"]
