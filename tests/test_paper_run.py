"""The paper-trading run as a whole: repeat runs, guards, order of orders, the log, fills."""
import sqlite3

import pytest

from tradebot import db
from tradebot.broker import PositionInfo


# ---------------------------------------------------------------- repeat runs

def test_second_run_does_not_resend_orders_that_are_still_open(bot):
    """Orders sent after the close sit unfilled until the open, so positions
    still read $0. A second run must not buy everything again."""
    bot.store("AAA"), bot.store("BBB")
    assert bot.run().bought == ["AAA", "BBB"]

    bot.broker.open_symbols = {"AAA", "BBB"}       # queued, nothing filled yet
    again = bot.run()
    assert again.sequence == []
    assert "earlier order is still open" in bot.log_text()


def test_only_the_symbol_with_an_open_order_is_skipped(bot):
    bot.store("AAA"), bot.store("BBB")
    bot.broker.open_symbols = {"AAA"}
    assert bot.run().bought == ["BBB"]


# ---------------------------------------------------------------- guards

def test_daily_loss_still_allows_sells_that_cut_risk(bot):
    """A 3% down day blocks new buys, not reductions."""
    bot.store("AAA"), bot.store("BBB")
    bot.broker.equity = 95_000.0
    bot.broker.held = {"AAA": PositionInfo(30_000.0, 0.10)}   # above the 19k target -> sell some
    b = bot.run()
    assert b.sold == ["AAA"] and b.bought == []               # the BBB buy is blocked


def test_stale_data_blocks_strategy_orders_entirely(bot):
    bot.store("AAA", days_old=30), bot.store("BBB", days_old=30)
    bot.broker.held = {"AAA": PositionInfo(30_000.0, 0.10)}
    assert bot.run().sequence == []


def test_live_trading_refused_cleanly(bot, monkeypatch):
    import main
    monkeypatch.setattr(main, "load_secrets",
                        lambda: {"alpaca_key": "k", "alpaca_secret": "s", "alpaca_paper": False})
    with pytest.raises(SystemExit, match="Live trading is disabled"):
        bot.run()


# ---------------------------------------------------------------- order of orders

def test_sells_go_out_before_buys(bot):
    bot.store("AAA"), bot.store("BBB")
    bot.cfg["paper_bot"]["symbols"] = ["BBB", "AAA"]          # the buy is listed first
    bot.broker.held = {"AAA": PositionInfo(30_000.0, 0.10)}
    assert bot.run().sequence == [("sell", "AAA"), ("buy", "BBB")]


def test_one_rejected_order_does_not_stop_the_rest(bot):
    bot.store("AAA"), bot.store("BBB")
    bot.broker.fail_on = {"AAA"}
    with pytest.raises(SystemExit, match="1 order"):
        bot.run()
    assert bot.broker.last.bought == ["BBB"]
    text = bot.log_text()
    assert "FAILED buy AAA" in text and "AAA rejected" in text
    assert bot.conn.execute("SELECT symbol FROM trades").fetchall() == [("BBB",)]


# ---------------------------------------------------------------- the log

def test_every_run_is_written_to_a_readable_log(bot):
    bot.store("AAA"), bot.store("BBB")
    bot.run(execute=False)
    bot.run()
    text = bot.log_text()
    assert text.count("=" * 72) == 4                          # two runs, a ruled header each
    assert "paper (dry run)" in text and "paper --execute" in text
    assert "equity $100,000.00" in text
    assert "Targets (buy_and_hold): AAA 1.00, BBB 1.00" in text
    assert "sent buy AAA" in text and "order buy-AAA" in text
    assert "Dry run" in text


# ---------------------------------------------------------------- fills

def test_next_run_records_what_happened_to_earlier_orders(bot):
    from tradebot.broker import OrderInfo

    bot.store("AAA"), bot.store("BBB")
    bot.run()
    assert bot.conn.execute(
        "SELECT status, price FROM trades WHERE symbol='AAA'").fetchone() == ("accepted", None)

    bot.broker.order_updates = {"buy-AAA": OrderInfo("filled", 40.0, 500.0),
                                "buy-BBB": OrderInfo("canceled", 0.0, None)}
    bot.broker.held = {"AAA": PositionInfo(20_000.0, 0.0), "BBB": PositionInfo(20_000.0, 0.0)}
    bot.run(execute=False)

    rows = {r[0]: r[1:] for r in bot.conn.execute(
        "SELECT symbol, status, qty, price, notional FROM trades")}
    assert rows["AAA"] == ("filled", 40.0, 500.0, 20_000.0)
    assert rows["BBB"][0] == "canceled" and rows["BBB"][2] is None
    text = bot.log_text()
    assert "filled: 40 @ $500.00" in text and "canceled" in text
    assert db.open_trades(bot.conn, "alpaca_paper") == []     # settled: not asked about again


def test_stale_check_uses_the_exchange_session_not_the_calendar_gap(bot):
    """Prices one session behind are stale, even though they are only a day old."""
    from datetime import date, timedelta
    bot.store("AAA", days_old=1), bot.store("BBB", days_old=1)
    assert bot.run().sequence == []                      # blocked: today's session is missing
    assert "needs " + date.today().isoformat() in bot.log_text()

    bot.broker.expected_session = (date.today() - timedelta(days=1)).isoformat()   # e.g. a holiday today
    assert bot.run().bought == ["AAA", "BBB"]


def test_equity_snapshots_never_overwrite_each_other(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    db.init_db(conn)
    same_moment = "2026-11-11T22:00:00.000000+00:00"
    for equity in (100.0, 101.0, 102.0):
        db.log_equity(conn, "alpaca_paper", equity, ts=same_moment)
    db.log_equity(conn, "other_account", 5.0, ts=same_moment)
    rows = conn.execute("SELECT ts, equity FROM equity_snapshots WHERE account='alpaca_paper' ORDER BY ts").fetchall()
    assert [r[1] for r in rows] == [100.0, 101.0, 102.0]            # all kept, in order
    assert len({r[0] for r in rows}) == 3
    db.log_equity(conn, "alpaca_paper", 103.0)                       # the normal call still works
    assert conn.execute("SELECT COUNT(*) FROM equity_snapshots").fetchone()[0] == 5


def test_old_database_gains_the_status_column(tmp_path):
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute("""CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
        account TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL, qty REAL, notional REAL,
        price REAL, fees REAL DEFAULT 0, strategy TEXT, followed_ai INTEGER, order_id TEXT, notes TEXT)""")
    old.execute("INSERT INTO trades (ts, account, symbol, side) VALUES ('t','a','AAPL','buy')")
    old.commit()
    old.close()

    conn = db.connect(path)
    db.init_db(conn)
    db.init_db(conn)   # running it again must be harmless
    assert conn.execute("SELECT symbol, status FROM trades").fetchall() == [("AAPL", None)]
