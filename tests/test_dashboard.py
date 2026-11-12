"""The dashboard: its calculations on known numbers, and the page itself run headless."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot import account_sync, db
from tradebot import dashboard_data as dd
from tradebot.broker import OrderInfo, PositionInfo

ROOT = Path(__file__).resolve().parent.parent
RISK = {"max_position_pct": 0.20, "max_risk_per_position_pct": 0.015}
DAYS = ["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"]


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "dash.db"))
    db.init_db(c)
    return c


def add_prices(conn, symbol, closes, days=DAYS):
    db.upsert_prices(conn, pd.DataFrame({"symbol": symbol, "date": days[:len(closes)], "open": closes,
                                         "high": closes, "low": closes, "close": closes, "volume": 1.0}))


def fill(conn):
    """A small, fully populated account: two holdings, orders, signals, notes, manual trades."""
    add_prices(conn, "SPY", [100.0, 100.0, 100.0, 101.0, 102.0])
    add_prices(conn, "AAPL", [50.0, 50.0, 50.0, 55.0, 60.0])
    add_prices(conn, "MSFT", [200.0, 200.0, 200.0, 190.0, 180.0])
    db.save_equity_daily(conn, "alpaca_paper", [("2026-11-10", 100_000.0), ("2026-11-11", 100_000.0),
                                                ("2026-11-12", 101_500.0), ("2026-11-13", 103_000.0)])
    db.log_equity(conn, "alpaca_paper", 103_000.0, 60_000.0, ts="2026-11-13T21:30:00+00:00")
    db.save_positions(conn, "alpaca_paper", {"AAPL": PositionInfo(24_000.0, 0.20, 400.0, 50.0),
                                             "MSFT": PositionInfo(18_500.0, -0.07, 97.0, 196.0)})
    db.log_trade(conn, "alpaca_paper", "AAPL", "buy", notional=20_000, order_id="o1", status="filled",
                 qty=400, price=50, ts="2026-11-11T22:00:00+00:00")
    db.log_trade(conn, "alpaca_paper", "MSFT", "buy", notional=20_000, order_id="o2", status="accepted",
                 ts="2026-11-11T22:00:01+00:00")
    db.log_signal(conn, "AAPL", "sma_crossover", 1.0, "as of 2026-11-13")
    db.log_signal(conn, "MSFT", "sma_crossover", 1.0, "as of 2026-11-13")
    db.log_signal(conn, "SPY", "sma_crossover", 0.0, "as of 2026-11-13")
    note = dict(as_of="2026-11-13", model="m", cache_key="k", status="ok", confidence=0.7,
                reasoning="r", sources=json.dumps(["A headline (benzinga, 2026-11-13)"]))
    db.save_research(conn, symbol="AAPL", stance="bullish", summary="Good news costing $5.", **note)
    db.save_research(conn, symbol="MSFT", stance="bearish", summary="Bad news.", **note)
    db.save_research(conn, symbol="SPY", stance="neutral", summary="Quiet.", **note)
    # attempts that failed or found no news store no stance, summary or sources
    db.save_research(conn, symbol="QQQ", as_of="2026-11-13", model="m", cache_key="k2", status="error",
                     detail="no model could answer")
    db.save_research(conn, symbol="NVDA", as_of="2026-11-13", model="m", cache_key="k3", status="no_news")
    db.log_trade(conn, "fidelity_manual", "AAPL", "buy", qty=10, price=50.0, notional=500.0, fees=1.0,
                 followed_ai=1, ts="2026-11-11T15:00:00+00:00")
    db.log_trade(conn, "fidelity_manual", "MSFT", "buy", qty=1, price=200.0, notional=200.0,
                 followed_ai=0, ts="2026-11-11T15:00:00+00:00")


# ---------------------------------------------------------------- account value

def test_equity_series_adds_a_newer_snapshot_but_not_a_same_day_one(conn):
    db.save_equity_daily(conn, "alpaca_paper", [("2026-11-10", 100.0), ("2026-11-11", 101.0)])
    db.log_equity(conn, "alpaca_paper", 999.0, ts="2026-11-11T20:00:00+00:00")     # same New York day
    assert dd.equity_series(conn, "alpaca_paper").tolist() == [100.0, 101.0]
    db.log_equity(conn, "alpaca_paper", 105.0, ts="2026-11-12T15:00:00+00:00")     # the next day
    series = dd.equity_series(conn, "alpaca_paper")
    assert series.tolist() == [100.0, 101.0, 105.0] and series.index[-1] == pd.Timestamp("2026-11-12")


def test_days_use_new_york_dates_not_utc(conn):
    # 02:00 UTC on the 12th is still the evening of the 11th in New York
    db.log_trade(conn, "alpaca_paper", "AAPL", "buy", notional=1, ts="2026-11-12T02:00:00+00:00")
    assert dd.first_trade_date(conn, "alpaca_paper") == pd.Timestamp("2026-11-11")
    assert dd.first_trade_date(conn, "nobody") is None


def test_comparison_starts_both_series_from_the_same_amount():
    idx = pd.to_datetime(DAYS)
    equity = pd.Series([100_000.0, 100_000, 100_000, 101_500, 103_000], index=idx)
    spy = pd.Series([90.0, 95.0, 100.0, 101.0, 102.0], index=idx)
    frame = dd.compare_with_benchmark(equity, spy, pd.Timestamp("2026-11-11"))
    assert frame["date"].dt.strftime("%m-%d").tolist() == ["11-11", "11-12", "11-13"]   # nothing before the start
    assert frame["bot"].tolist() == [100_000, 101_500, 103_000]
    assert frame["benchmark"].tolist() == pytest.approx([100_000, 101_000, 102_000])


def test_comparison_is_empty_until_there_are_two_shared_days():
    idx = pd.to_datetime(DAYS)
    equity = pd.Series([1.0, 1, 1, 1, 1], index=idx)
    assert dd.compare_with_benchmark(equity, equity, None).empty
    assert dd.compare_with_benchmark(equity, equity, pd.Timestamp("2026-11-13")).empty
    assert dd.compare_with_benchmark(equity, equity.iloc[:0], pd.Timestamp("2026-11-09")).empty


def test_performance_numbers_and_no_sharpe_on_short_history():
    stats = dd.performance(pd.Series([100.0, 110.0, 99.0, 104.5]))
    assert stats["total_return"] == pytest.approx(0.045) and stats["max_drawdown"] == pytest.approx(-0.10)
    assert stats["days"] == 3 and stats["sharpe"] is None           # 3 days is not a Sharpe ratio

    rng = np.random.default_rng(0)
    long = pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 60)))
    assert dd.performance(long)["sharpe"] is not None
    assert dd.performance(pd.Series(dtype=float))["total_return"] is None


# ---------------------------------------------------------------- holdings, orders, signals

def test_positions_show_share_of_account_and_distance_to_the_stop(conn):
    fill(conn)
    db.save_positions(conn, "alpaca_paper", {"AAPL": PositionInfo(24_000.0, 0.20, 400, 50),
                                             "MSFT": PositionInfo(18_500.0, -0.07, 97, 196),
                                             "NVDA": PositionInfo(9_000.0, -0.09, 50, 198)})
    table = dd.positions_table(conn, "alpaca_paper", 100_000.0, RISK).set_index("symbol")
    assert table.loc["AAPL", "share_of_account"] == pytest.approx(0.24)
    assert table.loc["MSFT", "room_before_stop"] == pytest.approx(0.005)      # stop at -7.5%
    assert table["status"].to_dict() == {"AAPL": "OK", "MSFT": "Near stop", "NVDA": "Stop hit"}


def test_saving_positions_replaces_the_old_ones(conn):
    db.save_positions(conn, "alpaca_paper", {"AAPL": PositionInfo(1.0, 0.0), "MSFT": PositionInfo(2.0, 0.0)})
    db.save_positions(conn, "alpaca_paper", {"MSFT": PositionInfo(3.0, 0.0)})          # AAPL was sold
    assert conn.execute("SELECT symbol, market_value FROM positions").fetchall() == [("MSFT", 3.0)]


def test_waiting_orders_and_signal_board(conn):
    fill(conn)
    orders = dd.orders_table(conn, "alpaca_paper")
    assert dd.waiting_orders(orders)["symbol"].tolist() == ["MSFT"]

    board = dd.signal_board(conn, "sma_crossover", ["AAPL", "MSFT", "SPY", "QQQ"]).set_index("symbol")
    assert board["agreement"].to_dict() == {"AAPL": "agree", "MSFT": "disagree",
                                            "SPY": "research is neutral", "QQQ": "nothing to compare"}
    assert board.loc["SPY", "bot_view"] == "out" and board.loc["QQQ", "bot_view"] == "no signal"


def test_research_log_handles_attempts_with_nothing_stored(conn):
    fill(conn)
    log = dd.research_log(conn).set_index("symbol")
    assert log.loc["AAPL", "sources"] == "A headline (benzinga, 2026-11-13)"
    assert log.loc["QQQ", "sources"] == "" and log.loc["NVDA", "sources"] == ""
    assert log.loc["QQQ", "status"] == "error"


# ---------------------------------------------------------------- your own trades

def test_manual_trades_are_valued_against_the_same_dollars_in_spy(conn):
    fill(conn)
    table, summary = dd.manual_review(conn)
    aapl, msft = table.set_index("symbol").loc["AAPL"], table.set_index("symbol").loc["MSFT"]
    assert aapl["profit"] == pytest.approx(10 * 60 - 500 - 1)                 # 99 after the $1 fee
    assert aapl["benchmark_profit"] == pytest.approx(500 * 102 / 100 - 500)   # SPY 100 -> 102
    assert msft["profit"] == pytest.approx(180 - 200)
    assert summary["follow_rate"] == 0.5 and summary["trades"] == 2
    by = summary["by_choice"].set_index("choice")["profit"].to_dict()
    assert by == {"Followed the AI": pytest.approx(99.0), "Own call": pytest.approx(-20.0)}


def test_manual_review_copes_with_no_trades_and_unpriced_symbols(conn):
    table, summary = dd.manual_review(conn)
    assert table.empty and summary["trades"] == 0

    add_prices(conn, "SPY", [100.0, 101.0])
    db.log_trade(conn, "fidelity_manual", "ZZZZ", "buy", qty=1, price=10.0, ts="2026-11-09T15:00:00+00:00")
    table, summary = dd.manual_review(conn)
    assert summary["unpriced"] == ["ZZZZ"] and np.isnan(table["profit"].iloc[0])


# ---------------------------------------------------------------- syncing from the broker

class Broker:
    account_name = "alpaca_paper"

    def account(self):
        return {"equity": 101_000.0, "cash": 80_000.0}

    def equity_history(self):
        return [("2026-11-11", 100_000.0), ("2026-11-12", 101_000.0)]

    def position_details(self):
        return {"AAPL": PositionInfo(21_000.0, 0.05, 420.0, 47.6)}

    def order_info(self, order_id):
        return OrderInfo("filled", 420.0, 47.6) if order_id == "o1" else None


def test_refresh_updates_balance_holdings_and_order_results_without_sending_orders(conn):
    db.log_trade(conn, "alpaca_paper", "AAPL", "buy", notional=20_000, order_id="o1", status="accepted")
    broker = Broker()
    assert not hasattr(broker, "submit_notional") and not hasattr(broker, "close_position")
    done = account_sync.refresh_from_broker(conn, broker)

    assert done["positions"] == 1 and done["orders_checked"] == 1 and done["equity"] == 101_000.0
    assert conn.execute("SELECT status, qty, price FROM trades").fetchone() == ("filled", 420.0, 47.6)
    assert dd.equity_series(conn, "alpaca_paper").iloc[-1] == 101_000.0
    assert dd.latest_snapshot(conn, "alpaca_paper")["cash"] == 80_000.0
    assert conn.execute("SELECT symbol, qty FROM positions").fetchall() == [("AAPL", 420.0)]


def test_a_paper_run_keeps_the_dashboard_data_current(bot):
    bot.store("AAA"), bot.store("BBB")
    bot.broker.history = [("2026-11-11", 100_000.0)]
    bot.broker.held = {"AAA": PositionInfo(20_000.0, 0.01, 10.0, 99.0), "BBB": PositionInfo(20_000.0, 0.0)}
    bot.run(execute=False)
    assert bot.conn.execute("SELECT date, equity FROM equity_daily").fetchall() == [("2026-11-11", 100_000.0)]
    assert {r[0] for r in bot.conn.execute("SELECT symbol FROM positions")} == {"AAA", "BBB"}


def test_a_failed_sync_never_stops_a_trading_run(bot, monkeypatch):
    bot.store("AAA"), bot.store("BBB")
    monkeypatch.setattr(type(bot.broker.last or bot.broker), "equity_history",
                        lambda self: (_ for _ in ()).throw(RuntimeError("history endpoint down")))
    assert bot.run().bought == ["AAA", "BBB"]
    assert "could not save account history" in bot.log_text()


# ---------------------------------------------------------------- the page itself

def run_page(db_path, reports, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("TRADEBOT_DB", str(db_path))
    monkeypatch.setenv("TRADEBOT_REPORTS", str(reports))
    page = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=60).run()
    assert not page.exception, [e.value for e in page.exception]
    return page


def test_page_opens_on_an_empty_database(tmp_path, monkeypatch):
    page = run_page(tmp_path / "empty.db", tmp_path, monkeypatch)
    assert page.title[0].value == "Trading assistant"
    assert {m.label: m.value for m in page.metric}["Account value"] == "n/a"
    shown = " ".join(i.value for i in page.info)
    assert "No orders have been sent yet" in shown and "No manual trades logged yet" in shown
    assert "Not run yet" in shown


def test_page_shows_a_filled_account(tmp_path, monkeypatch):
    path = tmp_path / "full.db"
    conn = db.connect(str(path))
    db.init_db(conn)
    fill(conn)
    conn.close()
    (tmp_path / "news_test_report.md").write_text("# News test\n\ncosts of $1 and $2", encoding="utf-8")

    page = run_page(path, tmp_path, monkeypatch)
    tiles = {m.label: m.value for m in page.metric}
    assert tiles["Account value"] == "$103,000.00"
    assert tiles["Return since the first trade"] == "+3.00%"      # 100,000 -> 103,000
    assert tiles["SPY over the same days"] == "+2.00%"            # 100 -> 102
    assert tiles["Sharpe ratio"] == "n/a"                         # two days is not enough
    assert tiles["Profit on your trades"] == "$79.00"             # +99 and -20
    assert tiles["Trades that followed the AI"] == "50.00%"
    assert len(page.dataframe) >= 6                               # holdings, orders, signals, notes, trades...
    assert not page.error
    # the test database has no prices for two of the bot's symbols, and the page says so
    assert [w.value for w in page.warning] == ["No prices for: NVDA, QQQ"]


# ---------------------------------------------------------------- the chart

def sample_frame(bot_end=103_000.0, spy_end=100_500.0):
    days = pd.bdate_range("2026-11-11", periods=4)
    return pd.DataFrame({"date": days, "bot": [100_000.0, 101_000, 102_000, bot_end],
                         "benchmark": [100_000.0, 100_200, 100_400, spy_end]})


def test_chart_gives_each_series_its_own_fixed_colour_in_both_modes():
    from tradebot.dashboard_charts import COLOURS, SERIES, comparison_chart
    for mode in ("light", "dark"):
        spec = comparison_chart(sample_frame(), mode).to_dict()        # also validates the chart spec
        scale = spec["layer"][0]["encoding"]["color"]["scale"]
        assert scale["domain"] == SERIES and scale["range"] == COLOURS[mode]
        assert spec["layer"][0]["mark"]["strokeWidth"] == 2
        assert spec["layer"][0]["encoding"]["y"]["scale"]["zero"] is False
    assert COLOURS["light"] != COLOURS["dark"]


def test_chart_labels_the_line_ends_only_when_they_are_far_enough_apart():
    from tradebot.dashboard_charts import comparison_chart
    marks = lambda frame: [layer["mark"]["type"] for layer in comparison_chart(frame, "light").to_dict()["layer"]]
    assert marks(sample_frame()) == ["line", "point", "text", "rule"]
    assert marks(sample_frame(bot_end=100_510.0)) == ["line", "point", "rule"]     # would collide: no labels


def test_chart_date_axis_steps_in_whole_days():
    from tradebot.dashboard_charts import comparison_chart
    axis = lambda frame: comparison_chart(frame, "light").to_dict()["layer"][0]["encoding"]["x"]["axis"]["tickCount"]
    assert axis(sample_frame()) == {"interval": "day", "step": 1}
    long = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=120), "bot": 1.0, "benchmark": 1.0})
    assert axis(long)["step"] > 10
