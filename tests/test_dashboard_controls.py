"""The dashboard's league, run history, alerts and buttons."""
import json
from pathlib import Path

import pandas as pd
import pytest

from tradebot import alerts, db, league
from tradebot import dashboard_data as dd
from tradebot.dashboard_charts import COLOURS, INK, OTHERS, league_chart

ROOT = Path(__file__).resolve().parent.parent
RISK = {"max_position_pct": 0.20, "max_total_exposure_pct": 1.0, "max_risk_per_position_pct": 0.015,
        "stop_cooldown_bars": 2, "daily_loss_limit_pct": 0.03, "min_order_notional": 1.0}
DAYS = ["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"]
CFG = {"benchmark": "SPY", "start_date": "2026-11-01", "backtest": {"cost_bps": 0}, "risk": RISK,
       "strategies": {}, "universes": {"pair": ["AAA", "BBB"]},
       "league": {"start_cash": 100_000, "history_start": "2026-11-09",
                  "bots": [{"name": "hold-pair", "strategy": "buy_and_hold", "universe": "pair"},
                           {"name": "hold-aaa", "strategy": "buy_and_hold", "symbols": ["AAA"]}]}}


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "dash.db"))
    db.init_db(c)
    return c


def store(conn, symbol, closes):
    db.upsert_prices(conn, pd.DataFrame({"symbol": symbol, "date": DAYS[:len(closes)], "open": closes,
                                         "high": closes, "low": closes, "close": closes, "volume": 1.0}))


def fill_league(conn):
    """Four days, a first league run (so the fifth day is live), then the fifth day."""
    for symbol in ("AAA", "BBB", "SPY"):
        store(conn, symbol, [10, 10, 10, 11])
    league.run_league(conn, CFG, lambda line: None)
    for symbol, last in (("AAA", 12.1), ("BBB", 11), ("SPY", 11.55)):
        store(conn, symbol, [10, 10, 10, 11, last])
    league.run_league(conn, CFG, lambda line: None)


# ---------------------------------------------------------------- league numbers

def test_league_curves_start_every_line_at_the_same_amount(conn):
    fill_league(conn)
    curves = dd.league_curves(conn, "SPY", "All history")
    assert list(curves.columns) == ["date", "hold-aaa", "hold-pair", dd.BENCHMARK_LABEL]
    assert curves.iloc[0, 1:].tolist() == [100_000.0] * 3
    assert curves["hold-pair"].iloc[-1] == pytest.approx(106_200)
    assert curves[dd.BENCHMARK_LABEL].iloc[-1] == pytest.approx(115_500)     # SPY 10 -> 11.55


def test_live_only_starts_on_the_day_the_settings_were_fixed(conn):
    fill_league(conn)
    live = dd.league_curves(conn, "SPY", "Live only", "2026-11-12")
    assert [d.strftime("%Y-%m-%d") for d in live["date"]] == ["2026-11-12", "2026-11-13"]
    assert live["hold-pair"].tolist() == pytest.approx([100_000, 100_000 * 106_200 / 104_000])
    assert dd.league_curves(conn, "SPY", "Live only", "2026-11-13").empty     # one day is not a line yet
    assert dd.league_curves(conn, "SPY", "Live only", None).empty


def test_league_holdings_and_trades(conn):
    fill_league(conn)
    held = dd.league_holdings(conn, "hold-pair").set_index("symbol")
    assert held.loc["AAA", "value"] == pytest.approx(24_200) and held.loc["AAA", "return_since_entry"] == pytest.approx(0.21)
    assert held["share_of_bot"].sum() == pytest.approx(46_200 / 106_200)
    trades = dd.league_trades(conn, "hold-pair")
    assert set(trades["symbol"]) == {"AAA", "BBB"} and set(trades["side"]) == {"buy"}
    assert dd.league_holdings(conn, "nobody").empty and dd.league_trades(conn, "nobody").empty


def test_league_chart_colours_the_chosen_bot_and_greys_the_rest(conn):
    fill_league(conn)
    curves = dd.league_curves(conn, "SPY", "All history")
    for mode in ("light", "dark"):
        spec = league_chart(curves, "hold-pair", mode, dd.BENCHMARK_LABEL, pd.Timestamp("2026-11-12")).to_dict()
        scale = spec["layer"][0]["encoding"]["color"]["scale"]
        assert scale["domain"] == ["hold-pair", dd.BENCHMARK_LABEL, OTHERS]
        assert scale["range"] == [COLOURS[mode][0], COLOURS[mode][1], INK[mode]["muted"]]
        assert any(layer["mark"].get("strokeDash") for layer in spec["layer"] if isinstance(layer["mark"], dict))


# ---------------------------------------------------------------- runs and alerts

def test_run_history_lists_steps_and_names_the_failed_ones(conn):
    run_id = db.start_run(conn, "2026-11-13")
    db.finish_run(conn, run_id, "failed", json.dumps([
        {"name": "fetch prices", "code": 1, "seconds": 3.0, "output": "Fetch failed for XYZ"},
        {"name": "paper trading", "code": 0, "seconds": 5.0, "output": "No orders."}]))
    db.start_run(conn, "2026-11-16")                                  # still going
    running, failed = dd.run_history(conn)
    assert running["status"] == "running" and running["minutes"] is None and running["steps"] == []
    assert failed["failed_steps"] == ["fetch prices"] and failed["minutes"] is not None
    assert dd.run_in_progress(conn)
    conn.execute("UPDATE daily_runs SET started = '2026-11-01T00:00:00+00:00' WHERE status = 'running'")
    assert not dd.run_in_progress(conn)                               # a run left over from long ago does not block


def test_alerts_table_marks_unread_ones(conn):
    alerts.raise_alert(conn, "error", "Daily run failed", "details", channels={})
    db.mark_alerts_read(conn)
    alerts.raise_alert(conn, "warning", "Stop-loss: selling AAA", channels={})
    table = dd.alerts_table(conn)
    assert list(table["title"]) == ["Stop-loss: selling AAA", "Daily run failed"]
    assert list(table["new"]) == [True, False] and table["when"].notna().all()


# ---------------------------------------------------------------- the page

def open_page(path, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("TRADEBOT_DB", str(path))
    monkeypatch.setenv("TRADEBOT_REPORTS", str(tmp_path))
    page = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=60).run()
    assert not page.exception, [e.value for e in page.exception]
    return page


def button(page, label):
    return next(b for b in page.button if b.label == label)


def test_page_shows_the_league_runs_and_unread_alerts(tmp_path, monkeypatch):
    path = tmp_path / "full.db"
    conn = db.connect(str(path))
    db.init_db(conn)
    fill_league(conn)
    alerts.raise_alert(conn, "error", "Daily run 2026-11-13: problem in fetch prices", "exit code 1 costs $5", channels={})
    alerts.raise_alert(conn, "info", "Daily run 2026-11-12: all ok", channels={})
    run_id = db.start_run(conn, "2026-11-13")
    db.finish_run(conn, run_id, "failed", json.dumps([
        {"name": "fetch prices", "code": 1, "seconds": 3.0, "output": "Fetch failed for XYZ"}]))

    page = open_page(path, tmp_path, monkeypatch)
    assert [(e.proto.title, e.value) for e in page.error] == [
        ("Daily run 2026-11-13: problem in fetch prices", "exit code 1 costs \\$5")]     # the $ is escaped
    assert "Bot league" in [t.label for t in page.tabs] and "Runs and alerts" in [t.label for t in page.tabs]
    assert page.selectbox(key="league_bot").options == ["hold-aaa", "hold-pair"]
    assert page.segmented_control(key="league_period").value == "All history"
    page.segmented_control(key="league_period").set_value("Live only").run()
    assert not page.exception
    assert any("PROBLEM in fetch prices" in e.label for e in page.expander)
    assert any("Fetch failed for XYZ" in c.value for c in page.code)

    button(page, "Mark alerts as read").click().run()
    assert not page.exception and not page.error
    assert db.alerts_df(db.connect(str(path)), unread_only=True).empty


def test_pause_and_resume_buttons_set_the_switch(tmp_path, monkeypatch):
    path = tmp_path / "p.db"
    page = open_page(path, tmp_path, monkeypatch)
    button(page, "Pause buying").click().run()
    assert db.is_paused(db.connect(str(path)))
    assert any("Buying is paused" in w.value for w in page.warning)
    button(page, "Resume buying").click().run()
    assert not db.is_paused(db.connect(str(path)))


def test_sending_orders_needs_the_box_ticked(tmp_path, monkeypatch):
    page = open_page(tmp_path / "s.db", tmp_path, monkeypatch)
    assert button(page, "Send the bot's orders now").disabled
    assert button(page, "Start the full daily run").disabled
    page.checkbox[0].check().run()
    assert not button(page, "Send the bot's orders now").disabled


def test_the_trade_form_saves_a_manual_trade(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    page = open_page(path, tmp_path, monkeypatch)
    page.text_input[0].set_value("aapl")
    numbers = {n.label: n for n in page.number_input}
    numbers["Shares"].set_value(2.0)
    numbers["Price per share"].set_value(350.0)
    button(page, "Save trade").click().run()
    assert not page.exception
    saved = db.trades_df(db.connect(str(path)))
    assert len(saved) == 1
    row = saved.iloc[0]
    assert (row["account"], row["symbol"], row["side"], row["qty"], row["price"], row["notional"]) == (
        "fidelity_manual", "AAPL", "buy", 2.0, 350.0, 700.0)
    assert row["followed_ai"] == 1


def test_the_trade_form_refuses_an_empty_trade(tmp_path, monkeypatch):
    path = tmp_path / "e.db"
    page = open_page(path, tmp_path, monkeypatch)
    button(page, "Save trade").click().run()
    assert any("Fill in the symbol" in e.value for e in page.error)
    assert db.trades_df(db.connect(str(path))).empty
