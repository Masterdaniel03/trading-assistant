"""The bot league: the simulated fills, on prices small enough to check by hand."""
import numpy as np
import pandas as pd
import pytest

from tradebot import db, league
from tradebot.league import BotSpec, load_bots, simulate, target_table

RISK = {"max_position_pct": 0.20, "max_total_exposure_pct": 1.0, "max_risk_per_position_pct": 0.015,
        "stop_cooldown_bars": 2, "daily_loss_limit_pct": 0.03, "min_order_notional": 1.0}
DAYS = pd.to_datetime(["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-16"])


def panel(**columns):
    return pd.DataFrame(columns, index=DAYS[:len(next(iter(columns.values())))], dtype=float)


def spec(symbols, strategy="buy_and_hold", stop_loss=True, **risk):
    return BotSpec("test", strategy, "custom", tuple(symbols), {}, {**RISK, **risk}, stop_loss)


def always(closes, value=1.0):
    return pd.DataFrame(value, index=closes.index, columns=closes.columns)


def test_an_order_decided_at_the_close_fills_at_the_next_open():
    opens, closes = panel(A=[10, 11, 12]), panel(A=[10, 12, 12])
    result = simulate(opens, closes, always(closes), spec(["A"]), 100_000, cost_bps=0)
    # Day 1: decides to buy 20% = $20,000. Day 2: fills at the open of 11, not at the close of 10.
    assert result.trades == [("2026-11-10", "A", "buy", pytest.approx(20_000 / 11), 11.0, 20_000.0, None)]
    assert result.equity.iloc[0] == 100_000                      # nothing held on the first day
    assert result.equity.iloc[1] == pytest.approx(80_000 + 20_000 / 11 * 12)   # shares valued at the close


def test_every_trade_pays_the_cost():
    opens, closes = panel(A=[10, 10, 10]), panel(A=[10, 10, 10])
    result = simulate(opens, closes, always(closes), spec(["A"]), 100_000, cost_bps=10)
    assert result.equity.iloc[-1] == pytest.approx(100_000 - 20_000 * 0.001)   # flat prices: only the cost is lost


def test_position_size_is_equal_weight_capped_at_the_limit():
    one = panel(A=[10, 10, 10])
    many = panel(**{name: [10, 10, 10] for name in "ABCDEFGHIJ"})
    assert simulate(one, one, always(one), spec(["A"]), 100_000, 0).positions["A"]["value"] == pytest.approx(20_000)
    result = simulate(many, many, always(many), spec(list("ABCDEFGHIJ")), 100_000, 0)
    assert {round(p["value"]) for p in result.positions.values()} == {10_000}   # ten symbols: 10% each


def test_a_stop_loss_sells_at_the_next_open_and_then_stays_out():
    #                  buy at 100   close 92 (-8%)  sold at 90    cooldown     may buy again
    opens = panel(A=[100, 100, 100, 90, 95, 95])
    closes = panel(A=[100, 100, 92, 90, 95, 95])
    result = simulate(opens, closes, always(closes), spec(["A"]), 100_000, cost_bps=0)
    trades = [(t[0], t[2], t[4], t[6]) for t in result.trades]
    assert trades[0] == ("2026-11-10", "buy", 100.0, None)
    assert trades[1] == ("2026-11-12", "sell", 90.0, "stop-loss")        # the gap makes the loss 10%, not 7.5%
    assert result.equity.iloc[3] == pytest.approx(100_000 - 200 * 10)    # 200 shares, $10 lost on each
    # stopped on day index 2 with a 2-bar cooldown: no wish on days 2 and 3, a new order on day 4, filled day 5
    assert trades[2:] == [("2026-11-16", "buy", 95.0, None)]


def test_no_stop_loss_when_switched_off():
    opens = panel(A=[100, 100, 100, 90, 95, 95])
    closes = panel(A=[100, 100, 92, 90, 95, 95])
    result = simulate(opens, closes, always(closes), spec(["A"], stop_loss=False), 100_000, 0)
    assert [t[2] for t in result.trades].count("sell") == 0


def test_a_symbol_with_no_bar_keeps_its_last_price_and_its_order_lapses():
    opens = panel(A=[10, 10, 10, 10], B=[10, np.nan, 10, 10])
    closes = panel(A=[10, 10, 10, 10], B=[10, np.nan, 12, 12])
    result = simulate(opens, closes, always(closes), spec(["A", "B"]), 100_000, 0)
    by_day = {(t[0], t[1]) for t in result.trades}
    assert ("2026-11-10", "A") in by_day and ("2026-11-10", "B") not in by_day    # B had no bar that morning
    assert ("2026-11-11", "B") in by_day                                           # planned again, then filled
    assert not result.equity.isna().any()


def test_a_symbol_is_ignored_until_its_first_price():
    closes = panel(A=[10, 10, 10, 10], B=[np.nan, np.nan, 10, 10])
    targets = target_table(closes, spec(["A", "B"]))
    assert targets["B"].isna().tolist() == [True, True, False, False]
    result = simulate(closes, closes, targets, spec(["A", "B"]), 100_000, 0)
    assert [t[1] for t in result.trades] == ["A", "B"]


def test_buys_never_spend_more_than_the_cash():
    closes = panel(**{name: [10, 10, 10] for name in "ABCDE"})
    result = simulate(closes, closes, always(closes), spec(list("ABCDE")), 100_000, cost_bps=50)
    assert result.cash.min() >= -1e-6                              # five 20% buys plus costs: the last is trimmed


def test_start_date_limits_the_replay_but_not_the_indicator_history():
    closes = panel(A=[10, 11, 12, 13, 14, 15])
    targets = target_table(closes, BotSpec("m", "momentum", "custom", ("A",), {"lookback": 2, "skip": 0}, RISK))
    assert targets["A"].tolist() == [0, 0, 1, 1, 1, 1]
    result = simulate(closes, closes, targets, spec(["A"]), 100_000, 0, start="2026-11-12")
    assert result.equity.index[0] == pd.Timestamp("2026-11-12")
    assert result.trades[0][0] == "2026-11-13"                     # already in a trend on the first day


# ---------------------------------------------------------------- config and storage

CFG = {"benchmark": "SPY", "start_date": "2026-11-01", "backtest": {"cost_bps": 0}, "risk": RISK,
       "strategies": {"momentum": {"lookback": 2, "skip": 0}},
       "universes": {"pair": ["AAA", "BBB"]},
       "league": {"start_cash": 100_000, "history_start": "2026-11-09",
                  "bots": [{"name": "hold-pair", "strategy": "buy_and_hold", "universe": "pair"},
                           {"name": "mom-aaa", "strategy": "momentum", "symbols": ["AAA"],
                            "params": {"lookback": 3}, "risk": {"max_position_pct": 0.5}}]}}


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "league.db"))
    db.init_db(c)
    return c


def store(conn, symbol, closes, days=DAYS):
    dates = [d.strftime("%Y-%m-%d") for d in days[:len(closes)]]
    db.upsert_prices(conn, pd.DataFrame({"symbol": symbol, "date": dates, "open": closes, "high": closes,
                                         "low": closes, "close": closes, "volume": 1.0}))


def test_bots_take_symbols_settings_and_limits_from_the_config():
    hold, mom = load_bots(CFG)
    assert hold.symbols == ("AAA", "BBB") and hold.universe == "pair" and hold.stop_loss
    assert mom.params == {"lookback": 3, "skip": 0}                # the bot's own setting wins
    assert mom.risk["max_position_pct"] == 0.5 and mom.risk["stop_cooldown_bars"] == 2
    assert league.league_symbols(CFG) == ["AAA", "BBB"]


@pytest.mark.parametrize("bots, message", [
    ([{"name": "x", "strategy": "sma_crossover", "universe": "nope"}], "not listed"),
    ([{"name": "x", "strategy": "made_up", "universe": "pair"}], "Unknown strategy"),
    ([{"name": "x", "strategy": "momentum", "symbols": ["BTC-USD"]}], "crypto"),
    ([{"name": "x", "strategy": "momentum", "universe": "pair"}] * 2, "both called"),
])
def test_config_mistakes_are_reported_in_plain_words(bots, message):
    with pytest.raises(ValueError, match=message):
        load_bots({**CFG, "league": {"bots": bots}})


def test_a_run_stores_values_trades_holdings_and_a_leaderboard(conn):
    for symbol in ("AAA", "BBB", "SPY"):
        store(conn, symbol, [10, 10, 10, 11])
    said = []
    board = league.run_league(conn, CFG, said.append)
    assert list(board["bot"]) == ["hold-pair", "mom-aaa"]
    row = board.iloc[0]
    assert row["symbols"] == 2 and row["live_since"] == "2026-11-12" and row["live_days"] == 0
    assert row["value"] == pytest.approx(100_000 + 2 * 2_000)       # two $20,000 positions, each up 10%
    assert row["invested"] == pytest.approx(44_000 / 104_000)
    assert any("new bot" in line for line in said)
    assert conn.execute("SELECT COUNT(*) FROM league_trades WHERE bot = 'hold-pair'").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM league_positions WHERE bot = 'hold-pair'").fetchone()[0] == 2


def test_days_after_the_first_run_count_as_live_and_a_replay_changes_nothing(conn):
    for symbol in ("AAA", "BBB", "SPY"):
        store(conn, symbol, [10, 10, 10, 11])
    league.run_league(conn, CFG, lambda line: None)
    for symbol, last in (("AAA", 12.1), ("BBB", 11), ("SPY", 11.55)):
        store(conn, symbol, [10, 10, 10, 11, last])
    board = league.run_league(conn, CFG, lambda line: None).set_index("bot")
    row = board.loc["hold-pair"]
    assert row["live_since"] == "2026-11-12" and row["live_days"] == 1
    assert row["live_return"] == pytest.approx((106_200 / 104_000) - 1)   # AAA +10% on $22,000, BBB flat
    assert row["spy_live_return"] == pytest.approx(0.05)
    again = league.run_league(conn, CFG, lambda line: None).set_index("bot")
    pd.testing.assert_frame_equal(board, again)


def test_changing_a_bot_restarts_its_live_record_and_removing_one_deletes_it(conn):
    for symbol in ("AAA", "BBB", "SPY"):
        store(conn, symbol, [10, 10, 10, 11])
    league.run_league(conn, CFG, lambda line: None)
    for symbol in ("AAA", "BBB", "SPY"):
        store(conn, symbol, [10, 10, 10, 11, 11])
    changed = {**CFG, "league": {**CFG["league"], "bots": [
        {"name": "hold-pair", "strategy": "buy_and_hold", "universe": "pair", "stop_loss": False}]}}
    said = []
    board = league.run_league(conn, changed, said.append)
    assert list(board["bot"]) == ["hold-pair"] and board.iloc[0]["live_since"] == "2026-11-13"
    assert any("settings changed" in line for line in said) and any("removed" in line for line in said)
    assert conn.execute("SELECT COUNT(*) FROM league_equity WHERE bot = 'mom-aaa'").fetchone()[0] == 0


def test_missing_prices_are_reported_and_an_empty_league_is_fine(conn):
    said = []
    assert league.run_league(conn, CFG, said.append).empty
    assert any("no prices for AAA, BBB" in line for line in said)
    assert league.run_league(conn, {**CFG, "league": {"bots": []}}, said.append).empty
