"""Rank bots in the league, stocks that stop trading, data breaks, and universes read from a file."""
import importlib.util
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tradebot import config, db, league, modeldata, rankmodel
from tradebot import dashboard_data as dd
from tradebot.league import BotSpec, load_bots, simulate, target_table
from tradebot.risk import stop_loss_pct, target_notional

RISK = {"max_position_pct": 0.20, "max_total_exposure_pct": 1.0, "max_risk_per_position_pct": 0.015,
        "stop_cooldown_bars": 2, "daily_loss_limit_pct": 0.03, "min_order_notional": 1.0}
DAYS = pd.bdate_range("2026-11-02", periods=12)
NAMES = list("ABCDEF")


def flat(names=NAMES, days=DAYS, price=10.0):
    return pd.DataFrame(price, index=days, columns=names)


def rank_spec(top_n=2, rebalance_days=5, stop_loss=False, **risk):
    return BotSpec("rank", "rank (ridge)", "wide", tuple(NAMES), {}, {**RISK, **risk}, stop_loss,
                   None, "ridge", top_n, rebalance_days)


def scores_table(first, second, switch_at=5):
    """Scores that favour `first` until `switch_at`, then `second`."""
    table = pd.DataFrame(0.0, index=DAYS, columns=NAMES)
    table.loc[DAYS[:switch_at], first] = 1.0
    table.loc[DAYS[switch_at:], second] = 1.0
    return table


# ---------------------------------------------------------------- rank bots

def test_a_rank_bot_holds_the_top_scores_in_equal_parts():
    prices = flat()
    result = simulate(prices, prices, scores_table(["A", "B"], ["A", "B"]), rank_spec(), 100_000, cost_bps=0)
    assert set(result.positions) == {"A", "B"}
    assert [round(p["value"]) for p in result.positions.values()] == [20_000, 20_000]   # 1/2 each, capped at 20%
    assert {t[0] for t in result.trades} == {"2026-11-03"}                               # bought once, then held


def test_a_rank_bot_re_picks_only_on_its_schedule():
    prices = flat()
    # The scores change on day index 3, but the bot picked on day 0 and looks again on day 5.
    result = simulate(prices, prices, scores_table(["A", "B"], ["C", "D"], switch_at=3), rank_spec(), 100_000, 0)
    by_day = {}
    for date, symbol, side, *_ in result.trades:
        by_day.setdefault(date, set()).add((side, symbol))
    assert by_day == {"2026-11-03": {("buy", "A"), ("buy", "B")},                        # picked day 0, filled day 1
                      "2026-11-10": {("sell", "A"), ("sell", "B"), ("buy", "C"), ("buy", "D")}}   # day 5 -> day 6
    assert set(result.positions) == {"C", "D"}


def test_a_stopped_out_pick_is_sold_between_re_picks_and_not_bought_back_at_once():
    opens, closes = flat(), flat()
    closes.loc[DAYS[2]:, "A"] = 9.0                       # A falls 10% from its entry and stays there
    opens.loc[DAYS[3]:, "A"] = 9.0
    spec = rank_spec(stop_loss=True, stop_cooldown_bars=20)
    result = simulate(opens, closes, scores_table(["A", "B"], ["A", "B"]), spec, 100_000, 0)
    sells = [(t[0], t[1], t[6]) for t in result.trades if t[2] == "sell"]
    assert sells == [("2026-11-05", "A", "stop-loss")]                     # day 2's stop, filled at day 3's open
    # At the re-pick on day 5, A still has the top score but is in its cooldown: the next best takes its place.
    assert "A" not in result.positions and len(result.positions) == 2 and "B" in result.positions


def test_a_rank_bot_waits_for_scores_and_skips_stocks_with_no_bar():
    prices = flat()
    prices.loc[:, "A"] = np.nan                                            # A has the best score but never trades
    scores = scores_table(["A", "B", "C"], ["A", "B", "C"])
    scores.iloc[:3] = np.nan                                               # no scores for the first three days
    result = simulate(prices, prices, scores, rank_spec(), 100_000, 0)
    assert result.trades[0][0] == "2026-11-06" and set(result.positions) == {"B", "C"}


# ---------------------------------------------------------------- stocks that stop trading

def test_a_holding_that_stops_trading_is_paid_out_at_its_last_price():
    prices = flat(names=["A", "B"])
    prices.loc[DAYS[4]:, "A"] = np.nan                                     # A's last bar is day index 3
    hold = BotSpec("hold", "buy_and_hold", "custom", ("A", "B"), {}, RISK, False)
    result = simulate(prices, prices, target_table(prices, hold), hold, 100_000, 0)
    paid = [t for t in result.trades if t[6] == "stopped trading"]
    assert [(t[0], t[1], t[5]) for t in paid] == [("2026-11-12", "A", 20_000.0)]   # five days with no bar
    assert "A" not in result.positions and result.equity.iloc[-1] == pytest.approx(100_000)
    assert result.cash.iloc[-1] == pytest.approx(80_000)                   # A's money came back as cash


def test_a_rule_bot_stops_wishing_for_a_stock_that_is_gone_but_rides_out_a_short_gap():
    prices = flat(names=["A", "B"])
    prices.loc[DAYS[4]:, "A"] = np.nan
    prices.loc[DAYS[6], "B"] = np.nan                                      # one missing day
    hold = BotSpec("hold", "buy_and_hold", "custom", ("A", "B"), {}, RISK, False)
    wishes = target_table(prices, hold)
    assert wishes["A"].notna().tolist() == [True] * 8 + [False] * 4        # 4 days of bars + 4 days of grace
    assert wishes["B"].notna().all()


# ---------------------------------------------------------------- data breaks

def test_a_series_is_cut_at_its_first_impossible_move_and_left_alone_before_it():
    days = pd.bdate_range("2026-01-05", periods=6)
    close = pd.DataFrame({"REAL": [10, 12, 9, 10, 11, 12.0],               # big but possible moves
                          "RELIST": [1.0, 0.5, 0.4, 30, 31, 32],           # new shares take over the ticker
                          "SPIN": [100, 101, 15, 16, 16, 17.0],            # an unadjusted split or spin-off
                          "GAP": [10, np.nan, np.nan, 40, 41, 42.0]}, index=days)
    panels = {field: close.copy() for field in ("open", "high", "low", "close", "volume")}
    cut, breaks = modeldata.cut_at_breaks(panels, ratio=3.0)
    assert dict(zip(breaks["symbol"], breaks["date"].dt.strftime("%Y-%m-%d"))) == {
        "RELIST": "2026-01-08", "SPIN": "2026-01-07", "GAP": "2026-01-08"}  # GAP: measured across its missing days
    assert cut["close"]["REAL"].tolist() == close["REAL"].tolist()
    assert cut["close"]["RELIST"].tolist()[:3] == [1.0, 0.5, 0.4] and cut["close"]["RELIST"].iloc[3:].isna().all()
    assert cut["open"]["SPIN"].iloc[2:].isna().all() and cut["volume"]["SPIN"].iloc[:2].notna().all()
    assert breaks.set_index("symbol").loc["RELIST", "move"] == pytest.approx(75.0)


def test_no_ratio_means_no_cutting():
    conn = db.connect(":memory:")
    db.init_db(conn)
    assert modeldata.load_clean_panels(conn, 3.0)[0] == {}
    db.upsert_prices(conn, pd.DataFrame({"symbol": "A", "date": ["2026-01-05", "2026-01-06"], "open": [1, 50.0],
                                         "high": [1, 50.0], "low": [1, 50.0], "close": [1, 50.0], "volume": 1.0}))
    assert modeldata.load_clean_panels(conn, None)[0]["close"]["A"].tolist() == [1.0, 50.0]
    cut, breaks = modeldata.load_clean_panels(conn, 3.0)
    assert np.isnan(cut["close"]["A"].iloc[1]) and list(breaks["symbol"]) == ["A"]


# ---------------------------------------------------------------- config and a whole league run

CFG = {"benchmark": "SPY", "start_date": "2026-11-01", "backtest": {"cost_bps": 0}, "risk": RISK,
       "strategies": {},
       "universes": {"pair": ["SPY", "QQQ"], "wide": NAMES},
       "model": {"universe": "wide", "prices_path": ":memory:", "horizon_days": 5, "first_test_year": 2026,
                 "top_n": 2, "break_ratio": 3.0, "models": {"ridge": {"alpha": 1.0}}},
       "league": {"start_cash": 100_000, "history_start": "2026-11-02",
                  "bots": [{"name": "hold-pair", "strategy": "buy_and_hold", "universe": "pair"},
                           {"name": "rank-ridge", "model": "ridge", "universe": "wide", "start": "2026-11-04"},
                           {"name": "hold-wide", "strategy": "buy_and_hold", "universe": "wide",
                            "stop_loss": False, "start": "2026-11-04"}]}}


def test_rank_bots_take_their_settings_from_the_model_section():
    hold, rank, wide = load_bots(CFG)
    assert (rank.model, rank.top_n, rank.rebalance_days, rank.strategy) == ("ridge", 2, 5, "rank (ridge)")
    assert rank.start == "2026-11-04" and wide.start == "2026-11-04" and hold.start is None
    assert load_bots({**CFG, "league": {"bots": [{"name": "r", "model": "ridge", "universe": "wide"}]}})[0].start \
        == "2026-01-01"                                    # by default, the first year the model was scored
    assert league.league_symbols(CFG) == ["SPY", "QQQ"]    # the wide universe's prices are fetched separately
    # the model's own settings are part of a rank bot: changing them restarts its live record
    changed = {**CFG, "model": {**CFG["model"], "models": {"ridge": {"alpha": 5.0}}}}
    marks = [load_bots(c)[1].fingerprint(100_000, 0, "2026-11-02") for c in (CFG, changed)]
    assert marks[0] != marks[1]


@pytest.mark.parametrize("entry, message", [
    ({"name": "x", "model": "nope", "universe": "wide"}, "not listed under `model.models`"),
    ({"name": "x", "model": "ridge", "universe": "pair"}, "must use the model's universe"),
])
def test_rank_bot_mistakes_are_reported_in_plain_words(entry, message):
    with pytest.raises(ValueError, match=message):
        load_bots({**CFG, "league": {"bots": [entry]}})


def store(conn, symbol, closes, days=DAYS):
    dates = [d.strftime("%Y-%m-%d") for d in days[:len(closes)]]
    db.upsert_prices(conn, pd.DataFrame({"symbol": symbol, "date": dates, "open": closes, "high": closes,
                                         "low": closes, "close": closes, "volume": 1.0}))


@pytest.fixture
def two_databases(tmp_path):
    main, model = db.connect(str(tmp_path / "main.db")), db.connect(str(tmp_path / "model.db"))
    db.init_db(main), db.init_db(model)
    for symbol in ("SPY", "QQQ"):
        store(main, symbol, [10.0] * 12)
    for symbol in NAMES:
        store(model, symbol, [10.0] * 12)
    store(model, "F", [10.0, 10.0, 10.0, 500.0] + [500.0] * 8)            # a data break on the fourth day
    scores = scores_table(["A", "B"], ["A", "B"]).iloc[2:]                # scored from 2026-11-04
    rows = scores.stack().reset_index()
    rows.columns = ["date", "symbol", "score"]
    rows["date"] = rows["date"].dt.strftime("%Y-%m-%d")
    rankmodel.save_scores(model, "ridge", rows)
    return main, model


def test_a_league_run_with_rank_bots_reads_the_models_prices_and_scores(two_databases):
    main, model = two_databases
    said = []
    board = league.run_league(main, CFG, said.append, model_conn=model).set_index("bot")
    assert list(board.index) == ["hold-pair", "rank-ridge", "hold-wide"]
    assert board.loc["rank-ridge", "strategy"] == "rank (ridge)" and board.loc["rank-ridge", "symbols"] == 6
    assert (board.loc["hold-pair", "since"], board.loc["rank-ridge", "since"]) == ("2026-11-02", "2026-11-04")
    held = dict(main.execute("SELECT symbol, value FROM league_positions WHERE bot = 'rank-ridge'"))
    assert held == {"A": pytest.approx(20_000), "B": pytest.approx(20_000)}
    # The break in F is cut out: the baseline never books F's 50-fold "gain".
    assert board.loc["hold-wide", "value"] == pytest.approx(100_000)
    assert "F" not in dict(main.execute("SELECT symbol, value FROM league_positions WHERE bot = 'hold-wide'"))


def test_a_rank_bot_without_scores_is_reported_and_skipped(two_databases):
    main, model = two_databases
    model.execute("DELETE FROM model_scores")
    said = []
    board = league.run_league(main, CFG, said.append, model_conn=model)
    assert "rank-ridge" not in list(board["bot"]) and "hold-wide" in list(board["bot"])
    assert any("no scores yet" in line for line in said)


def test_the_chart_starts_where_the_chosen_bot_does_and_leaves_out_later_starters(two_databases):
    main, model = two_databases
    league.run_league(main, CFG, lambda line: None, model_conn=model)
    early = dd.league_curves(main, "SPY", "All history", None, "hold-pair")
    assert early["date"].iloc[0] == pd.Timestamp("2026-11-02")
    assert list(early.columns) == ["date", "hold-pair", dd.BENCHMARK_LABEL]         # the others had not started
    late = dd.league_curves(main, "SPY", "All history", None, "rank-ridge")
    assert late["date"].iloc[0] == pd.Timestamp("2026-11-04")
    assert set(late.columns) == {"date", "hold-pair", "hold-wide", "rank-ridge", dd.BENCHMARK_LABEL}
    assert late.iloc[0, 1:].tolist() == [100_000.0] * 4                            # all level on the first day


# ---------------------------------------------------------------- universes from a file

def test_a_universe_can_be_a_file_of_tickers(tmp_path, monkeypatch):
    (tmp_path / "list.txt").write_text("# a comment\n\naapl\n  BRK.B \n# another\nMSFT\n", encoding="utf-8")
    assert config.read_tickers(tmp_path / "list.txt") == ["AAPL", "BRK.B", "MSFT"]
    (tmp_path / "config.yaml").write_text(
        "db_path: data/x.db\nuniverses:\n  core: [SPY]\n  wide: 'file:list.txt'\n"
        "model: {prices_path: data/model.db}\n", encoding="utf-8")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    cfg = config.load_config(tmp_path / "config.yaml")
    assert cfg["universes"] == {"core": ["SPY"], "wide": ["AAPL", "BRK.B", "MSFT"]}
    assert cfg["model"]["prices_path"] == str((tmp_path / "data" / "model.db").resolve())


def test_the_real_config_loads_with_the_wide_universe_and_valid_league_bots():
    cfg = config.load_config()
    wide = cfg["universes"][cfg["model"]["universe"]]
    assert len(wide) > 400 and "AAPL" in wide and len(set(wide)) == len(wide)
    bots = {bot.name: bot for bot in load_bots(cfg)}
    assert bots["rank-ridge"].top_n == cfg["model"]["top_n"] and bots["rank-gbm"].model == "gbm"
    assert bots["hold-sp500"].universe == cfg["model"]["universe"]


def test_the_sized_rank_bots_differ_from_their_twins_only_in_the_stop():
    bots = {bot.name: bot for bot in load_bots(config.load_config())}
    for name in ("rank-ridge", "rank-gbm", "rank-ridge-fund", "rank-gbm-fund"):
        plain, sized = bots[name], bots[f"{name}-sized"]
        assert (sized.model, sized.top_n, sized.rebalance_days, sized.symbols, sized.start, sized.stop_loss) == \
               (plain.model, plain.top_n, plain.rebalance_days, plain.symbols, plain.start, plain.stop_loss)
        assert {k: v for k, v in sized.risk.items() if k != "max_position_pct"} == \
               {k: v for k, v in plain.risk.items() if k != "max_position_pct"}
        assert stop_loss_pct(plain.risk) == pytest.approx(0.075) and stop_loss_pct(sized.risk) == pytest.approx(0.375)
        # Both put 1/top_n of the account in each stock: the 4% cap equals that weight, so it never binds.
        assert target_notional(100_000, 1.0, plain.top_n, sized.risk) == \
               pytest.approx(target_notional(100_000, 1.0, plain.top_n, plain.risk))


# ---------------------------------------------------------------- what the stop and the costs take

def test_the_replay_variants_change_only_the_stop_and_the_costs():
    spec = rank_spec(stop_loss=True)
    out = league.variants(spec, 5.0)
    assert list(out) == ["as set", "no stop", "no stop or costs"]
    assert out == {"as set": (spec, 5.0), "no stop": (replace(spec, stop_loss=False), 5.0),
                   "no stop or costs": (replace(spec, stop_loss=False), 0.0)}


def test_beta_and_the_return_left_after_it():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2026-01-05", periods=200)
    base = pd.Series(rng.normal(0, 0.01, 200), index=days)
    double = league.against(2 * base, base)
    assert double["beta"] == pytest.approx(2.0) and double["alpha"] == pytest.approx(0.0, abs=1e-12)
    lifted = league.against(base + 0.001 + pd.Series(rng.normal(0, 0.001, 200), index=days), base)
    assert lifted["beta"] == pytest.approx(1.0, abs=0.05)
    assert lifted["alpha"] == pytest.approx(0.252, abs=0.05) and lifted["t_stat"] > 5   # 0.1% a day, 252 days


def attribution_script():
    path = config.ROOT / "scripts" / "league_attribution.py"
    spec = importlib.util.spec_from_file_location("league_attribution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_attribution_script_replays_each_rank_bot_with_and_without_its_stop(two_databases, capsys):
    _, model = two_databases
    # A, a top pick from 2026-11-04, falls 10% two days after it is bought, then recovers to 11.
    store(model, "A", [10.0, 10.0, 10.0, 10.0, 9.0, 9.0, 10.5, 11.0, 11.0, 11.0, 11.0, 11.0])
    script = attribution_script()
    table, base = script.replay(CFG, model)
    rows = table.set_index("variant")
    assert list(table["bot"].unique()) == ["rank-ridge"] and base["name"] == "hold-wide"
    assert rows["stop_outs"].to_dict() == {"as set": 1, "no stop": 0, "no stop or costs": 0}
    assert rows.loc["no stop", "total_return"] > rows.loc["as set", "total_return"]   # sold at the low
    assert f"{base['first']:%Y-%m-%d}" == "2026-11-04"
    _, cut = script.replay(CFG, model, end="2026-11-10")
    assert f"{cut['last']:%Y-%m-%d}" == "2026-11-10"
    script.show(table, base)
    shown = capsys.readouterr().out
    assert "Rank bots against hold-wide, 2026-11-04 to 2026-11-17" in shown and "no stop or costs" in shown


# ---------------------------------------------------------------- the commands

def test_the_model_commands_are_wired_into_the_command_line():
    import main
    parser = main.build_parser()
    assert parser.parse_args(["model-fetch", "--full"]).full is True
    assert parser.parse_args(["model-score"]).rebuild is False
    assert parser.parse_args(["model-test", "--rebuild"]).rebuild is True
    for name in ("model-fetch", "model-score", "model-test", "league"):
        assert callable(main.COMMANDS[name])


def test_the_daily_run_trades_before_it_touches_the_model():
    from tradebot.daily import DAILY
    order = [step.args[0] for step in DAILY if not step.weekly]
    assert order.index("paper") < order.index("model-fetch") < order.index("model-score") < order.index("league")
    assert order.index("model-fetch") < order.index("fundamentals-fetch") < order.index("model-score")
    assert [step.args[0] for step in DAILY if step.weekly][-1] == "model-test"
