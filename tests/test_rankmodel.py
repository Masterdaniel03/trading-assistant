"""The ranking model: features that never look ahead, a walk-forward that never trains on its test year."""
import numpy as np
import pandas as pd
import pytest

from tradebot import db, features, modeldata, rankmodel

HORIZON = 5


def make_panels(days=700, symbols=60, seed=0, signal=0.0):
    """Random daily bars. With `signal`, a stock's coming week is helped by its past month (momentum)."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2016-01-04", periods=days)
    names = [f"S{i:02d}" for i in range(symbols)]
    returns = rng.normal(0, 0.02, (days, symbols))
    if signal:
        for t in range(22, days):
            past = returns[t - 21:t].sum(axis=0)
            returns[t] += signal * (past - past.mean())
    close = 100 * np.exp(np.cumsum(returns, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.003, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, close.shape)))
    volume = rng.lognormal(13, 0.5, close.shape)
    frame = lambda values: pd.DataFrame(values, index=index, columns=names)  # noqa: E731
    return {"open": frame(open_), "high": frame(high), "low": frame(low), "close": frame(close),
            "volume": frame(volume)}


# ---------------------------------------------------------------- features

def test_every_feature_is_computed_and_named():
    out = features.compute(make_panels(days=300))
    assert list(out) == features.NAMES and len(features.NAMES) == 25
    last = pd.DataFrame({name: frame.iloc[-1] for name, frame in out.items()})
    assert last.notna().all().all()                       # 300 days is enough history for every feature
    assert out["money_flow_14"].iloc[-1].between(0, 1).all()
    assert out["rsi_14"].iloc[-1].between(0, 100).all()
    assert out["close_loc_5d"].iloc[-1].between(0, 1).all()


def test_features_never_use_later_days():
    """Cutting the data off after a day must leave that day's features unchanged."""
    panels = make_panels(days=320)
    full = features.compute(panels)
    for cut in (270, 300):
        short = features.compute({k: v.iloc[:cut + 1] for k, v in panels.items()})
        for name in features.NAMES:
            pd.testing.assert_series_equal(short[name].iloc[cut], full[name].iloc[cut], check_names=False)


def test_known_values_on_a_hand_made_series():
    index = pd.bdate_range("2026-01-05", periods=6)
    panels = {k: pd.DataFrame({"A": v}, index=index, dtype=float) for k, v in {
        "open": [10, 10, 11, 12, 12, 12], "high": [11, 12, 12, 13, 13, 13],
        "low": [9, 10, 10, 11, 11, 11], "close": [10, 11, 12, 12, 12, 13], "volume": [100] * 6}.items()}
    out = features.compute(panels)
    assert out["ret_1d"]["A"].iloc[1] == pytest.approx(0.10)
    assert out["ret_5d"]["A"].iloc[5] == pytest.approx(0.30)
    # closes sat at 50%, 100%, 50%, 50%, 100% of the day's range on the last five days
    assert out["close_loc_5d"]["A"].iloc[5] == pytest.approx((0.5 + 1 + 0.5 + 0.5 + 1) / 5)
    # bodies: (11-10)/2, (12-11)/2, 0, 0, (13-12)/2
    assert out["body_5d"]["A"].iloc[5] == pytest.approx((0.5 + 0.5 + 0 + 0 + 0.5) / 5)


def test_a_stock_with_no_bar_has_no_features_that_day():
    panels = make_panels(days=300)
    for frame in panels.values():
        frame.iloc[250:, 0] = np.nan                      # the first stock stops trading
    out = features.compute(panels)
    assert all(frame.iloc[260, 0] != frame.iloc[260, 0] for frame in out.values())      # NaN
    assert out["ret_21d"].iloc[260, 1:].notna().all()


def test_ranks_run_from_minus_half_to_plus_half_and_skip_missing_values():
    ranked = features.rank_rows(pd.DataFrame([[1.0, 5.0, np.nan, 3.0]]))
    assert ranked.iloc[0].tolist()[:2] == [pytest.approx(1 / 3 - 0.5), pytest.approx(0.5)]
    assert np.isnan(ranked.iloc[0, 2])


def test_forward_return_starts_at_the_next_open():
    opens = pd.DataFrame({"A": [10.0, 20.0, 21.0, 22.0, 30.0]})
    forward = features.forward_return(opens, horizon=2)
    assert forward["A"].iloc[0] == pytest.approx(22 / 20 - 1)     # buy at open 1, sell at open 3
    assert forward["A"].iloc[1] == pytest.approx(30 / 21 - 1)
    assert forward["A"].iloc[2:].isna().all()                     # the outcome is not known yet


# ---------------------------------------------------------------- walk-forward

def test_no_fold_trains_on_outcomes_from_its_own_test_year():
    data = rankmodel.build_dataset(make_panels(), HORIZON)
    plan = rankmodel.folds(data, 2017)
    assert [fold["year"] for fold in plan] == [2017, 2018]
    for fold in plan:
        last_outcome_day = data.days[fold["last_train"] + 1 + HORIZON]      # where the last training outcome ends
        assert last_outcome_day.year == fold["year"] - 1
        assert last_outcome_day == data.days[data.days.year < fold["year"]][-1]
        assert data.days[fold["test"][0]].year == fold["year"]
    assert len(rankmodel.audit(plan)) == 2


def test_the_audit_stops_a_fold_that_leaks():
    data = rankmodel.build_dataset(make_panels(), HORIZON)
    fold = rankmodel.folds(data, 2017)[0]
    fold["trained_through"] = fold["first_test_day"]
    with pytest.raises(AssertionError):
        rankmodel.audit([fold])


@pytest.mark.parametrize("name, settings", [("ridge", {"alpha": 1.0}),
                                            ("gbm", {"max_iter": 30, "min_samples_leaf": 50})])
def test_a_planted_signal_is_found_and_noise_is_not(name, settings):
    conn = db.connect(":memory:")
    db.init_db(conn)
    cfg = {"first_test_year": 2017, "models": {name: settings}}
    found = {}
    for label, signal in (("signal", 0.03), ("noise", 0.0)):
        conn.execute("DELETE FROM model_scores")
        data = rankmodel.build_dataset(make_panels(signal=signal, seed=1), HORIZON)
        rankmodel.update_scores(conn, data, cfg, None, lambda line: None)
        found[label] = rankmodel.evaluate(rankmodel.load_scores(conn, name), data, top_n=10, cost_bps=5)
    assert found["signal"]["ic"]["mean"] > 0.05 and found["signal"]["ic"]["t_stat"] > 3
    assert abs(found["noise"]["ic"]["t_stat"]) < 3
    assert found["signal"]["quintiles"][4] > found["signal"]["quintiles"][0]      # top fifth beats bottom fifth
    assert found["signal"]["top_gross"]["per_year"] > 0


def test_stored_scores_are_never_overwritten_and_only_missing_days_are_scored(tmp_path):
    conn = db.connect(":memory:")
    db.init_db(conn)
    cfg = {"first_test_year": 2017, "models": {"ridge": {"alpha": 1.0}}}
    panels = make_panels()
    said = []
    earlier = rankmodel.build_dataset({k: v.iloc[:-3] for k, v in panels.items()}, HORIZON)
    rankmodel.update_scores(conn, earlier, cfg, tmp_path, said.append)
    first = rankmodel.load_scores(conn, "ridge")
    conn.execute("UPDATE model_scores SET score = 99 WHERE date = ?", (first.index[0].strftime("%Y-%m-%d"),))

    data = rankmodel.build_dataset(panels, HORIZON)
    said.clear()
    added = rankmodel.update_scores(conn, data, cfg, tmp_path, said.append)
    after = rankmodel.load_scores(conn, "ridge")
    assert added["ridge"] == 3 * 60 and len(after) == len(first) + 3      # three new days, sixty stocks each
    assert (after.iloc[0] == 99).all()                                    # the edited day was left alone
    assert len(said) == 1 and "2018" in said[0]                           # 2017 was complete: not trained again
    assert len(list(tmp_path.glob("ridge_2018_*.joblib"))) == 1           # the year's model is saved and reused
    assert rankmodel.update_scores(conn, data, cfg, tmp_path, said.append) == {"ridge": 0}


def test_scores_for_a_day_do_not_depend_on_later_prices(tmp_path):
    """Scoring day t with data up to t, or with data far past t, must give the same numbers."""
    panels = make_panels(signal=0.02)
    cfg = {"first_test_year": 2018, "models": {"ridge": {"alpha": 1.0}}}
    results = []
    for cut in (600, 700):
        conn = db.connect(":memory:")
        db.init_db(conn)
        data = rankmodel.build_dataset({k: v.iloc[:cut] for k, v in panels.items()}, HORIZON)
        rankmodel.update_scores(conn, data, cfg, None, lambda line: None)
        results.append(rankmodel.load_scores(conn, "ridge"))
    shared = results[0].index
    pd.testing.assert_frame_equal(results[0], results[1].loc[shared], check_exact=False, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------- inputs from company filings

def test_price_only_models_keep_the_fingerprint_they_had_before_filings_were_added():
    """A changed fingerprint would retrain the live ridge and gbm models: their record must not move."""
    import hashlib
    import json

    import sklearn
    data = rankmodel.build_dataset(make_panels(), HORIZON)
    fold = rankmodel.folds(data, 2017)[0]
    settings = {"alpha": 100.0}
    old = json.dumps({"model": "ridge", "settings": settings, "features": features.NAMES,
                      "sklearn": sklearn.__version__, "horizon": HORIZON,
                      "trained_through": str(fold["trained_through"].date()), "rows": fold["train_rows"]},
                     sort_keys=True)
    assert rankmodel.fingerprint("ridge", settings, data, fold) == hashlib.sha1(old.encode()).hexdigest()[:12]
    assert rankmodel.model_spec("ridge", settings) == ("ridge", ["price"], {"alpha": 100.0})
    assert rankmodel.model_spec("r", {"kind": "gbm", "inputs": ["price", "fundamentals"], "max_iter": 5}) == \
        ("gbm", ["price", "fundamentals"], {"max_iter": 5})
    with pytest.raises(ValueError):
        rankmodel.model_spec("r", {"inputs": ["tea leaves"]})


def test_a_model_given_filing_inputs_uses_them_and_its_twin_does_not(tmp_path):
    """Plant next week's winners in one extra input: only the model that sees it can find them."""
    from tradebot import fundamentals
    panels = make_panels(seed=2)
    forward = features.forward_return(panels["open"], HORIZON)
    rng = np.random.default_rng(3)
    extra = {name: pd.DataFrame(rng.normal(size=forward.shape), index=forward.index, columns=forward.columns)
             for name in fundamentals.NAMES}
    extra["roa"] = forward.rank(axis=1, pct=True) + rng.normal(0, 0.3, forward.shape)
    data = rankmodel.build_dataset(panels, HORIZON, extra)
    assert data.names == features.NAMES + fundamentals.NAMES and data.x.shape[2] == 34
    cfg = {"first_test_year": 2017,
           "models": {"ridge": {"alpha": 1.0}, "ridge-fund": {"kind": "ridge", "inputs": ["price", "fundamentals"],
                                                             "alpha": 1.0}}}
    assert rankmodel.needs_fundamentals(cfg) and rankmodel.twins(cfg) == {"ridge-fund": "ridge"}
    conn = db.connect(":memory:")
    db.init_db(conn)
    rankmodel.update_scores(conn, data, cfg, tmp_path, lambda line: None)
    ic = {name: rankmodel.evaluate(rankmodel.load_scores(conn, name), data, 10, 5)["ic"]["mean"]
          for name in cfg["models"]}
    assert ic["ridge-fund"] > 0.3 and abs(ic["ridge"]) < 0.05

    price_only = rankmodel.build_dataset(panels, HORIZON)
    with pytest.raises(ValueError, match="fundamentals-fetch"):
        rankmodel.fit_fold("ridge-fund", cfg["models"]["ridge-fund"], price_only, rankmodel.folds(price_only, 2017)[0])


def test_the_report_compares_each_filing_model_with_its_twin(tmp_path):
    from tradebot import fundamentals
    panels = make_panels(seed=4)
    extra = {name: panels["close"].pct_change(5) for name in fundamentals.NAMES}
    data = rankmodel.build_dataset(panels, HORIZON, extra)
    cfg = {"first_test_year": 2017, "models": {"ridge": {"alpha": 1.0},
                                               "ridge-fund": {"kind": "ridge", "inputs": ["price", "fundamentals"],
                                                              "alpha": 1.0}}}
    conn = db.connect(":memory:")
    db.init_db(conn)
    rankmodel.update_scores(conn, data, cfg, None, lambda line: None)
    days = data.days[data.days.year >= 2017]
    results = {name: rankmodel.evaluate(rankmodel.load_scores(conn, name), data, 10, 5) for name in cfg["models"]}
    info = {"symbols": 60, "ended": 0, "first_bar": "2016-01-04", "last_bar": "2018-09-07", "horizon": 5,
            "top_n": 10, "cost_bps": 5, "model_groups": {n: rankmodel.model_spec(n, s)[1] for n, s in cfg["models"].items()},
            "fund_coverage": rankmodel.input_coverage(data, fundamentals.NAMES, days),
            "fund_companies": "60 of 60 companies have filing data", "twins": rankmodel.twins(cfg)}
    text = rankmodel.render(results, rankmodel.single_inputs(data, days), rankmodel.look_ahead_example(panels, data, days),
                            rankmodel.folds(data, 2017), info)
    for piece in ("price, volume and company filings", "## Company filings", "ridge-fund minus ridge",
                  "| earnings_yield |", "ridge: 25 from price and volume; ridge-fund: 25 from price and volume plus 9",
                  "public float leaves out", "| 2017 | 100% |"):
        assert piece in text


# ---------------------------------------------------------------- evaluation

def test_overlap_stats_count_each_window_once():
    noisy = pd.Series(np.random.default_rng(0).normal(0.01, 0.05, 500))   # 500 days, 5-day outcomes
    stats = rankmodel.overlap_stats(noisy, 5)
    assert stats["windows"] == 100 and stats["mean"] == pytest.approx(noisy.mean())
    naive_t = noisy.mean() / noisy.std() * np.sqrt(len(noisy))
    assert 0 < stats["t_stat"] < naive_t                       # stricter than counting every day as new evidence
    assert stats["t_stat"] == pytest.approx(naive_t / np.sqrt(5), rel=0.15)
    assert rankmodel.overlap_stats(pd.Series([0.01] * 3), 5)["windows"] == 0    # too short to say anything


def test_top_n_cost_is_charged_on_what_was_replaced():
    days = pd.bdate_range("2026-01-05", periods=4)
    names = [f"S{i}" for i in range(60)]
    scores = pd.DataFrame(0.0, index=days, columns=names)
    scores.iloc[0, :2] = 1.0                      # day 1: S0, S1
    scores.iloc[1, [0, 2]] = 1.0                  # day 2: S0 kept, S1 replaced by S2
    scores.iloc[2, :2] = 1.0
    scores.iloc[3, :2] = 1.0
    forward = pd.DataFrame(0.0, index=days, columns=names)
    forward.iloc[1, [0, 2]] = 0.06                # both of day 2's picks earn 6%; the rest earn nothing
    out = rankmodel.top_n_excess(scores, forward, horizon=1, top_n=2, cost_bps=10)
    assert out["turnover"].tolist() == [1.0, 0.5, 0.5, 0.0]
    assert out["gross"].iloc[1] == pytest.approx(0.06 - 0.12 / 60)
    assert out["net"].iloc[1] == pytest.approx(out["gross"].iloc[1] - 2 * 0.5 * 0.001)


def test_the_report_renders_with_every_section(tmp_path):
    conn = db.connect(":memory:")
    db.init_db(conn)
    cfg = {"first_test_year": 2017, "models": {"ridge": {"alpha": 1.0}}}
    panels = make_panels(signal=0.03)
    data = rankmodel.build_dataset(panels, HORIZON)
    rankmodel.update_scores(conn, data, cfg, None, lambda line: None)
    days = data.days[data.days.year >= 2017]
    results = {"ridge": rankmodel.evaluate(rankmodel.load_scores(conn, "ridge"), data, 10, 5)}
    cheat = rankmodel.look_ahead_example(panels, data, days)
    assert cheat["mean"] > 0.1                                           # tomorrow's return 'predicts' the week
    text = rankmodel.render(results, rankmodel.single_inputs(data, days), cheat, rankmodel.folds(data, 2017),
                            {"symbols": 60, "ended": 0, "first_bar": "2016-01-04", "last_bar": "2018-09-07",
                             "horizon": 5, "top_n": 10, "cost_bps": 5})
    for heading in ("Point-in-time audit", "Result: PASS", "Results (out of sample)", "Each input on its own",
                    "What look-ahead would look like", "Limits", "| ridge |", "money_flow_14"):
        assert heading in text


# ---------------------------------------------------------------- the model's own price database

def bars(symbol, dates, closes):
    return pd.DataFrame({"symbol": symbol, "date": dates, "open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": 1000.0})


NOW = pd.Timestamp("2026-11-14 12:00", tz="UTC")
JAN = ["2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07"]


def test_first_download_keeps_real_members_and_drops_reused_tickers():
    conn = db.connect(":memory:")
    db.init_db(conn)
    everything = pd.concat([bars("AAA", JAN, [10, 11, 12, 13]), bars("BBB", JAN[:2], [5, 6]),
                            bars("NEW", ["2017-11-01", "2017-11-02"], [50, 51])])   # another company, later
    said = []
    out = modeldata.update_prices(conn, ["AAA", "BBB", "NEW", "GONE"], "2016-01-01", None,
                                  lambda symbols, start: everything, said.append, now=NOW)
    assert out["symbols"] == 2 and out["reused"] == ["NEW"] and out["missing"] == ["GONE"]
    assert set(modeldata.stored_range(conn)) == {"AAA", "BBB"}
    assert any("reused" in line and "NEW" in line for line in said)


def test_refresh_adds_new_bars_and_redownloads_only_readjusted_history():
    conn = db.connect(":memory:")
    db.init_db(conn)
    db.upsert_prices(conn, pd.concat([bars("AAA", JAN, [10, 11, 12, 13]), bars("BBB", JAN, [20, 21, 22, 23]),
                                      bars("DEAD", JAN[:2], [5, 6])]))
    calls = []

    def fetch(symbols, start):
        calls.append((sorted(symbols), start))
        if start == "2016-01-01":                                  # full history, asked for BBB only
            return bars("BBB", JAN + ["2016-01-08"], [10, 10.5, 11, 11.5, 12])       # after a 2-for-1 split
        return pd.concat([bars("AAA", JAN[2:] + ["2016-01-08"], [12, 13, 14]),
                          bars("BBB", JAN[2:] + ["2016-01-08"], [11, 11.5, 12])])    # DEAD: nothing new

    out = modeldata.update_prices(conn, ["AAA", "BBB", "DEAD"], "2016-01-01", None, fetch,
                                  lambda line: None, now=NOW)
    assert out["refetched"] == ["BBB"] and calls[1] == (["BBB"], "2016-01-01")
    stored = modeldata.load_panels(conn)["close"]
    assert stored["AAA"].dropna().tolist() == [10, 11, 12, 13, 14]
    assert stored["BBB"].tolist() == [10, 10.5, 11, 11.5, 12]       # the whole history was replaced
    assert stored["DEAD"].dropna().tolist() == [5, 6]               # kept as it was


def test_a_bar_that_is_still_forming_is_left_out():
    conn = db.connect(":memory:")
    db.init_db(conn)
    during_session = pd.Timestamp("2016-01-07 15:00", tz="UTC")     # 10:00 in New York on the 7th
    modeldata.update_prices(conn, ["AAA"], "2016-01-01", None,
                            lambda symbols, start: bars("AAA", JAN, [10, 11, 12, 13]),
                            lambda line: None, now=during_session)
    assert modeldata.stored_range(conn)["AAA"][1] == "2016-01-06"


def test_a_company_that_ended_long_ago_is_never_asked_about_again():
    """Asked for recent days only, the data source answers with whoever uses the ticker now."""
    conn = db.connect(":memory:")
    db.init_db(conn)
    recent = ["2026-11-10", "2026-11-11", "2026-11-12"]
    db.upsert_prices(conn, pd.concat([bars("AAA", recent, [10, 11, 12]),
                                      bars("OLD", ["2016-01-04", "2016-01-05"], [30, 31])]))   # ended in 2016
    asked = []

    def fetch(symbols, start):
        asked.append(sorted(symbols))
        return pd.concat([bars("AAA", recent + ["2026-11-13"], [10, 11, 12, 13]),
                          bars("OLD", ["2026-11-12", "2026-11-13"], [5, 6])])    # another company's bars

    said = []
    out = modeldata.update_prices(conn, ["AAA", "OLD"], "2016-01-01", None, fetch, said.append, now=NOW)
    assert asked == [["AAA"]] and out["refetched"] == []
    stored = modeldata.stored_range(conn)
    assert stored["OLD"] == ("2016-01-04", "2016-01-05", 2)          # untouched, even though bars came back
    assert stored["AAA"][1] == "2026-11-13"
    assert any("1 of 2 symbols still trade" in line for line in said)
