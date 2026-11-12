"""The news test's point-in-time rules, checked on small made-up data. No network."""
import numpy as np
import pandas as pd
import pytest

from tradebot import newstest
from tradebot.newstest import (Settings, assign_sessions, audit_alignment, daily_ic, fama_macbeth,
                               fit_text_model, lexicon_counts, long_short, permutation_test,
                               prepare_articles, return_panels, same_day_session,
                               walk_forward_splits)

WEEK = pd.DatetimeIndex(["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13",
                         "2026-11-16"])   # Mon-Fri, then the next Monday


def ny(text):
    return pd.Timestamp(text, tz="America/New_York").tz_convert("UTC")


def sessions(*times, days=WEEK):
    out = assign_sessions(pd.Series([ny(t) for t in times]), days)
    return [(None if pd.isna(s) else s.strftime("%m-%d"), b if isinstance(b, str) else None)
            for s, b in zip(out["session"], out["bucket"])]


# ---------------------------------------------------------------- rule 2: which session

def test_news_before_the_decision_moment_is_used_that_day():
    assert sessions("2026-11-10 08:59", "2026-11-10 09:00:00") == [("11-10", "overnight")] * 2


def test_news_after_the_decision_moment_waits_for_the_next_session():
    assert sessions("2026-11-10 09:00:01", "2026-11-10 09:45", "2026-11-10 15:59",
                    "2026-11-10 16:00:00") == [("11-11", "prior_session")] * 4
    assert sessions("2026-11-10 16:00:01", "2026-11-10 23:30") == [("11-11", "overnight")] * 2


def test_weekend_news_goes_to_monday_and_news_after_the_last_decision_is_unused():
    assert sessions("2026-11-13 18:00", "2026-11-14 12:00", "2026-11-15 22:00") == [("11-16", "overnight")] * 3
    assert sessions("2026-11-16 10:00") == [(None, None)]
    assert sessions("2026-11-08 12:00") == [("11-09", "overnight")]      # before the first day


def test_alignment_follows_new_york_time_through_daylight_saving():
    summer = pd.DatetimeIndex(["2026-07-06", "2026-07-07"])     # New York = UTC-4
    winter = pd.DatetimeIndex(["2026-12-07", "2026-12-08"])     # New York = UTC-5
    utc = lambda t: pd.Series([pd.Timestamp(t, tz="UTC")])
    assert assign_sessions(utc("2026-07-06 12:59"), summer)["session"][0] == pd.Timestamp("2026-07-06")
    assert assign_sessions(utc("2026-07-06 13:01"), summer)["session"][0] == pd.Timestamp("2026-07-07")
    assert assign_sessions(utc("2026-12-07 13:59"), winter)["session"][0] == pd.Timestamp("2026-12-07")
    assert assign_sessions(utc("2026-12-07 14:01"), winter)["session"][0] == pd.Timestamp("2026-12-08")


def test_same_day_session_is_only_for_news_inside_market_hours():
    got = same_day_session(pd.Series([ny("2026-11-10 09:30:00"), ny("2026-11-10 10:00"),
                                      ny("2026-11-10 16:00"), ny("2026-11-10 16:01"),
                                      ny("2026-11-14 12:00")]), WEEK)
    assert [None if pd.isna(d) else d.strftime("%m-%d") for d in got] == [None, "11-10", "11-10", None, None]


def test_audit_catches_a_headline_used_before_it_was_known():
    good = pd.DataFrame({"known_at": [ny("2026-11-10 08:00")], "session": [pd.Timestamp("2026-11-10")]})
    assert audit_alignment(good, WEEK)["headlines checked"] == 1
    assert audit_alignment(good, WEEK)["closest headline to its decision (minutes before)"] == 60

    used_early = good.assign(known_at=[ny("2026-11-10 09:30")])
    with pytest.raises(AssertionError, match="before they were known"):
        audit_alignment(used_early, WEEK)
    used_late = good.assign(known_at=[ny("2026-11-09 08:00")])     # belonged to the day before
    with pytest.raises(AssertionError, match="later session"):
        audit_alignment(used_late, WEEK)


# ---------------------------------------------------------------- rule 1: which text

def article(id, headline, symbols, created="2026-11-10 08:00", revised_min=0):
    created = ny(created)
    return {"id": id, "created_at": created, "updated_at": created + pd.Timedelta(minutes=revised_min),
            "headline": headline, "symbols": symbols}


def test_prepare_articles_applies_every_filter():
    raw = pd.DataFrame([
        article(1, "Apple beats estimates", ["AAPL"]),
        article(2, "Apple beats estimates!", ["AAPL"], created="2026-11-10 08:30"),   # same headline again
        article(3, "Quietly rewritten later", ["AAPL"], revised_min=600),
        article(4, "Small fix soon after", ["AAPL"], revised_min=10),
        article(5, "Ten stocks moving today", ["AAPL", "A", "B", "C", "D", "E"]),
        article(6, "United Technologies wins contract", ["UTX"]),
        article(7, "Something about another company", ["ZZZZ"]),
    ])
    items, counts = prepare_articles(raw, ["AAPL", "RTX"], {"UTX": "RTX"})

    assert sorted(items["id"]) == [1, 4, 6]
    assert counts["dropped: revised long after publication"] == 1
    assert counts["dropped: round-up tagged with many tickers"] == 1
    assert counts["dropped: repeat of an earlier headline"] == 1
    assert items.set_index("id").loc[6, "symbol"] == "RTX"                      # old ticker mapped
    # a revised article counts as known from its revision, not its first publication
    assert items.set_index("id").loc[4, "known_at"] == ny("2026-11-10 08:10")


def test_lexicon_counts():
    assert lexicon_counts("Shares SURGE after upgrade; rival plunges on lawsuit") == (2, 2)
    assert lexicon_counts("Company announces annual meeting date") == (0, 0)


# ---------------------------------------------------------------- rule 3 and 5: target, controls

def price_table(closes: dict, opens: dict | None = None):
    days = pd.bdate_range("2026-01-05", periods=len(next(iter(closes.values()))))
    rows = []
    for sym, values in closes.items():
        for i, (d, c) in enumerate(zip(days, values)):
            rows.append({"symbol": sym, "date": d, "open": (opens or {}).get(sym, values)[i] if opens
                         else (values[i - 1] if i else c), "close": c, "volume": 1000.0})
    return pd.DataFrame(rows)


def test_target_is_open_to_close_and_excess_sums_to_zero():
    prices = price_table({"A": [100.0, 110.0], "B": [100.0, 99.0]}, opens={"A": [100.0, 100.0], "B": [100.0, 100.0]})
    p = return_panels(prices, ["A", "B"])
    assert p["open_close"].iloc[1].tolist() == pytest.approx([0.10, -0.01])
    assert p["excess"].iloc[1].sum() == pytest.approx(0)


def test_controls_never_use_the_current_day():
    rng = np.random.default_rng(0)
    closes = {s: (100 * np.exp(np.cumsum(rng.normal(0, 0.01, 40)))).tolist() for s in "AB"}
    before = return_panels(price_table(closes), ["A", "B"])
    closes["A"][-1] *= 3                                   # change today's close only
    after = return_panels(price_table(closes), ["A", "B"])
    for name in ("ret_1d", "ret_5d", "vol_20d", "dollar_volume_20d"):
        assert before[name].iloc[-1].equals(after[name].iloc[-1]), name
    assert not before["open_close"].iloc[-1].equals(after["open_close"].iloc[-1])


# ---------------------------------------------------------------- rule 4: walk-forward

def test_splits_never_overlap_and_training_always_ends_before_testing():
    days = list(pd.bdate_range("2020-01-01", periods=100))
    splits = list(walk_forward_splits(days, initial_train=40, test_size=15, embargo=2))
    assert len(splits) == 4
    tested = [d for _, test in splits for d in test]
    assert tested == days[42:] and len(set(tested)) == len(tested)       # back to back, no repeats
    for train, test in splits:
        assert train[0] == days[0]                                        # expanding window
        assert days.index(test[0]) - days.index(train[-1]) == 3           # 2 embargo days between
        assert max(train) < min(test)


def test_rolling_window_keeps_a_fixed_length_and_bad_arguments_are_refused():
    days = list(range(100))
    for train, _ in walk_forward_splits(days, 40, 15, embargo=0, expanding=False):
        assert len(train) == 40
    with pytest.raises(ValueError):
        list(walk_forward_splits(days, 0, 15))


def test_vocabulary_is_built_from_training_text_only():
    train = pd.DataFrame({"text": ["profits surge on demand"] * 6 + ["profits plunge on weak demand"] * 6,
                          "y": [0.01] * 6 + [-0.01] * 6})
    vectorizer, model = fit_text_model(train, Settings(min_df=2))
    assert "surge" in vectorizer.vocabulary_ and "futureword" not in vectorizer.vocabulary_
    unseen = vectorizer.transform(["futureword futureword"])
    assert unseen.nnz == 0                                  # a word never seen in training carries no weight
    up, down = model.predict(vectorizer.transform(["profits surge", "profits plunge"]))
    assert up > down


def test_company_names_are_hidden_from_the_text_model():
    train = pd.DataFrame({"text": ["apple aapl microsoft surge"] * 6, "y": [0.01, -0.01] * 3})
    vectorizer, _ = fit_text_model(train, Settings(min_df=1))
    assert set(vectorizer.vocabulary_) == {"surge"}


# ---------------------------------------------------------------- scoring

def frame(rows, cols="ABCD"):
    return pd.DataFrame(rows, index=pd.bdate_range("2026-01-05", periods=len(rows)), columns=list(cols))


def test_daily_ic():
    target = frame([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, np.nan, np.nan]])
    signal = frame([[10, 20, 30, 40], [4, 3, 2, 1], [5, 5, 5, 5], [1, 2, 3, 4]])
    ic = daily_ic(signal, target, min_names=3)
    assert ic.tolist() == pytest.approx([1.0, -1.0])        # flat-signal day and thin day are skipped


def test_long_short_is_dollar_neutral_and_charges_four_trades_a_day():
    signal = frame([[1, 0, 0, -1], [3, 3, 3, 3]])
    returns = frame([[0.02, 0.0, 0.0, -0.01], [0.05, 0.05, 0.05, 0.05]])
    book = long_short(signal, returns, cost_bps=5, min_names=2)
    assert book["gross"].tolist() == pytest.approx([0.03, 0.0])
    assert book["net"].tolist() == pytest.approx([0.03 - 0.0020, 0.0])   # no position, no cost
    assert book["active"].tolist() == [True, False]


def test_permutation_test_separates_a_real_signal_from_noise():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2024-01-01", periods=200)
    target = pd.DataFrame(rng.normal(size=(200, 12)), index=days)
    settings = Settings(min_names=6, permutations=60)
    real = permutation_test(target + rng.normal(size=target.shape), target, settings)
    noise = permutation_test(pd.DataFrame(rng.normal(size=target.shape), index=days), target, settings)
    assert real["real"] > 0.5 and real["p_value"] == pytest.approx(1 / 61)
    assert abs(real["shuffled_mean"]) < 0.02
    assert noise["p_value"] > 0.05


def test_fama_macbeth_recovers_a_planted_effect_next_to_a_useless_control():
    rng = np.random.default_rng(2)
    days = pd.bdate_range("2024-01-01", periods=300)
    signal = pd.DataFrame(rng.normal(size=(300, 20)), index=days)
    control = pd.DataFrame(rng.normal(size=(300, 20)), index=days)
    target = 0.0010 * signal + pd.DataFrame(rng.normal(0, 0.01, size=(300, 20)), index=days)
    table = fama_macbeth(target, {"signal": signal, "control": control})
    assert table.loc["signal", "bps_per_sd"] == pytest.approx(10, abs=2.5)
    assert table.loc["signal", "t_stat"] > 5 and abs(table.loc["control", "t_stat"]) < 3


# ---------------------------------------------------------------- end to end

def make_world(publish: str, n_days=520, n_stocks=12, seed=0):
    """Prices plus headlines that give away each day's move, published at `publish` New York."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-02", periods=n_days)
    universe = [f"S{i:02d}" for i in range(n_stocks)]
    prices, articles, uid = [], [], 0
    for sym in universe:
        level = 100.0
        for day in days:
            move = rng.normal(0, 0.01)
            prices.append({"symbol": sym, "date": day, "open": level, "close": level * (1 + move),
                           "volume": 1e6})
            level *= 1 + move
            if rng.random() < 0.6:
                uid += 1
                when = pd.Timestamp(f"{day.date()} {publish}", tz="America/New_York").tz_convert("UTC")
                word = "surges" if move > 0 else "plunges"
                articles.append({"id": uid, "created_at": when, "updated_at": when, "symbols": [sym],
                                 "headline": f"Stock {word} in trading note number {uid}"})
    return pd.DataFrame(articles), pd.DataFrame(prices), universe


SMALL = Settings(initial_train_days=250, test_days=45, min_names=6, permutations=40, min_df=2)
WORDS = "word-list sentiment, all news since the last decision"
MODEL = "learned text model (walk-forward)"


def test_news_published_before_the_decision_is_found_and_used():
    articles, prices, universe = make_world(publish="08:00")
    r = newstest.run(articles, prices, universe, start="2020-01-02", settings=SMALL)
    assert r.signals[WORDS]["ic"]["mean"] > 0.3 and r.signals[WORDS]["ic"]["t_stat"] > 10
    assert r.signals[MODEL]["ic"]["mean"] > 0.3            # the model learned the words from the past
    assert r.signals[WORDS]["net"]["annual_return"] > 0
    assert r.permutation[WORDS]["p_value"] < 0.05
    assert all(f["train_to"] < f["test_from"] for f in r.folds) and len(r.folds) >= 5
    assert r.oos_period[0] > r.period[0]
    assert "PASS" in newstest.render(r)


def test_the_same_news_published_after_the_open_cannot_reach_that_days_prediction():
    """The strongest check. The headlines still give away the day's move, but
    they come out at 10:00, after the decision. If any path let them through,
    the IC would be large. It has to be zero."""
    articles, prices, universe = make_world(publish="10:00")
    r = newstest.run(articles, prices, universe, start="2020-01-02", settings=SMALL)
    for name in (WORDS, MODEL):
        assert abs(r.signals[name]["ic"]["mean"]) < 0.05, name
        assert abs(r.signals[name]["ic"]["t_stat"]) < 3, name
    # ...while the deliberately leaky comparison shows the giveaway is really there
    assert r.leaky["ic"]["mean"] > 0.3


def test_each_fold_trains_only_on_sessions_before_its_test_block(monkeypatch):
    articles, prices, universe = make_world(publish="08:00", n_days=420)
    seen, original = [], newstest.fit_text_model

    def spy(train, settings):
        seen.append(train["session"].max())
        return original(train, settings)

    monkeypatch.setattr(newstest, "fit_text_model", spy)
    r = newstest.run(articles, prices, universe, start="2020-01-02", settings=SMALL)
    assert len(seen) == len(r.folds) >= 3
    for last_training_session, fold in zip(seen, r.folds):
        assert last_training_session < fold["test_from"]
        assert last_training_session <= fold["train_to"]


def test_too_little_history_is_a_clear_error():
    articles, prices, universe = make_world(publish="08:00", n_days=100)
    with pytest.raises(ValueError, match="not enough to train and test"):
        newstest.run(articles, prices, universe, start="2020-01-02", settings=SMALL)
