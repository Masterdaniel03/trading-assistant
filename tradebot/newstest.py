"""Does past news predict the next session's return? A walk-forward test.

Research only: nothing in this module trades.

The whole point is that no prediction may use anything that was not yet known.
Each rule below is enforced in code, checked again at run time (the "audit"
in the report) and covered by tests.

1. WHEN A HEADLINE IS KNOWN. The archive serves each article's latest text,
   so a headline counts as known from its `updated_at`, not its `created_at`.
   An article revised more than `revision_tolerance_min` after publication is
   dropped altogether.
2. WHICH SESSION IT CAN AFFECT. The decision for trading day D is made at
   `decision_time` New York, before the open. A headline known at or before
   that moment may be used for D. Anything later waits for the next session.
3. WHAT IS PREDICTED. D's open-to-close return minus the same-day average of
   the universe. It begins after the decision is made.
4. WHAT THE MODEL MAY LEARN FROM. Walk-forward: the text model's vocabulary
   and weights are fitted on sessions that ended before each test block
   (minus an embargo), then frozen for that block. Nothing is fitted on the
   whole timeline.
5. CONTROLS (momentum, volatility, liquidity) use prices up to D-1's close.

Every setting is fixed before looking at results. Tuning them against the
out-of-sample numbers would be its own kind of look-ahead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NY = "America/New_York"
OPEN, CLOSE = pd.Timedelta(hours=9, minutes=30), pd.Timedelta(hours=16)

# A small fixed finance word list (in the spirit of the Loughran-McDonald
# dictionary). It holds no knowledge of any particular event, so it cannot leak.
POSITIVE = frozenset("""
gain gains gained rise rises rose rising surge surges surged jump jumps jumped soar soars soared
rally rallies rallied climb climbs climbed beat beats strong stronger strength upgrade upgrades
upgraded raise raises raised record growth grows grew profit profits profitable outperform
bullish buy boost boosts boosted win wins won approval approved approves expand expands
expansion exceed exceeds exceeded tops topped higher best breakthrough positive optimistic upbeat
rebound rebounds recover recovers recovery buyback upside robust milestone
""".split())
NEGATIVE = frozenset("""
loss losses decline declines declined drop drops dropped fall falls fell falling plunge plunges
plunged slump slumps tumble tumbles tumbled sink sinks slide slides slips weak weaker weakness
miss misses missed cut cuts downgrade downgrades downgraded lawsuit sued sues probe investigation
fraud recall recalls warning warns warned bankruptcy layoffs layoff concern concerns risk risks
fear fears crash crisis default delay delays delayed halt halts halted penalty fined bearish
selloff underperform lower lowers lowered worst disappointing disappoints shortfall slowdown
drag hurt hurts trouble struggles struggling plummet plummets scandal breach strike shutdown
negative loses losing lost sell
""".split())

# Company names and tickers are hidden from the text model, so it has to learn
# from what a headline says rather than from which company it is about.
COMPANY_WORDS = frozenset("""
apple aapl american express amex axp boeing ba caterpillar cat cisco csco chevron cvx disney
walt dis dow goldman sachs gs home depot hd ibm intel intc johnson jnj jpmorgan chase jpm coca
cola ko mcdonald mcdonalds mcd mmm merck mrk microsoft msft nike nke pfizer pfe procter gamble
pg raytheon rtx united technologies utx travelers trv unitedhealth unh visa verizon vz walgreens
boots wba walmart wmt exxon mobil xom
""".split())


@dataclass(frozen=True)
class Settings:
    decision_time: str = "09:00"        # New York; 30 minutes before the open
    revision_tolerance_min: int = 60    # an article revised later than this is dropped
    max_tickers_per_article: int = 5    # more than this is a market round-up, not stock news
    initial_train_days: int = 504       # about two years before the first test block
    test_days: int = 63                 # one quarter per test block
    embargo_days: int = 1               # gap between the end of training and the test block
    cost_bps: float = 5.0               # per trade, each side, each way
    min_names: int = 10                 # a day needs this many stocks to be scored
    ridge_alpha: float = 10.0
    min_df: int = 5                     # a word must appear in this many training documents
    max_features: int = 20000
    permutations: int = 500
    seed: int = 0
    level_window: int = 250             # sessions used for a stock's "usual" sentiment (one year)


# ---------------------------------------------------------------- 1. which text is usable

def prepare_articles(raw: pd.DataFrame, universe: list[str], aliases: dict[str, str],
                     settings: Settings = Settings()) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per (headline, stock) that is safe to use, with the moment it became known.

    raw: id, created_at, updated_at (UTC), headline, symbols (list of tickers).
    Returns (items, counts). items has id, symbol, known_at, headline.
    """
    counts = {"articles downloaded": len(raw)}
    lag_min = (raw["updated_at"] - raw["created_at"]).dt.total_seconds() / 60
    stable = raw[(lag_min >= 0) & (lag_min <= settings.revision_tolerance_min)]
    counts["dropped: revised long after publication"] = len(raw) - len(stable)

    focused = stable[stable["symbols"].map(len) <= settings.max_tickers_per_article]
    counts["dropped: round-up tagged with many tickers"] = len(stable) - len(focused)

    items = (focused[["id", "updated_at", "headline", "symbols"]]
             .rename(columns={"updated_at": "known_at", "symbols": "symbol"}).explode("symbol"))
    items["symbol"] = items["symbol"].replace(aliases)
    items = items[items["symbol"].isin(universe)]
    items["key"] = (items["headline"].str.lower()
                    .str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip())
    items = items[items["key"] != ""].sort_values(["known_at", "id"])
    unique = items.drop_duplicates(["symbol", "key"], keep="first")     # earliest copy wins
    counts["dropped: repeat of an earlier headline"] = len(items) - len(unique)
    counts["headline-stock pairs kept"] = len(unique)
    return unique.drop(columns="key").reset_index(drop=True), counts


# ---------------------------------------------------------------- 2. which session it can affect

def session_times(trading_days: pd.DatetimeIndex, settings: Settings = Settings()) -> pd.DataFrame:
    """UTC moments for each trading day: decision, open and close."""
    days = pd.DatetimeIndex(trading_days)
    local = days.tz_localize(NY)
    return pd.DataFrame({
        "decision": (local + pd.Timedelta(settings.decision_time + ":00")).tz_convert("UTC"),
        "open": (local + OPEN).tz_convert("UTC"),
        "close": (local + CLOSE).tz_convert("UTC"),
    }, index=days)


def assign_sessions(known_at: pd.Series, trading_days: pd.DatetimeIndex,
                    settings: Settings = Settings()) -> pd.DataFrame:
    """For each timestamp: the first trading day whose decision moment is at or
    after it (NaT if there is none yet), and whether it arrived overnight or
    during the previous session.

    A headline known at 09:01 on Tuesday is too late for Tuesday and is used
    for Wednesday, in the "prior_session" bucket. One known at 18:00 on
    Tuesday is also used for Wednesday, in the "overnight" bucket.
    """
    times = session_times(trading_days, settings)
    decision = pd.DatetimeIndex(times["decision"])
    close = pd.DatetimeIndex(times["close"])
    known = pd.DatetimeIndex(known_at)
    pos = decision.searchsorted(known, side="left")
    in_range = pos < len(decision)
    safe = np.where(in_range, pos, 0)
    session = pd.Series(times.index[safe], index=known_at.index).where(in_range)
    after_prev_close = known > close[np.maximum(safe - 1, 0)]
    overnight = (safe == 0) | np.asarray(after_prev_close)
    bucket = pd.Series(np.where(overnight, "overnight", "prior_session"),
                       index=known_at.index).where(in_range)
    return pd.DataFrame({"session": session, "bucket": bucket})


def same_day_session(known_at: pd.Series, trading_days: pd.DatetimeIndex,
                     settings: Settings = Settings()) -> pd.Series:
    """The trading day a timestamp falls INSIDE (open < t <= close), else NaT.

    Used only for the deliberately leaky comparison: news written during a
    session describes that session's move, so it must never be used to
    predict it.
    """
    times = session_times(trading_days, settings)
    close, open_ = pd.DatetimeIndex(times["close"]), pd.DatetimeIndex(times["open"])
    known = pd.DatetimeIndex(known_at)
    pos = close.searchsorted(known, side="left")
    in_range = pos < len(close)
    safe = np.where(in_range, pos, 0)
    inside = in_range & np.asarray(known > open_[safe])
    return pd.Series(times.index[safe], index=known_at.index).where(inside)


def audit_alignment(items: pd.DataFrame, trading_days: pd.DatetimeIndex,
                    settings: Settings = Settings()) -> dict:
    """Run-time proof of rule 2: every headline is known at or before the
    decision moment of the session it is used for, and after the previous one.
    Raises AssertionError if any is not."""
    times = session_times(trading_days, settings)
    used = items.dropna(subset=["session"])
    decision = times["decision"].reindex(used["session"]).to_numpy()
    previous = times["decision"].shift(1).reindex(used["session"]).to_numpy()
    known = pd.DatetimeIndex(used["known_at"]).tz_convert("UTC").to_numpy()
    decision, previous = pd.DatetimeIndex(decision).to_numpy(), pd.DatetimeIndex(previous).to_numpy()
    late = known > decision
    early = ~pd.isna(previous) & (known <= previous)
    assert not late.any(), f"{late.sum()} headline(s) used before they were known"
    assert not early.any(), f"{early.sum()} headline(s) assigned to a later session than needed"
    gap = (decision - known) / np.timedelta64(1, "m")
    return {"headlines checked": int(len(used)),
            "closest headline to its decision (minutes before)": float(gap.min()) if len(gap) else None}


# ---------------------------------------------------------------- features from text

_WORD = re.compile(r"[a-z]+")


def lexicon_counts(headline: str) -> tuple[int, int]:
    words = _WORD.findall(headline.lower())
    return sum(w in POSITIVE for w in words), sum(w in NEGATIVE for w in words)


def sentiment_panel(items: pd.DataFrame, days: pd.DatetimeIndex, universe: list[str],
                    session_col: str = "session") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sentiment, headline count), each days x stocks. Sentiment is
    (positive - negative) / (positive + negative + 1) over a stock's headlines
    for that session, and 0 where it had none."""
    used = items.dropna(subset=[session_col])
    grouped = used.groupby([session_col, "symbol"])[["pos", "neg"]].sum()
    score = (grouped["pos"] - grouped["neg"]) / (grouped["pos"] + grouped["neg"] + 1)
    count = used.groupby([session_col, "symbol"]).size()

    def wide(series):
        return (series.unstack("symbol").reindex(index=days, columns=universe).fillna(0.0))

    return wide(score), wide(count)


def usual_level(sentiment: pd.DataFrame, window: int = 250, min_periods: int = 60) -> pd.DataFrame:
    """Each stock's average sentiment over the previous `window` sessions, not
    counting the current one.

    Some companies simply get more upbeat coverage than others. Separating that
    standing level from the day's news shows whether a result comes from news
    ARRIVING (today minus the usual level) or only from WHICH stocks tend to
    get good news (the usual level alone). The second is not news timing at all.
    """
    return sentiment.shift(1).rolling(window, min_periods=min_periods).mean()


# ---------------------------------------------------------------- 3. what is predicted

def return_panels(prices: pd.DataFrame, universe: list[str]) -> dict[str, pd.DataFrame]:
    """Panels (days x stocks) from daily bars: the target and the controls.

    open_close[D] is the target. Every control at D is shifted so that it uses
    closes up to D-1 only.
    """
    def wide(column):
        return prices.pivot(index="date", columns="symbol", values=column).reindex(columns=universe)

    open_, close, volume = wide("open"), wide("close"), wide("volume")
    open_close = close / open_ - 1
    close_close = close.pct_change(fill_method=None)
    return {
        "open_close": open_close,
        "excess": open_close.sub(open_close.mean(axis=1), axis=0),
        "ret_1d": close_close.shift(1),
        "ret_5d": (close / close.shift(5) - 1).shift(1),
        "vol_20d": close_close.rolling(20).std().shift(1),
        "dollar_volume_20d": (close * volume).rolling(20).mean().shift(1),
    }


# ---------------------------------------------------------------- 4. walk-forward

def walk_forward_splits(days, initial_train: int, test_size: int, embargo: int = 0,
                        expanding: bool = True):
    """Yield (train_days, test_days) over an ordered list of days.

    Training always ends `embargo` days before its test block starts. With
    `expanding` the training window grows from the first day; otherwise it
    rolls, keeping `initial_train` days. Test blocks are back to back and
    never overlap.
    """
    if initial_train <= 0 or test_size <= 0 or embargo < 0:
        raise ValueError("initial_train and test_size must be positive, embargo not negative")
    days = list(days)
    start = initial_train + embargo
    while start < len(days):
        train_end = start - embargo
        train_start = 0 if expanding else train_end - initial_train
        yield days[train_start:train_end], days[start:start + test_size]
        start += test_size


def build_documents(items: pd.DataFrame, excess: pd.DataFrame) -> pd.DataFrame:
    """One document per (session, stock): its headlines joined, plus the target."""
    used = items.dropna(subset=["session"])
    docs = (used.groupby(["session", "symbol"])["headline"].agg(" . ".join)
            .rename("text").reset_index())
    target = excess.stack().rename("y")
    target.index.names = ["session", "symbol"]
    return docs.join(target, on=["session", "symbol"])


def fit_text_model(train: pd.DataFrame, settings: Settings = Settings()):
    """Fit vocabulary, word weights and regression on the training rows ONLY."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    train = train.dropna(subset=["y"])
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"[a-z]{2,}", ngram_range=(1, 2),
                                 min_df=min(settings.min_df, max(1, len(train))),
                                 max_features=settings.max_features, sublinear_tf=True,
                                 stop_words=sorted(COMPANY_WORDS))
    x = vectorizer.fit_transform(train["text"])
    limit = 3 * train["y"].std()            # tame outliers, using training rows only
    y = train["y"].clip(-limit, limit) if limit > 0 else train["y"]
    return vectorizer, Ridge(alpha=settings.ridge_alpha).fit(x, y)


def walk_forward_text_model(docs: pd.DataFrame, days: pd.DatetimeIndex, universe: list[str],
                            settings: Settings = Settings()) -> tuple[pd.DataFrame, list[dict]]:
    """Out-of-sample predictions (days x stocks; NaN outside the test blocks)
    and one audit record per fold."""
    predictions = pd.DataFrame(np.nan, index=days, columns=universe)
    folds = []
    splits = walk_forward_splits(days, settings.initial_train_days, settings.test_days,
                                 settings.embargo_days)
    for train_days, test_days in splits:
        train = docs[docs["session"].isin(train_days)]
        test = docs[docs["session"].isin(test_days)]
        assert train_days[-1] < test_days[0], "training must end before the test block"
        assert train["session"].max() < test_days[0], "a training row is dated inside the test block"
        if train["y"].notna().sum() < 50:
            continue
        vectorizer, model = fit_text_model(train, settings)
        # A stock with no headlines gets what the model says about an empty text.
        predictions.loc[test_days] = float(model.predict(vectorizer.transform([""]))[0])
        if len(test):
            values = model.predict(vectorizer.transform(test["text"]))
            for day, symbol, value in zip(test["session"], test["symbol"], values):
                predictions.at[day, symbol] = value
        folds.append({"train_from": train_days[0], "train_to": train_days[-1],
                      "test_from": test_days[0], "test_to": test_days[-1],
                      "train_docs": int(train["y"].notna().sum()), "test_docs": int(len(test)),
                      "vocabulary": len(vectorizer.vocabulary_)})
    return predictions, folds


# ---------------------------------------------------------------- scoring

def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-by-row correlation of two matrices, ignoring cells missing in either."""
    both = np.isfinite(a) & np.isfinite(b)
    n = both.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        da = np.where(both, a - (np.where(both, a, 0).sum(axis=1) / n)[:, None], 0.0)
        db = np.where(both, b - (np.where(both, b, 0).sum(axis=1) / n)[:, None], 0.0)
        return (da * db).sum(axis=1) / np.sqrt((da ** 2).sum(axis=1) * (db ** 2).sum(axis=1))


def daily_ic(signal: pd.DataFrame, target: pd.DataFrame, min_names: int = 10,
             rank: bool = True) -> pd.Series:
    """Information coefficient per day: the correlation, across stocks, between
    the signal and what happened. `rank=True` gives the rank IC (Spearman).
    Days with too few stocks, or where every stock has the same signal, are skipped."""
    signal, target = signal.align(target, join="inner")
    both = signal.notna() & target.notna()
    signal, target = signal.where(both), target.where(both)
    if rank:
        signal, target = signal.rank(axis=1), target.rank(axis=1)
    ic = pd.Series(_row_corr(signal.to_numpy(float), target.to_numpy(float)), index=signal.index)
    ic[both.sum(axis=1) < min_names] = np.nan
    return ic.dropna()


def ic_summary(ic: pd.Series) -> dict:
    n = len(ic)
    if n < 2:
        return {"days": n, "mean": np.nan, "t_stat": np.nan, "share_positive": np.nan}
    return {"days": n, "mean": float(ic.mean()),
            "t_stat": float(ic.mean() / ic.std() * np.sqrt(n)) if ic.std() > 0 else np.nan,
            "share_positive": float((ic > 0).mean())}


def long_short(signal: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 5.0,
               min_names: int = 10) -> pd.DataFrame:
    """Daily return of $1 long / $1 short, weighted by the signal.

    Each stock's weight is its signal minus the day's average, scaled so the
    longs sum to +1 and the shorts to -1. Positions are opened at the open and
    closed at the close, so both sides trade twice a day: the cost is
    4 x cost_bps of $1 on every day a position is held.
    """
    signal, returns = signal.align(returns, join="inner")
    both = signal.notna() & returns.notna()
    signal, returns = signal.where(both), returns.where(both)
    tilt = signal.sub(signal.mean(axis=1), axis=0)
    gross_tilt = tilt.abs().sum(axis=1)
    active = (gross_tilt > 0) & (both.sum(axis=1) >= min_names)
    weights = tilt.div(gross_tilt.where(active) / 2, axis=0)
    gross = (weights * returns).sum(axis=1).where(active, 0.0)
    cost = active * 4 * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "net": gross - cost, "active": active})


def performance(returns: pd.Series) -> dict:
    if len(returns) < 2 or returns.std() == 0:
        return {"annual_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    wealth = (1 + returns).cumprod()
    return {"annual_return": float(returns.mean() * 252),
            "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)),
            "max_drawdown": float((wealth / wealth.cummax() - 1).min())}


def permutation_test(signal: pd.DataFrame, target: pd.DataFrame, settings: Settings = Settings()) -> dict:
    """Shuffle which day each day's signals belong to, many times.

    If the real result comes from news lining up with the returns that follow
    it, breaking that line-up must destroy it: the shuffled mean IC should sit
    around zero and the real one far out in the tail. p is the share of
    shuffles at least as extreme as the real result.
    """
    real = daily_ic(signal, target, settings.min_names).mean()
    rng = np.random.default_rng(settings.seed)
    values = signal.to_numpy(float)
    shuffled = np.array([
        daily_ic(pd.DataFrame(values[rng.permutation(len(values))], index=signal.index,
                              columns=signal.columns), target, settings.min_names).mean()
        for _ in range(settings.permutations)])
    return {"real": float(real), "shuffled_mean": float(np.nanmean(shuffled)),
            "shuffled_std": float(np.nanstd(shuffled)),
            "p_value": float((1 + (np.abs(shuffled) >= abs(real)).sum()) / (1 + len(shuffled)))}


def fama_macbeth(target: pd.DataFrame, regressors: dict[str, pd.DataFrame],
                 min_names: int = 10) -> pd.DataFrame:
    """Does the signal still matter once momentum and volatility are accounted for?

    Each day, regress the stocks' returns on the day's standardized
    regressors; then average each coefficient over days. `bps_per_sd` is the
    return, in basis points, that goes with a one-standard-deviation higher
    value on that day, holding the others fixed.
    """
    names = list(regressors)
    rows = []
    for day in target.index:
        y = target.loc[day]
        x = pd.DataFrame({n: regressors[n].loc[day] for n in names if day in regressors[n].index})
        ok = y.notna() & x.notna().all(axis=1) if len(x.columns) == len(names) else None
        if ok is None or ok.sum() < max(min_names, len(names) + 2):
            continue
        x, y = x[ok], y[ok]
        spread = x.std()
        if (spread == 0).any():
            continue                      # e.g. no stock had news that day
        z = (x - x.mean()) / spread
        design = np.column_stack([np.ones(len(z)), z.to_numpy(float)])
        rows.append(np.linalg.lstsq(design, y.to_numpy(float), rcond=None)[0][1:])
    betas = pd.DataFrame(rows, columns=names)
    return pd.DataFrame({"bps_per_sd": betas.mean() * 1e4,
                         "t_stat": betas.mean() / betas.std() * np.sqrt(len(betas)),
                         "days": len(betas)})


def ic_by_liquidity(signal: pd.DataFrame, target: pd.DataFrame, dollar_volume: pd.DataFrame,
                    min_names: int = 6) -> dict[str, dict]:
    """Rank IC within thirds of the universe by recent dollar volume (known the day before)."""
    pct = dollar_volume.rank(axis=1, pct=True)
    groups = {"least traded third": pct <= 1 / 3, "middle third": (pct > 1 / 3) & (pct <= 2 / 3),
              "most traded third": pct > 2 / 3}
    return {name: ic_summary(daily_ic(signal.where(mask), target, min_names))
            for name, mask in groups.items()}


# ---------------------------------------------------------------- the experiment

@dataclass
class Result:
    settings: Settings
    universe: list[str]
    no_prices: list[str]
    period: tuple
    oos_period: tuple
    counts: dict
    coverage: float
    audit: dict
    folds: list[dict]
    signals: dict[str, dict]            # name -> ic / long-short numbers on the out-of-sample days
    leaky: dict
    permutation: dict[str, dict]
    controls: pd.DataFrame
    liquidity: dict[str, dict]
    yearly: pd.DataFrame = field(default_factory=pd.DataFrame)
    level_vs_timing: dict[str, dict] = field(default_factory=dict)
    liquidity_timing: dict[str, dict] = field(default_factory=dict)


def evaluate(signal: pd.DataFrame, panels: dict, days, settings: Settings) -> dict:
    signal, excess = signal.loc[days], panels["excess"].loc[days]
    ic = daily_ic(signal, excess, settings.min_names)
    book = long_short(signal, panels["open_close"].loc[days], settings.cost_bps, settings.min_names)
    held = book[book["active"]]
    return {"ic": ic_summary(ic), "pearson_ic": ic_summary(daily_ic(signal, excess, settings.min_names,
                                                                     rank=False)),
            "gross": performance(book["gross"]), "net": performance(book["net"]),
            "days_held": int(book["active"].sum()),
            "break_even_bps": float(held["gross"].mean() / 4 * 1e4) if len(held) else np.nan,
            "ic_series": ic}


def run(articles: pd.DataFrame, prices: pd.DataFrame, universe: list[str],
        aliases: dict[str, str] | None = None, start: str = "2020-01-01",
        settings: Settings = Settings(), no_prices: list[str] | None = None) -> Result:
    panels = return_panels(prices, universe)
    enough = panels["open_close"].notna().sum(axis=1) >= settings.min_names
    days = panels["open_close"].index[enough & (panels["open_close"].index >= pd.Timestamp(start))]
    if len(days) <= settings.initial_train_days + settings.embargo_days:
        raise ValueError(f"Only {len(days)} trading days from {start}: not enough to train and test")

    items, counts = prepare_articles(articles, universe, aliases or {}, settings)
    items = items.join(assign_sessions(items["known_at"], days, settings))
    counts["dropped: no session for it in the test period"] = int(items["session"].isna().sum())
    audit = audit_alignment(items, days, settings)
    items[["pos", "neg"]] = [lexicon_counts(h) for h in items["headline"]]

    sentiment, count = sentiment_panel(items, days, universe)
    overnight, _ = sentiment_panel(items[items["bucket"] == "overnight"], days, universe)
    prior, _ = sentiment_panel(items[items["bucket"] == "prior_session"], days, universe)
    traded = panels["open_close"].loc[days].notna()
    coverage = float(((count > 0) & traded).sum().sum() / traded.sum().sum())

    docs = build_documents(items, panels["excess"])
    predictions, folds = walk_forward_text_model(docs, days, universe, settings)
    oos = predictions.index[predictions.notna().any(axis=1)]
    audit["folds where training ended before the test block began"] = (
        f"{sum(f['train_to'] < f['test_from'] for f in folds)} of {len(folds)}")

    signals = {
        "word-list sentiment, all news since the last decision": sentiment,
        "word-list sentiment, overnight news only": overnight,
        "word-list sentiment, previous session's news only": prior,
        "learned text model (walk-forward)": predictions,
    }
    scored = {name: evaluate(sig, panels, oos, settings) for name, sig in signals.items()}

    # Deliberately broken version, for scale: news written DURING a session
    # "predicting" that same session. This is what look-ahead looks like.
    inside = items.assign(same_day=same_day_session(items["known_at"], days, settings))
    leaky_signal, _ = sentiment_panel(inside, days, universe, session_col="same_day")
    leaky = evaluate(leaky_signal, panels, oos, settings)

    main = ["word-list sentiment, all news since the last decision", "learned text model (walk-forward)"]
    excess_oos = panels["excess"].loc[oos]
    permutation = {name: permutation_test(signals[name].loc[oos], excess_oos, settings) for name in main}
    controls = fama_macbeth(excess_oos, {
        "news sentiment": sentiment.loc[oos], "yesterday's return": panels["ret_1d"].loc[oos],
        "5-day return": panels["ret_5d"].loc[oos], "20-day volatility": panels["vol_20d"].loc[oos]},
        settings.min_names)
    liquidity = ic_by_liquidity(sentiment.loc[oos], excess_oos, panels["dollar_volume_20d"].loc[oos])

    # Level versus timing: is it the news arriving, or just which stocks get good news?
    level = usual_level(sentiment, settings.level_window)
    surprise = sentiment - level
    level_vs_timing = {
        "each stock's usual sentiment over the past year, ignoring today's news":
            evaluate(level, panels, oos, settings),
        "today's sentiment minus that usual level (the news itself)":
            evaluate(surprise, panels, oos, settings),
    }
    liquidity_timing = ic_by_liquidity(surprise.loc[oos], excess_oos,
                                       panels["dollar_volume_20d"].loc[oos])

    yearly = pd.DataFrame({name: scored[name]["ic_series"].groupby(lambda d: d.year).mean()
                           for name in main})
    return Result(settings=settings, universe=universe, no_prices=list(no_prices or []),
                  period=(days[0], days[-1]), oos_period=(oos[0], oos[-1]), counts=counts,
                  coverage=coverage, audit=audit, folds=folds, signals=scored, leaky=leaky,
                  permutation=permutation, controls=controls, liquidity=liquidity, yearly=yearly,
                  level_vs_timing=level_vs_timing, liquidity_timing=liquidity_timing)


# ---------------------------------------------------------------- the report

def verdict(t_stat: float) -> str:
    if not np.isfinite(t_stat):
        return "could not be measured"
    if abs(t_stat) < 2:
        return "not distinguishable from zero"
    return "positive (|t| above 2)" if t_stat > 0 else "negative (|t| above 2)"


def render(r: Result) -> str:
    s = r.settings
    out = ["# News test: does past news predict the next session?", ""]
    out += [f"- Universe: {len(r.universe)} stocks" +
            (f" (no price history for {', '.join(r.no_prices)}, left out)" if r.no_prices else ""),
            f"- Period: {r.period[0].date()} to {r.period[1].date()}; scored out of sample from "
            f"{r.oos_period[0].date()} to {r.oos_period[1].date()}",
            f"- Target: open-to-close return minus the universe average, decided at "
            f"{s.decision_time} New York",
            f"- Walk-forward: train on everything up to {s.embargo_days} day(s) before each "
            f"{s.test_days}-day block, {len(r.folds)} blocks", ""]

    out += ["## Data", ""]
    out += [f"- {name}: {value:,}" for name, value in r.counts.items()]
    out += [f"- Stock-days with at least one usable headline: {r.coverage:.0%}", ""]

    out += ["## Point-in-time audit", ""]
    out += [f"- {name}: {value:,.0f}" if isinstance(value, (int, float)) else f"- {name}: {value}"
            for name, value in r.audit.items()]
    out += ["- Result: PASS (the run stops with an error if any check fails)", ""]

    out += ["## Results (out of sample)", "",
            "| Signal | Days | Rank IC | t | L/S gross /yr | Sharpe gross | L/S net /yr | Sharpe net | Break-even cost |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, e in r.signals.items():
        out.append(f"| {name} | {e['ic']['days']} | {e['ic']['mean']:+.4f} | {e['ic']['t_stat']:+.2f} | "
                   f"{e['gross']['annual_return']:+.1%} | {e['gross']['sharpe']:+.2f} | "
                   f"{e['net']['annual_return']:+.1%} | {e['net']['sharpe']:+.2f} | "
                   f"{e['break_even_bps']:+.2f} bps |")
    out += ["", f"Costs assumed: {s.cost_bps:g} bps per trade. The strategy opens and closes both "
                "sides every day, so that is 4 trades a day. Break-even cost is the cost per trade "
                "at which the gross return would be exactly used up.", ""]

    out += ["## Checks", "", "**Shuffled dates** (the result must vanish when news is paired with "
            "the wrong days):", ""]
    for name, p in r.permutation.items():
        out.append(f"- {name}: real rank IC {p['real']:+.4f}; shuffled {p['shuffled_mean']:+.4f} "
                   f"+/- {p['shuffled_std']:.4f}; p = {p['p_value']:.3f}")
    out += ["", "A shuffled result away from zero means part of the real one does not depend on "
                "timing at all. The next check separates the two.", "",
            "**Level versus timing** (is it the news arriving, or just which stocks tend to get "
            "good news? This check was added after the first run, when the shuffled result came "
            "out away from zero; the settings and results above were not changed):", ""]
    out += [f"- {name}: rank IC {e['ic']['mean']:+.4f}, t = {e['ic']['t_stat']:+.2f}, "
            f"L/S gross {e['gross']['annual_return']:+.1%} a year"
            for name, e in r.level_vs_timing.items()]
    e = r.leaky
    out += ["", "**What look-ahead would look like** (deliberately wrong: news written during a "
                "session 'predicting' that same session):", "",
            f"- Rank IC {e['ic']['mean']:+.4f}, t = {e['ic']['t_stat']:+.1f}, "
            f"L/S gross {e['gross']['annual_return']:+.0%} a year", "",
            "**Momentum and volatility controls** (return in bps per one standard deviation, "
            "same-day regression across stocks, averaged over days):", ""]
    out += [f"- {name}: {row.bps_per_sd:+.2f} bps, t = {row.t_stat:+.2f}"
            for name, row in r.controls.iterrows()]
    out += ["", "**By trading volume** (rank IC within each third of the universe; first the "
                "word-list sentiment, then the same with each stock's usual level removed):", ""]
    for name, g in r.liquidity.items():
        t = r.liquidity_timing.get(name, {})
        out.append(f"- {name}: {g['mean']:+.4f}, t = {g['t_stat']:+.2f}; usual level removed: "
                   f"{t.get('mean', float('nan')):+.4f}, t = {t.get('t_stat', float('nan')):+.2f}")
    out += ["", "**By year** (mean rank IC):", ""]
    out += [f"- {year}: " + ", ".join(f"{col.split(',')[0].split(' (')[0]} {row[col]:+.4f}"
                                       for col in r.yearly.columns)
            for year, row in r.yearly.iterrows()]

    out += ["", "## Reading", ""]
    for name in r.permutation:
        e = r.signals[name]
        out.append(f"- {name}: rank IC is {verdict(e['ic']['t_stat'])} "
                   f"(t = {e['ic']['t_stat']:+.2f}); after costs the long-short return is "
                   f"{e['net']['annual_return']:+.1%} a year.")
    for name, e in r.level_vs_timing.items():
        out.append(f"- {name}: rank IC is {verdict(e['ic']['t_stat'])} (t = {e['ic']['t_stat']:+.2f}).")
    out += ["", "About ten signals and sub-groups are scored in this report. With that many, one "
                "|t| above 2 is expected by chance alone, so a single such value is weak evidence. "
                "A real effect should also survive the shuffled-dates test and the level-versus-"
                "timing check, and be larger than the break-even cost."]
    return "\n".join(out)
