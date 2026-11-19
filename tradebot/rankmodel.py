"""The ranking model: which stocks will do best over the next few days?

Each day every stock in the universe gets a score from its price and volume
history (tradebot/features.py). The model is trained to predict where a
stock's forward return will rank among that day's stocks.

Walk-forward, with no hindsight:

* The scores for calendar year Y come from a model trained only on days whose
  outcome was fully known before Y began (the last `horizon + 1` trading days
  of the year before are left out, because their outcome runs into Y).
* A decision for day t uses bars up to t's close. Its outcome is measured from
  the next open, so the close it was decided on is never also the entry price.
* The universe is a list written down before the first test year.
* A score, once stored, is never recomputed. What the league bots saw stays put.

Model settings come from config.yaml (model.models) and were fixed before the
first test run. A model sees the price and volume inputs (tradebot/features.py)
unless its settings say `inputs: [price, fundamentals]`, which adds the numbers
from company filings (tradebot/fundamentals.py); `kind` then says which model
it is (ridge or gbm).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tradebot import features, fundamentals
from tradebot.newstest import daily_ic, verdict

MIN_NAMES = 50                 # a day with fewer scored stocks than this is not evaluated
MIN_HISTORY_FEATURE = "ret_21d"   # a stock enters once it has a month of bars
INPUTS = {"price": features.NAMES, "fundamentals": fundamentals.NAMES}
DESCRIPTIONS = {**features.FEATURES, **fundamentals.FEATURES}


@dataclass
class Dataset:
    days: pd.DatetimeIndex
    symbols: list[str]
    x: np.ndarray                  # (days, symbols, inputs): each day's ranks, NaN where an input is missing
    valid: np.ndarray              # (days, symbols): the stock traded that day and has a month of history
    target: np.ndarray             # (days, symbols): rank of the forward return within the day; NaN if unknown
    forward: pd.DataFrame          # the forward returns themselves
    raw: dict[str, pd.DataFrame]   # unranked inputs, for the single-input table
    horizon: int
    names: list[str] = field(default_factory=lambda: list(features.NAMES))   # the inputs in x, in order


def build_dataset(panels: dict[str, pd.DataFrame], horizon: int,
                  extra: dict[str, pd.DataFrame] | None = None) -> Dataset:
    """`extra`: more inputs as (days x symbols) tables, e.g. fundamentals.compute(...)."""
    close = panels["close"]
    raw = features.compute(panels)
    for name, frame in (extra or {}).items():
        raw[name] = frame.reindex(index=close.index, columns=close.columns).astype(float)
    names = list(raw)
    valid = close.notna() & raw[MIN_HISTORY_FEATURE].notna()
    raw = {name: frame.where(valid) for name, frame in raw.items()}
    x = np.stack([features.rank_rows(raw[name]).to_numpy(np.float32) for name in names], axis=2)
    forward = features.forward_return(panels["open"], horizon).where(valid)
    target = features.rank_rows(forward).to_numpy(np.float32)
    return Dataset(close.index, list(close.columns), x, valid.to_numpy(), target, forward, raw, horizon, names)


# ---------------------------------------------------------------- which model sees what

def model_spec(name: str, settings: dict | None) -> tuple[str, list[str], dict]:
    """(kind, input groups, the estimator's own settings) for one `model.models` entry.

    `ridge: {alpha: 100}` is a ridge model on the price inputs. `ridge-fund:
    {kind: ridge, inputs: [price, fundamentals], alpha: 100}` is the same model
    given the company-filing inputs too."""
    params = dict(settings or {})
    kind = params.pop("kind", name)
    groups = list(params.pop("inputs", ["price"]))
    unknown = [g for g in groups if g not in INPUTS]
    if unknown:
        raise ValueError(f"Model {name!r}: unknown inputs {unknown}. Options: {list(INPUTS)}")
    return kind, groups, params


def model_inputs(name: str, settings: dict | None) -> list[str]:
    return [n for group in model_spec(name, settings)[1] for n in INPUTS[group]]


def needs_fundamentals(model_cfg: dict) -> bool:
    return any("fundamentals" in model_spec(name, settings)[1]
               for name, settings in (model_cfg.get("models") or {}).items())


def _columns(name: str, settings: dict | None, data: Dataset) -> np.ndarray:
    wanted = model_inputs(name, settings)
    missing = [n for n in wanted if n not in data.names]
    if missing:
        raise ValueError(f"Model {name!r} needs inputs that were not loaded ({', '.join(missing[:3])}"
                         f"{', ...' if len(missing) > 3 else ''}). For company filings, run "
                         "`python main.py fundamentals-fetch` first.")
    return np.array([data.names.index(n) for n in wanted])


# ---------------------------------------------------------------- walk-forward schedule

def folds(data: Dataset, first_test_year: int) -> list[dict]:
    """One fold per calendar year from first_test_year: which days train its model, which days it scores."""
    out = []
    years = data.days.year
    for year in range(first_test_year, int(years.max()) + 1):
        test = np.flatnonzero(years == year)
        before = np.flatnonzero(years < year)
        if len(test) == 0 or len(before) == 0:
            continue
        # Day i's outcome ends at the open of day i + 1 + horizon. It must be on or before the
        # last day of the year before, so that nothing from the test year leaks into training.
        last_train = before[-1] - (1 + data.horizon)
        if last_train < 0:
            continue
        usable = data.valid[:last_train + 1] & np.isfinite(data.target[:last_train + 1])
        out.append({"year": year, "last_train": int(last_train), "test": test,
                    "trained_through": data.days[last_train], "outcomes_known_by": data.days[before[-1]],
                    "first_test_day": data.days[test[0]], "train_rows": int(usable.sum())})
    return out


def make_model(kind: str, settings: dict):
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=float(settings.get("alpha", 1.0)))
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(random_state=0, **settings)
    raise ValueError(f"Unknown model {kind!r}. Options: ridge, gbm")


def _prepare(kind: str, x: np.ndarray) -> np.ndarray:
    """Ridge cannot take missing values: a missing rank becomes 0, the middle. The trees handle NaN themselves."""
    return np.nan_to_num(x, nan=0.0) if kind == "ridge" else x


def fit_fold(name: str, settings: dict, data: Dataset, fold: dict):
    kind, _, params = model_spec(name, settings)
    columns = _columns(name, settings, data)
    rows = data.valid[:fold["last_train"] + 1] & np.isfinite(data.target[:fold["last_train"] + 1])
    x = data.x[:fold["last_train"] + 1][rows][:, columns]
    y = data.target[:fold["last_train"] + 1][rows]
    return make_model(kind, params).fit(_prepare(kind, x), y)


def fingerprint(name: str, settings: dict, data: Dataset, fold: dict) -> str:
    """Changes whenever the trained model would: settings, inputs, horizon, the training period,
    or the scikit-learn version (a model pickled by one version may load wrongly in another).
    A price-only model's fingerprint is the same as before company filings were added."""
    import sklearn
    text = json.dumps({"model": name, "settings": settings, "features": model_inputs(name, settings),
                       "sklearn": sklearn.__version__,
                       "horizon": data.horizon, "trained_through": str(fold["trained_through"].date()),
                       "rows": fold["train_rows"]}, sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def get_model(name: str, settings: dict, data: Dataset, fold: dict, cache_dir: Path | None, say=print):
    """The fold's fitted model, from the cache if this exact model was trained before."""
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{name}_{fold['year']}_{fingerprint(name, settings, data, fold)}.joblib"
        if path.exists():
            try:
                import joblib
                return joblib.load(path)
            except Exception as error:  # noqa: BLE001 - a cache that cannot be read is simply rebuilt
                say(f"  could not read {path.name} ({type(error).__name__}); training again")
    model = fit_fold(name, settings, data, fold)
    if path is not None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
    return model


def score_days(name: str, settings: dict, model, data: Dataset, day_indexes) -> pd.DataFrame:
    """Scores for the given days as rows of (date, symbol, score)."""
    kind = model_spec(name, settings)[0]
    columns = _columns(name, settings, data)
    parts = []
    for i in day_indexes:
        rows = data.valid[i]
        if rows.sum() == 0:
            continue
        scores = model.predict(_prepare(kind, data.x[i][rows][:, columns]))
        parts.append(pd.DataFrame({"date": data.days[i].strftime("%Y-%m-%d"),
                                   "symbol": np.asarray(data.symbols)[rows], "score": scores.astype(float)}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "symbol", "score"])


# ---------------------------------------------------------------- stored scores

def stored_days(conn, model: str) -> set[str]:
    return {row[0] for row in conn.execute("SELECT DISTINCT date FROM model_scores WHERE model = ?", (model,))}


def save_scores(conn, model: str, rows: pd.DataFrame) -> int:
    """Add scores. A score already stored for that model, day and symbol is left as it is."""
    cur = conn.executemany("INSERT OR IGNORE INTO model_scores VALUES (?,?,?,?)",
                           [(model, d, s, float(v)) for d, s, v in rows.itertuples(index=False)])
    conn.commit()
    return cur.rowcount


def load_scores(conn, model: str) -> pd.DataFrame:
    """One model's scores as a (days x symbols) table."""
    rows = pd.read_sql_query("SELECT date, symbol, score FROM model_scores WHERE model = ?", conn,
                             params=(model,), parse_dates=["date"])
    return rows.pivot(index="date", columns="symbol", values="score").sort_index() if len(rows) else pd.DataFrame()


def update_scores(conn, data: Dataset, model_cfg: dict, cache_dir: Path | None = None, say=print) -> dict[str, int]:
    """Score every day from first_test_year on that has no stored score yet. Returns new rows per model."""
    added = {}
    plan = folds(data, int(model_cfg["first_test_year"]))
    for name, settings in model_cfg["models"].items():
        have = stored_days(conn, name)
        added[name] = 0
        for fold in plan:
            missing = [i for i in fold["test"] if data.days[i].strftime("%Y-%m-%d") not in have]
            if not missing:
                continue
            model = get_model(name, settings or {}, data, fold, cache_dir, say)
            rows = score_days(name, settings or {}, model, data, missing)
            added[name] += save_scores(conn, name, rows)
            say(f"  {name} {fold['year']}: trained on {fold['train_rows']:,} stock-days up to "
                f"{fold['trained_through']:%Y-%m-%d}; scored {len(missing)} day(s)")
    return added


# ---------------------------------------------------------------- evaluation

def overlap_stats(values: pd.Series, horizon: int) -> dict:
    """Average and t-statistic of a daily series whose values overlap for `horizon` days.

    Each day starts a `horizon`-day outcome, so neighbouring days share most of
    their outcome and are not independent. The standard error therefore comes
    from windows that do not overlap (every `horizon`-th day), averaged over
    the `horizon` possible starting days.
    """
    values = values.dropna()
    errors = []
    for k in range(horizon):
        part = values.iloc[k::horizon]
        if len(part) > 2 and part.std() > 0:
            errors.append(part.std() / np.sqrt(len(part)))
    if not errors:
        return {"mean": np.nan, "t_stat": np.nan, "windows": 0, "share_positive": np.nan}
    return {"mean": float(values.mean()), "t_stat": float(values.mean() / np.mean(errors)),
            "windows": len(values) // horizon, "share_positive": float((values > 0).mean())}


def quintile_excess(scores: pd.DataFrame, forward: pd.DataFrame, horizon: int) -> list[float]:
    """Yearly return of each fifth of the stocks (lowest scores first), minus the average stock."""
    scores, forward = scores.align(forward, join="inner")
    both = scores.notna() & forward.notna()
    scores, forward = scores.where(both), forward.where(both)
    enough = both.sum(axis=1) >= MIN_NAMES
    bucket = np.ceil(scores.rank(axis=1, pct=True) * 5)
    average = forward.mean(axis=1)
    return [float((forward.where(bucket == k).mean(axis=1) - average)[enough].mean() * 252 / horizon)
            for k in range(1, 6)]


def top_n_excess(scores: pd.DataFrame, forward: pd.DataFrame, horizon: int, top_n: int,
                 cost_bps: float) -> pd.DataFrame:
    """Holding the `top_n` highest scores for `horizon` days, against holding every stock equally.

    One row per starting day: gross and net excess return over the period, and
    turnover (the share of the holdings replaced since the previous rebalance).
    Replacing a holding is a sale and a purchase, so it costs 2 x cost_bps.
    The picks do not depend on the outcome being known: a pick that stops
    trading inside the window is left out of that window's average.
    """
    scores, forward = scores.align(forward, join="inner")
    scored = scores.notna()
    picks = scores.rank(axis=1, ascending=False, method="first") <= top_n
    held = forward.where(picks).mean(axis=1)
    everyone = forward.where(scored).mean(axis=1)
    kept = (picks & picks.shift(horizon, fill_value=False)).sum(axis=1)
    turnover = (1 - kept / top_n).where(picks.shift(horizon, fill_value=False).any(axis=1), 1.0)
    gross = held - everyone
    out = pd.DataFrame({"gross": gross, "net": gross - 2 * turnover * cost_bps / 1e4, "turnover": turnover,
                        "held": held, "everyone": everyone})
    return out[(scored.sum(axis=1) >= MIN_NAMES) & held.notna()]


def evaluate(scores: pd.DataFrame, data: Dataset, top_n: int, cost_bps: float) -> dict:
    h = data.horizon
    ic = daily_ic(scores, data.forward, min_names=MIN_NAMES)
    top = top_n_excess(scores, data.forward, h, top_n, cost_bps)
    per_year = 252 / h
    return {
        "days": len(ic), "first_day": ic.index.min(), "last_day": ic.index.max(),
        "ic_daily": ic,
        "ic": overlap_stats(ic, h),
        "ic_by_year": ic.groupby(ic.index.year).mean().to_dict(),
        "quintiles": quintile_excess(scores, data.forward, h),
        "top_gross": {**overlap_stats(top["gross"], h), "per_year": float(top["gross"].mean() * per_year)},
        "top_net": {**overlap_stats(top["net"], h), "per_year": float(top["net"].mean() * per_year)},
        "top_net_by_year": (top["net"].groupby(top.index.year).mean() * per_year).to_dict(),
        "turnover": float(top["turnover"].mean()),
        "held_per_year": float(top["held"].mean() * per_year),
        "everyone_per_year": float(top["everyone"].mean() * per_year),
    }


def single_inputs(data: Dataset, days: pd.DatetimeIndex) -> pd.DataFrame:
    """How well each input ranks stocks on its own, over the test days."""
    rows = []
    forward = data.forward.loc[days]
    for name in data.names:
        stats = overlap_stats(daily_ic(data.raw[name].loc[days], forward, min_names=MIN_NAMES), data.horizon)
        group, meaning = DESCRIPTIONS.get(name, ("other", name))
        rows.append({"input": name, "group": group, "meaning": meaning, "ic": stats["mean"], "t_stat": stats["t_stat"]})
    return pd.DataFrame(rows)


def input_coverage(data: Dataset, names: list[str], days: pd.DatetimeIndex) -> pd.DataFrame:
    """Share of scored stock-days that have each input, by year (rows: years, columns: inputs)."""
    valid = pd.DataFrame(data.valid, index=data.days, columns=data.symbols).loc[days]
    per_year = {name: data.raw[name].loc[days].notna().groupby(days.year).sum().sum(axis=1)
                for name in names}
    stock_days = valid.groupby(days.year).sum().sum(axis=1)
    return pd.DataFrame({name: counts / stock_days for name, counts in per_year.items()})


def look_ahead_example(panels: dict[str, pd.DataFrame], data: Dataset, days: pd.DatetimeIndex) -> dict:
    """A deliberately wrong signal: the NEXT day's return, which is not known at the decision."""
    cheat = (panels["close"].shift(-1) / panels["close"] - 1).loc[days]
    return overlap_stats(daily_ic(cheat, data.forward.loc[days], min_names=MIN_NAMES), data.horizon)


# ---------------------------------------------------------------- report

def audit(plan: list[dict]) -> list[str]:
    """Raises if any fold could have seen its own test period. Returns the lines for the report."""
    lines = []
    for fold in plan:
        if not (fold["trained_through"] < fold["outcomes_known_by"] < fold["first_test_day"]):
            raise AssertionError(f"Fold {fold['year']} trains on outcomes from its own test year.")
        lines.append(f"| {fold['year']} | {fold['trained_through']:%Y-%m-%d} | {fold['outcomes_known_by']:%Y-%m-%d} "
                     f"| {fold['first_test_day']:%Y-%m-%d} | {fold['train_rows']:,} |")
    return lines


def _data_section(info: dict) -> list[str]:
    breaks = info.get("breaks")
    if breaks is None or not len(breaks):
        return []
    ratio = info["break_ratio"]
    listed = ", ".join(f"{row.symbol} {row.date:%Y-%m-%d} (x{row.move:.2f})"
                       for row in breaks.sort_values("date").itertuples())
    return ["## Data", "",
            f"A close more than {ratio:g} times the previous one, or less than 1/{ratio:g} of it, is treated as a "
            "break in the data, not a return: a bankrupt company's new shares taking over the ticker, or a "
            "spin-off or split that arrived unadjusted. The stock's history ends the day before, as if it had "
            f"stopped trading. {len(breaks)} series end this way: {listed}.", ""]


def twins(model_cfg: dict) -> dict[str, str]:
    """Each model that sees company filings -> the price-only model with the same kind and settings."""
    specs = {name: model_spec(name, settings) for name, settings in (model_cfg.get("models") or {}).items()}
    out = {}
    for name, (kind, groups, params) in specs.items():
        if "fundamentals" in groups:
            out.update({name: other for other, (k, g, p) in specs.items()
                        if g == ["price"] and k == kind and p == params})
    return out


def _inputs_lines(info: dict, count: int) -> list[str]:
    groups = info.get("model_groups")
    if not groups:
        return [f"- Inputs: {count} numbers per stock per day, each replaced by its rank among that day's stocks"]
    words = {"price": f"{len(features.NAMES)} from price and volume",
             "fundamentals": f"{len(fundamentals.NAMES)} from company filings"}
    same: dict[tuple, list[str]] = {}
    for name, used in groups.items():
        same.setdefault(tuple(used), []).append(name)
    parts = [f"{', '.join(names)}: {' plus '.join(words[g] for g in used)}" for used, names in same.items()]
    return ["- Inputs, each replaced by its rank among that day's stocks. " + "; ".join(parts)]


def _filings_section(info: dict, results: dict) -> list[str]:
    coverage = info.get("fund_coverage")
    if coverage is None:
        return []
    out = ["## Company filings", "",
           f"The models marked `-fund` also see {len(fundamentals.NAMES)} numbers from each company's 10-K and 10-Q "
           f"filings (SEC EDGAR's XBRL company facts; {info.get('fund_companies', 'n/a')}). They are dated by when "
           "they became public: a filing dated D is first used after the close of the next trading day; a restated "
           "number replaces the old one only from the date of the filing that restated it; a number for a period "
           "that ended more than a year earlier counts as unknown. Market value is the public float each 10-K "
           "reports, carried forward by the stock's own return since the float's date. Each `-fund` model has the "
           "same settings as its price-only twin, so the inputs are the only difference.", "",
           "Share of scored stock-days that have each input:", "",
           "| Year | " + " | ".join(coverage.columns) + " |", "|---|" + "---|" * len(coverage.columns)]
    for year, row in coverage.iterrows():
        out.append(f"| {year} | " + " | ".join(f"{v:.0%}" for v in row) + " |")
    pairs = [(name, twin) for name, twin in info.get("twins", {}).items() if name in results and twin in results]
    if pairs:
        out += ["", "**With and without the filings** (each day's rank IC of the `-fund` model minus its "
                    "price-only twin's, over the same days)", "",
                "| Models | Rank IC difference | t | Days the filings helped |", "|---|---|---|---|"]
        for name, twin in pairs:
            diff = (results[name]["ic_daily"] - results[twin]["ic_daily"]).dropna()
            stats = overlap_stats(diff, info["horizon"])
            out.append(f"| {name} minus {twin} | {_num(stats['mean'], 4)} | {_num(stats['t_stat'])} | "
                       f"{stats['share_positive']:.0%} |")
        out += ["", "A t above 2 would mean the filings added ranking skill beyond what price and volume already "
                    "carried; below 2, the difference could be luck."]
    return out + [""]


def _pct(value, digits: int = 1) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:+.{digits}%}"


def _num(value, digits: int = 2) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:+.{digits}f}"


def render(results: dict[str, dict], inputs: pd.DataFrame, cheat: dict, plan: list[dict], info: dict) -> str:
    h, n = info["horizon"], info["top_n"]
    first = next(iter(results.values()))
    filings = info.get("fund_coverage") is not None
    out = [
        "# Ranking model test: can " + ("price, volume and company filings" if filings else
                                        "price and volume history") + " rank next week's winners?",
        "",
        f"- Universe: {info['symbols']} stocks, the S&P 500 as it was listed at the end of 2015 "
        f"({info['ended']} of them stopped trading, or hit a data break, before today; they are kept in the "
        f"test up to their last day)",
        f"- Prices: {info['first_bar']} to {info['last_bar']}; scored out of sample from "
        f"{first['first_day']:%Y-%m-%d} to {first['last_day']:%Y-%m-%d}",
        f"- Decision: after each day's close. Outcome: buy at the next open, sell at the open {h} trading days later",
        *_inputs_lines(info, len(inputs)),
        "- Walk-forward: each year's scores come from a model trained only on earlier years",
        "",
        *_data_section(info),
        *_filings_section(info, results),
        "## Point-in-time audit",
        "",
        "| Test year | Trained on days up to | Whose outcomes were known by | First test day | Stock-days trained on |",
        "|---|---|---|---|---|",
        *audit(plan),
        "",
        "Result: PASS (the run stops with an error if a model could have seen its own test year).",
        "",
        "## Results (out of sample)",
        "",
        f"| Model | Days | Rank IC | t | Days positive | Top {n} vs all, before costs /yr | After costs /yr | t | "
        f"Replaced each rebalance |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        out.append(f"| {name} | {r['days']} | {_num(r['ic']['mean'], 4)} | {_num(r['ic']['t_stat'])} | "
                   f"{r['ic']['share_positive']:.0%} | {_pct(r['top_gross']['per_year'])} | "
                   f"{_pct(r['top_net']['per_year'])} | {_num(r['top_net']['t_stat'])} | {r['turnover']:.0%} |")
    out += [
        "",
        f"Rank IC: the correlation, across stocks, between the score and what happened next (0 = no skill; "
        f"0.02 to 0.05 is what professionals call good). \"Top {n} vs all\" is the yearly return of holding the "
        f"{n} highest-scored stocks, rebalanced every {h} days, minus the return of holding every stock equally. "
        f"Costs: {info['cost_bps']:g} bps per trade. The t-statistics count each {h}-day window once.",
        "",
        "**Verdicts**",
        "",
    ]
    for name, r in results.items():
        out.append(f"- {name}: ranking skill is {verdict(r['ic']['t_stat'])}; "
                   f"the top {n} after costs are {verdict(r['top_net']['t_stat'])} against holding everything")
    out += ["", "**Stocks sorted into fifths by score** (yearly return minus the average stock; a model with "
                "skill shows a rising line from left to right)", "",
            "| Model | Lowest fifth | 2nd | 3rd | 4th | Highest fifth |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        out.append(f"| {name} | " + " | ".join(_pct(q) for q in r["quintiles"]) + " |")
    years = sorted({year for r in results.values() for year in r["ic_by_year"]})
    out += ["", "**Year by year** (rank IC; top-" + str(n) + " return after costs minus the average stock)", "",
            "| Model | " + " | ".join(str(y) for y in years) + " |", "|---|" + "---|" * len(years)]
    for name, r in results.items():
        out.append(f"| {name}, rank IC | " + " | ".join(_num(r["ic_by_year"].get(y), 3) for y in years) + " |")
        out.append(f"| {name}, top {n} | " + " | ".join(_pct(r["top_net_by_year"].get(y)) for y in years) + " |")
    out += [
        "",
        "## Each input on its own",
        "",
        "Rank IC of every input by itself over the same days. A negative value means stocks with a high "
        "reading tended to do worse. No model is involved here, so nothing was fitted.",
        "",
        "| Input | What it measures | Rank IC | t |",
        "|---|---|---|---|",
    ]
    for group, part in inputs.groupby("group", sort=False):
        out.append(f"| **{group}** | | | |")
        for row in part.itertuples():
            out.append(f"| {row.input} | {row.meaning} | {_num(row.ic, 4)} | {_num(row.t_stat)} |")
    strong = int((inputs["t_stat"].abs() > 2).sum())
    out += [
        "",
        f"{strong} of {len(inputs)} inputs have |t| above 2. With {len(inputs)} inputs tested, about "
        f"{len(inputs) * 0.05:.0f} would cross that line by luck alone, so a single strong row proves little.",
        "",
        "## What look-ahead would look like",
        "",
        "A deliberately wrong signal, the next day's own return (not known when the decision is made): "
        f"rank IC {_num(cheat['mean'], 4)}, t = {_num(cheat['t_stat'], 1)}. A result of that size from an "
        "honest signal would mean a leak, not a discovery.",
        "",
        "## Limits",
        "",
        "- The inputs and model settings were chosen once, before this test was run. They still reflect what "
        "is known today about which signals have worked, so the test is not perfectly free of hindsight.",
        "- A stock's last few days before it stops trading have no measured outcome and are left out. "
        "For a company that went bankrupt, that hides part of the final loss.",
        "- Fills are assumed at the opening price. A real order moves the price a little, more so in small stocks.",
        "- The price history is not perfectly clean. Breaks larger than the limit in the Data section are cut "
        "out, but a few smaller ones remain: some spin-offs appear as a one-day fall that holders did not "
        "really suffer, and one company split-up appears as a one-day rise. They move the dollar returns "
        "above more than the rank IC, which only looks at the order of the stocks.",
        "- The live record of the league's rank bots, which started after this test, is the evidence that counts.",
    ]
    if filings:
        out[-1:-1] = [
            "- Company filings: the public float leaves out shares held by insiders, so founder-controlled "
            "companies look smaller, and cheaper, than they are.",
            "- A company that moved its filings to a new SEC number is followed there only if it is listed in "
            "universes/sp500_2016_cik_successors.csv (21 tickers, such as Google to Alphabet). One that is missed, "
            "or whose successor files in IFRS rather than US GAAP, has no filing data after the move: a number "
            "older than a year counts as unknown, not as current.",
            "- SEC's company-facts files lag for a few companies (in November 2026, no 10-Q from mid-2026 for about "
            "ten, including Citigroup and NextEra); their numbers age until they drop out.",
            "- Companies tag the same line differently, and a few tag it wrongly. The coverage table shows how much "
            "is missing; a wrong tag that slipped through is not caught.",
            "- The nine filing inputs come from published research that is known today, so their choice is not "
            "perfectly free of hindsight either.",
        ]
    return "\n".join(out)
