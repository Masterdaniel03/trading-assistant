"""Everything the dashboard shows, worked out from the database.

Plain functions that return tables and numbers, with no web code, so each one
can be tested. The page (dashboard.py) only lays the results out.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from tradebot import db

NY = "America/New_York"
MIN_DAYS_FOR_SHARPE = 20      # fewer daily returns than this and a Sharpe ratio is noise
NEAR_STOP = 0.02              # within 2 percentage points of the stop counts as "near"


def _ny_date(ts: str) -> pd.Timestamp:
    """The New York calendar day of a stored UTC timestamp."""
    stamp = pd.Timestamp(ts)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
    return stamp.tz_convert(NY).tz_localize(None).normalize()


# ---------------------------------------------------------------- account value

def latest_snapshot(conn, account: str) -> dict | None:
    row = conn.execute("SELECT ts, equity, cash FROM equity_snapshots WHERE account = ? "
                       "ORDER BY ts DESC LIMIT 1", (account,)).fetchone()
    return {"ts": row[0], "equity": row[1], "cash": row[2]} if row else None


def equity_series(conn, account: str) -> pd.Series:
    """Account value by day: the broker's end-of-day figures, plus the latest
    snapshot if it is from a later day than the broker has closed out."""
    daily = pd.read_sql_query("SELECT date, equity FROM equity_daily WHERE account = ? ORDER BY date",
                              conn, params=(account,), parse_dates=["date"])
    series = daily.set_index("date")["equity"]
    latest = latest_snapshot(conn, account)
    if latest and (series.empty or _ny_date(latest["ts"]) > series.index[-1]):
        series.loc[_ny_date(latest["ts"])] = latest["equity"]
    return series.rename("equity")


def first_trade_date(conn, account: str) -> pd.Timestamp | None:
    row = conn.execute("SELECT MIN(ts) FROM trades WHERE account = ?", (account,)).fetchone()
    return _ny_date(row[0]) if row and row[0] else None


def compare_with_benchmark(equity: pd.Series, benchmark_close: pd.Series,
                           start: pd.Timestamp | None) -> pd.DataFrame:
    """The account next to the same money put into the benchmark at `start`.

    Both columns are in dollars and begin at the same value, so they share one
    axis. Only days that have both a broker figure and a benchmark close are
    kept. Empty if there are not yet two such days.
    """
    empty = pd.DataFrame(columns=["date", "bot", "benchmark"])
    if start is None or equity.empty or benchmark_close.empty:
        return empty
    both = pd.concat([equity.rename("bot"), benchmark_close.rename("close")], axis=1, join="inner")
    both = both[both.index >= start].dropna()
    if len(both) < 2:
        return empty
    both["benchmark"] = both["bot"].iloc[0] * both["close"] / both["close"].iloc[0]
    return both[["bot", "benchmark"]].rename_axis("date").reset_index()


def performance(values: pd.Series) -> dict:
    """Return, worst drop and (only with enough history) Sharpe ratio of a value series."""
    values = values.dropna()
    if len(values) < 2:
        return {"days": max(len(values) - 1, 0), "total_return": None, "max_drawdown": None, "sharpe": None}
    returns = values.pct_change().dropna()
    enough = len(returns) >= MIN_DAYS_FOR_SHARPE and returns.std() > 0
    return {"days": len(returns),
            "total_return": float(values.iloc[-1] / values.iloc[0] - 1),
            "max_drawdown": float((values / values.cummax() - 1).min()),
            "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)) if enough else None}


# ---------------------------------------------------------------- holdings and orders

def positions_table(conn, account: str, equity: float | None, risk: dict) -> pd.DataFrame:
    """Current holdings with how far each is from its stop-loss."""
    held = pd.read_sql_query(
        "SELECT symbol, qty, avg_entry_price, market_value, unrealized_plpc, ts FROM positions "
        "WHERE account = ? ORDER BY market_value DESC", conn, params=(account,))
    stop = risk["max_risk_per_position_pct"] / risk["max_position_pct"]
    held["share_of_account"] = held["market_value"] / equity if equity else np.nan
    held["room_before_stop"] = held["unrealized_plpc"] + stop
    held["status"] = np.select([held["room_before_stop"] <= 0, held["room_before_stop"] <= NEAR_STOP],
                               ["Stop hit", "Near stop"], default="OK")
    return held


def orders_table(conn, account: str) -> pd.DataFrame:
    orders = pd.read_sql_query(
        "SELECT ts, symbol, side, notional, qty, price, status, notes FROM trades "
        "WHERE account = ? ORDER BY id DESC", conn, params=(account,))
    orders["status"] = orders["status"].fillna("unknown")
    return orders


def waiting_orders(orders: pd.DataFrame) -> pd.DataFrame:
    return orders[~orders["status"].isin(db.SETTLED)]


# ---------------------------------------------------------------- signals and research

def signal_board(conn, strategy: str, symbols: list[str]) -> pd.DataFrame:
    """Per symbol: the bot's latest signal next to the latest kept research note."""
    rows = []
    for symbol in symbols:
        signal = conn.execute("SELECT target, ts FROM signals WHERE symbol = ? AND strategy = ? "
                              "ORDER BY id DESC LIMIT 1", (symbol, strategy)).fetchone()
        note = conn.execute("SELECT stance, confidence, summary, as_of, model FROM research "
                            "WHERE symbol = ? AND status = 'ok' ORDER BY id DESC LIMIT 1", (symbol,)).fetchone()
        target = signal[0] if signal else None
        stance = note[0] if note else None
        rows.append({"symbol": symbol, "bot_target": target, "bot_view": _bot_view(target),
                     "research_stance": stance, "confidence": note[1] if note else None,
                     "agreement": _agreement(target, stance),
                     "summary": note[2] if note else "", "prices_to": note[3] if note else None})
    return pd.DataFrame(rows)


def _bot_view(target: float | None) -> str:
    if target is None:
        return "no signal"
    return "in" if target >= 0.5 else "out"


def _agreement(target: float | None, stance: str | None) -> str:
    if target is None or stance is None:
        return "nothing to compare"
    if stance == "neutral":
        return "research is neutral"
    return "agree" if (target >= 0.5) == (stance == "bullish") else "disagree"


def research_log(conn, limit: int = 50) -> pd.DataFrame:
    """Recent research attempts, kept or not, newest first."""
    notes = pd.read_sql_query(
        "SELECT ts, symbol, status, stance, confidence, summary, reasoning, sources, detail, model "
        "FROM research ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    # Attempts that failed or were skipped have no sources (stored as NULL).
    notes["sources"] = notes["sources"].map(
        lambda s: "; ".join(json.loads(s)) if isinstance(s, str) and s else "")
    return notes


# ---------------------------------------------------------------- your own trades

def manual_review(conn, account: str = "fidelity_manual", benchmark: str = "SPY") -> tuple[pd.DataFrame, dict]:
    """Each manual trade valued at the latest close, next to what the same
    dollars would be worth in the benchmark, bought on the same day.

    Returns (per-trade table, summary). The summary splits results by whether
    the trade followed the AI's signal.
    """
    trades = pd.read_sql_query(
        "SELECT ts, symbol, side, qty, price, fees, followed_ai, notes FROM trades "
        "WHERE account = ? AND qty IS NOT NULL AND price IS NOT NULL ORDER BY ts", conn, params=(account,))
    summary = {"trades": len(trades), "follow_rate": None, "by_choice": pd.DataFrame()}
    if trades.empty:
        return trades, summary

    bench = db.load_close(conn, benchmark)
    closes = {s: db.load_close(conn, s) for s in trades["symbol"].unique()}
    rows = []
    for t in trades.itertuples():
        sign = 1 if t.side == "buy" else -1
        paid = sign * t.qty * t.price
        last = closes[t.symbol].iloc[-1] if not closes[t.symbol].empty else np.nan
        pnl = sign * t.qty * last - paid - (t.fees or 0)
        day = _ny_date(t.ts)
        bench_then = bench[bench.index <= day]
        bench_pnl = (paid * bench.iloc[-1] / bench_then.iloc[-1] - paid) if len(bench_then) else np.nan
        rows.append({"date": day.date(), "symbol": t.symbol, "side": t.side, "qty": t.qty,
                     "price": t.price, "last_close": last, "profit": pnl,
                     "benchmark_profit": bench_pnl, "followed_ai": t.followed_ai, "notes": t.notes})
    table = pd.DataFrame(rows)

    flagged = table.dropna(subset=["followed_ai"])
    if len(flagged):
        summary["follow_rate"] = float(flagged["followed_ai"].mean())
        summary["by_choice"] = (flagged.assign(choice=flagged["followed_ai"].map({1: "Followed the AI", 0: "Own call"}))
                                .groupby("choice").agg(trades=("profit", "size"), profit=("profit", "sum"),
                                                       benchmark_profit=("benchmark_profit", "sum")).reset_index())
    summary["profit"] = float(table["profit"].sum(skipna=True))
    summary["benchmark_profit"] = float(table["benchmark_profit"].sum(skipna=True))
    summary["unpriced"] = sorted(table.loc[table["last_close"].isna(), "symbol"].unique())
    return table, summary


# ---------------------------------------------------------------- is everything up to date?

def health(conn, account: str, bot_symbols: list[str]) -> dict:
    latest = latest_snapshot(conn, account)
    price_dates = {s: db.last_price_date(conn, s) for s in bot_symbols}
    known = [d for d in price_dates.values() if d]
    last_run = pd.Timestamp(latest["ts"]).tz_convert(NY) if latest else None
    return {"last_account_update": last_run,
            "oldest_price_date": min(known) if known else None,
            "symbols_without_prices": sorted(s for s, d in price_dates.items() if not d),
            "research_notes": conn.execute("SELECT COUNT(*) FROM research WHERE status='ok'").fetchone()[0],
            "stops_fired": conn.execute("SELECT COUNT(*) FROM stop_events").fetchone()[0]}


# ---------------------------------------------------------------- the bot league

BENCHMARK_LABEL = "SPY buy and hold"
PERIODS = {"Live only": None, "1 year": 365, "3 years": 3 * 365, "All history": 0}


def league_curves(conn, benchmark: str, period: str = "All history",
                  live_from: str | None = None, selected: str | None = None) -> pd.DataFrame:
    """Each bot's value, one column per bot plus the benchmark, over the chosen period.

    Every column is rescaled to start the period at the same amount (the league's
    starting cash), so the lines can be compared at a glance. "Live only" starts
    on `live_from`. Bots do not all start on the same day: the period begins no
    earlier than the `selected` bot does, and a bot with no value on the first
    day shown is left out, because its line could not start level with the rest.
    Returns a `date` column followed by one column per series.
    """
    from tradebot.league import equity_table
    equity = equity_table(conn)
    if equity.empty:
        return pd.DataFrame()
    days = PERIODS[period]
    if days is None:
        if not live_from:
            return pd.DataFrame()
        begin = pd.Timestamp(live_from)
    elif days:
        begin = equity.index.max() - pd.Timedelta(days=days)
    else:
        begin = equity.index.min()
    if selected in equity.columns:
        begin = max(begin, equity[selected].first_valid_index())
    window = equity[equity.index >= begin].dropna(how="all")
    if len(window) < 2:
        return pd.DataFrame()
    window = window.loc[:, window.iloc[0].notna()]
    base = float(equity.dropna(how="all").iloc[0].dropna().iloc[0])      # every bot starts with the same cash
    spy = db.load_close(conn, benchmark).reindex(window.index).ffill()
    out = window / window.iloc[0] * base
    if spy.notna().any():
        out[BENCHMARK_LABEL] = spy / spy.dropna().iloc[0] * base
    return out.rename_axis("date").reset_index()


def league_holdings(conn, bot: str) -> pd.DataFrame:
    table = pd.read_sql_query(
        "SELECT symbol, value, qty, entry_price, last_price FROM league_positions WHERE bot = ? "
        "ORDER BY value DESC", conn, params=(bot,))
    if table.empty:
        return table
    total = conn.execute("SELECT equity FROM league_equity WHERE bot = ? ORDER BY date DESC LIMIT 1",
                         (bot,)).fetchone()
    table["share_of_bot"] = table["value"] / total[0] if total else np.nan
    table["return_since_entry"] = table["last_price"] / table["entry_price"] - 1
    return table


def league_trades(conn, bot: str, limit: int = 40) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT date, symbol, side, qty, price, notional, reason FROM league_trades WHERE bot = ? "
        "ORDER BY date DESC, symbol LIMIT ?", conn, params=(bot, limit), parse_dates=["date"])


# ---------------------------------------------------------------- runs and alerts

def _local(ts: str | None) -> pd.Timestamp | None:
    """A stored UTC time on this PC's clock, without the zone (for display)."""
    if not isinstance(ts, str) or not ts:        # a missing value arrives as None or NaN
        return None
    stamp = pd.Timestamp(ts)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
    return stamp.tz_convert(datetime.now().astimezone().tzinfo).tz_localize(None)


def run_history(conn, limit: int = 15) -> list[dict]:
    """Recent unattended runs, newest first, each with its steps."""
    runs = []
    for row in db.runs_df(conn, limit).itertuples():
        started, finished = _local(row.started), _local(row.finished)
        steps = json.loads(row.steps) if isinstance(row.steps, str) else []
        runs.append({"session": row.session, "status": row.status, "started": started,
                     "minutes": (finished - started).total_seconds() / 60 if finished is not None else None,
                     "steps": steps,
                     "failed_steps": [s["name"] for s in steps if s["code"] != 0]})
    return runs


def run_in_progress(conn, max_hours: float = 2.0) -> bool:
    """True if a daily run started recently and has not finished (so a second one should wait)."""
    row = conn.execute("SELECT started FROM daily_runs WHERE status = 'running' ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return False
    age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(row[0])
    return age < pd.Timedelta(hours=max_hours)


def alerts_table(conn, limit: int = 50) -> pd.DataFrame:
    table = db.alerts_df(conn, limit=limit)
    if table.empty:
        return table
    table["when"] = table["ts"].map(_local)
    table["new"] = table["read"] == 0
    return table[["when", "level", "title", "detail", "delivered", "failed", "new"]]

