"""Prices for the ranking model's wide universe (a few hundred stocks), from Alpaca.

They live in their own database (config: model.prices_path), apart from the
bot's prices, for two reasons:

* One source per series. The bot's prices come from Yahoo. Mixing two
  sources' dividend adjustments inside one price history would put small
  jumps in it.
* The universe is a list of tickers as they were on a past day (`asof`).
  Alpaca follows each one through later renames and up to the day it stopped
  trading, so companies that were bought or went bankrupt stay in the test.
  A list of today's survivors would flatter every result.

A ticker that only starts trading well after the start date is dropped: it
means the original company was already gone and the letters were later
reused by a different one.

Ended series. Once a symbol has had no bar for longer than the refresh
window, it is never asked about again. This matters: asked for recent days
only, Alpaca answers with whichever company uses those letters today, which
for a company that ended years ago is a different one. (Alpaca's own history
for some tickers also stops at a reorganisation the company survived, such as
LMT in 2016. Those are treated like any other stock that stopped trading.)

Breaks. Following a ticker through eleven years of corporate history is not
always clean. A bankrupt company's cancelled shares can be followed by the
new company's shares under the same ticker (a "rise" of several thousand
percent that no holder received), and some spin-offs and splits arrive
unadjusted (a "fall" of 80 or 90 percent that no holder suffered). A close
more than `break_ratio` times the previous one, or less than 1/break_ratio of
it, is treated as such a break: the stock's history ends the day before, as
if it had stopped trading. Nothing before the break changes, so the rule uses
no hindsight. The price of the rule is that a real one-day move that large
would be cut too.
"""
from __future__ import annotations

import pandas as pd

from tradebot import db
from tradebot.data import PRICE_COLUMNS, first_incomplete_date, history_was_readjusted
from tradebot.retry import call_with_retry

LATE_START_DAYS = 21        # a first bar later than this after `start` is treated as a reused ticker
OVERLAP_DAYS = 15           # how far back a refresh re-reads, to notice re-adjusted history


def fetch_bars(symbols: list[str], start: str, key: str, secret: str, asof: str | None = None,
               chunk: int = 50) -> pd.DataFrame:
    """Daily split/dividend-adjusted bars for many symbols: symbol, date, open, high, low, close, volume."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from tradebot.broker import disable_builtin_retry

    client = StockHistoricalDataClient(key, secret)
    disable_builtin_retry(client)
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)).to_pydatetime()   # free plan: 15 min delay
    parts = []
    for i in range(0, len(symbols), chunk):
        group = [s.upper() for s in symbols[i:i + chunk]]
        extra = {"asof": asof} if asof else {}
        request = StockBarsRequest(symbol_or_symbols=group, timeframe=TimeFrame.Day,
                                   start=pd.Timestamp(start, tz="UTC").to_pydatetime(), end=end,
                                   adjustment=Adjustment.ALL, feed=DataFeed.SIP, **extra)
        bars = call_with_retry(client.get_stock_bars, request, what=f"alpaca bars {group[0]}..{group[-1]}").df
        if not bars.empty:
            parts.append(bars.reset_index())
    if not parts:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    raw = pd.concat(parts, ignore_index=True)
    dates = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    out = raw[["symbol", "open", "high", "low", "close", "volume"]].copy()
    out.insert(1, "date", dates.to_numpy())
    out[PRICE_COLUMNS[2:]] = out[PRICE_COLUMNS[2:]].astype(float)
    return out.dropna(subset=["close"]).reset_index(drop=True)[PRICE_COLUMNS]


def stored_range(conn) -> dict[str, tuple[str, str, int]]:
    """symbol -> (first date, last date, number of bars)."""
    return {sym: (first, last, n) for sym, first, last, n in
            conn.execute("SELECT symbol, MIN(date), MAX(date), COUNT(*) FROM prices GROUP BY symbol")}


def update_prices(conn, symbols: list[str], start: str, asof: str | None, fetch, say=print,
                  full: bool = False, now: pd.Timestamp | None = None) -> dict:
    """Bring the model's price database up to date.

    fetch(symbols, start) -> bars. The first run (or full=True) downloads
    everything. Later runs re-read the last few days for every symbol that is
    still trading, add the new bars, and download again in full only the
    symbols whose past prices changed (a split or dividend re-adjusts the whole
    history). A symbol whose stored history ended before that window is left
    alone for good.
    """
    cutoff = first_incomplete_date("a-stock", now)           # leave out a bar that is still forming
    have = stored_range(conn)
    wanted = [s.upper() for s in symbols]

    if full or not have:
        bars = fetch(wanted, start)
        bars = bars[bars["date"] < cutoff]
        first = bars.groupby("symbol")["date"].min()
        late = (pd.Timestamp(start) + pd.Timedelta(days=LATE_START_DAYS)).strftime("%Y-%m-%d")
        reused = sorted(first[first > late].index)
        bars = bars[~bars["symbol"].isin(reused)]
        conn.execute("DELETE FROM prices")
        db.upsert_prices(conn, bars)
        kept = sorted(set(bars["symbol"]))
        missing = sorted(set(wanted) - set(kept) - set(reused))
        say(f"Downloaded {len(bars):,} daily bars for {len(kept)} of {len(wanted)} symbols, "
            f"{bars['date'].min()} to {bars['date'].max()}.")
        if reused:
            say(f"Left out {len(reused)} whose first bar comes long after {start} (the ticker was reused "
                f"by another company): {', '.join(reused)}")
        if missing:
            say(f"No data at all for {len(missing)}: {', '.join(missing)}")
        return {"symbols": len(kept), "bars": len(bars), "reused": reused, "missing": missing, "refetched": []}

    known = [s for s in wanted if s in have]
    latest = max(last for _, last, _ in have.values())
    since = (pd.Timestamp(latest) - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")
    active = [s for s in known if have[s][1] >= since]       # measured against the newest stored bar,
    fresh = fetch(active, since)                             # so a PC that was off for weeks loses nothing
    fresh = fresh[(fresh["date"] < cutoff) & fresh["symbol"].isin(active)]
    old = pd.read_sql_query("SELECT symbol, date, close FROM prices WHERE date >= ?", conn,
                            params=(since,), parse_dates=["date"])
    old_by_symbol = {sym: part.set_index("date")["close"] for sym, part in old.groupby("symbol")}
    changed = []
    for sym, part in fresh.groupby("symbol"):
        before = old_by_symbol.get(sym)
        if before is None or history_was_readjusted(before, part):
            changed.append(sym)
    added = 0
    steady = fresh[~fresh["symbol"].isin(changed)]
    if len(steady):
        before_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        db.upsert_prices(conn, steady)
        added = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0] - before_count
    if changed:
        again = fetch(changed, start)
        again = again[again["date"] < cutoff]
        for sym in set(again["symbol"]):
            conn.execute("DELETE FROM prices WHERE symbol = ?", (sym,))
        db.upsert_prices(conn, again)
        added += int((again["date"] > latest).sum())
    conn.commit()
    newest = conn.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    say(f"Model prices: {added} new bars, up to {newest}. {len(active)} of {len(known)} symbols still trade.")
    if changed:
        say(f"Past prices changed for {len(changed)} symbols (split or dividend); downloaded again in full.")
    return {"symbols": len(known), "bars": added, "reused": [], "missing": [], "refetched": changed}


def load_panels(conn, start: str | None = None) -> dict[str, pd.DataFrame]:
    """The whole price database as five tables (open, high, low, close, volume):
    one row per day, one column per symbol, NaN where a symbol has no bar."""
    where, params = ("WHERE date >= ?", (start,)) if start else ("", ())
    bars = pd.read_sql_query(f"SELECT symbol, date, open, high, low, close, volume FROM prices {where}",
                             conn, params=params, parse_dates=["date"])
    if bars.empty:
        return {}
    return {field: bars.pivot(index="date", columns="symbol", values=field).sort_index()
            for field in PRICE_COLUMNS[2:]}


def cut_at_breaks(panels: dict[str, pd.DataFrame], ratio: float) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """End each symbol's history the day before its first impossible one-day move.

    Returns the cut panels and a table of the breaks: symbol, date, move
    (that day's close divided by the previous close)."""
    close = panels["close"]
    move = close / close.ffill().shift(1)
    broken = (move > ratio) | (move < 1 / ratio)
    gone = broken.cummax()                               # True from the break day onwards
    rows = []
    for sym in broken.columns[broken.any()]:
        at = int(broken[sym].to_numpy().argmax())
        rows.append({"symbol": sym, "date": broken.index[at], "move": float(move[sym].iloc[at])})
    breaks = pd.DataFrame(rows, columns=["symbol", "date", "move"])
    return {field: frame.mask(gone) for field, frame in panels.items()}, breaks


def load_clean_panels(conn, ratio: float | None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """The price database as panels, with each broken series ended at its break."""
    panels = load_panels(conn)
    if not panels or not ratio:
        return panels, pd.DataFrame(columns=["symbol", "date", "move"])
    return cut_at_breaks(panels, float(ratio))
