"""Historical news archive, for research only (the trading bot never reads it).

Downloads every article tagged with the given tickers from Alpaca's news API
into its own SQLite file, keeping BOTH timestamps of each article:

    created_at   when it was first published
    updated_at   when it was last revised

That second one matters. The text Alpaca serves today is the revised text, so
an article edited long after publication cannot be treated as "known" at its
publication time. The news test (tradebot/newstest.py) uses updated_at to
decide when a piece of text can first be used.

The download is resumable: finished months are remembered and skipped.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from tradebot.retry import RetryPolicy, call_with_retry

URL = "https://data.alpaca.markets/v1beta1/news"
PAGE_SIZE = 50
MIN_INTERVAL_S = 0.35     # between requests; keeps us under Alpaca's 200 requests a minute
POLICY = RetryPolicy(attempts=6, base_delay=2.0, rate_limit_delay=15.0, max_delay=60.0)

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id         INTEGER PRIMARY KEY,   -- the provider's article id
    created_at TEXT NOT NULL,         -- first published (UTC, ISO)
    updated_at TEXT NOT NULL,         -- last revised (UTC, ISO)
    headline   TEXT NOT NULL,
    summary    TEXT,
    source     TEXT,
    symbols    TEXT NOT NULL          -- JSON list of every ticker the article is tagged with
);
CREATE INDEX IF NOT EXISTS articles_updated ON articles (updated_at);
CREATE TABLE IF NOT EXISTS finished_months (
    month   TEXT NOT NULL,            -- YYYY-MM
    symbols TEXT NOT NULL,            -- the ticker list it was downloaded for
    n       INTEGER NOT NULL,
    PRIMARY KEY (month, symbols)
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    """First day of every month that overlaps [start, end)."""
    first = pd.Timestamp(start).normalize().replace(day=1)
    return list(pd.date_range(first, pd.Timestamp(end), freq="MS"))


def _get_page(session, headers: dict, params: dict) -> dict:
    response = session.get(URL, headers=headers, params=params, timeout=30)
    response.raise_for_status()     # the retry layer reads the status off the error
    return response.json()


def save_articles(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO articles VALUES (?,?,?,?,?,?,?)",
        [(int(r["id"]), r["created_at"], r["updated_at"], r.get("headline") or "",
          r.get("summary") or "", r.get("source") or "", json.dumps(r.get("symbols") or []))
         for r in rows if r.get("headline")],
    )
    conn.commit()


def download(conn: sqlite3.Connection, key: str, secret: str, symbols: list[str], start: str,
             end: str | None = None, say: Callable[[str], None] = print, session=None,
             sleep: Callable[[float], None] = time.sleep, now: pd.Timestamp | None = None) -> int:
    """Fetch all articles tagged with any of `symbols` from `start` up to `end` (default: now).

    Returns the number of articles fetched this run. Months already finished
    for the same ticker list are skipped; the current month is always refreshed.
    """
    if session is None:
        import requests
        session = requests.Session()
    now = now or pd.Timestamp.now(tz="UTC").tz_localize(None)
    end_ts = min(pd.Timestamp(end), now) if end else now
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    symbols_key = ",".join(sorted(symbols))
    done = {row[0] for row in conn.execute(
        "SELECT month FROM finished_months WHERE symbols = ?", (symbols_key,))}

    total = 0
    for month in month_starts(start, end_ts.strftime("%Y-%m-%d")):
        label = month.strftime("%Y-%m")
        if label in done:
            continue
        lo = max(month, pd.Timestamp(start))
        hi = min(month + pd.offsets.MonthBegin(1), end_ts)
        if lo >= hi:
            continue
        params = {"symbols": symbols_key, "start": lo.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": (hi - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "limit": PAGE_SIZE, "sort": "asc", "include_content": "false"}
        count, token = 0, None
        while True:
            page = dict(params, page_token=token) if token else params
            body = call_with_retry(_get_page, session, headers, page, policy=POLICY,
                                   sleep=sleep, what=f"alpaca news archive {label}")
            rows = body.get("news") or []
            save_articles(conn, rows)
            count += len(rows)
            token = body.get("next_page_token")
            sleep(MIN_INTERVAL_S)
            if not token:
                break
        total += count
        if month + pd.offsets.MonthBegin(1) <= now:      # only a month that is over is final
            conn.execute("INSERT OR REPLACE INTO finished_months VALUES (?,?,?)",
                         (label, symbols_key, count))
            conn.commit()
        say(f"  {label}: {count:>5} articles")
    return total


def load_articles(conn: sqlite3.Connection) -> pd.DataFrame:
    """Every stored article: id, created_at, updated_at (UTC), headline, summary, symbols (list)."""
    df = pd.read_sql_query(
        "SELECT id, created_at, updated_at, headline, summary, symbols FROM articles", conn)
    for col in ("created_at", "updated_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["symbols"] = df["symbols"].map(json.loads)
    return df
