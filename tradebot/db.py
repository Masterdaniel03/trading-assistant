"""SQLite storage: prices, signals, trades, equity snapshots.

Everything the system sees or does is recorded here, so results are
reproducible and can be analyzed later (and shown on a resume/dashboard).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,          -- YYYY-MM-DD
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS signals (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,       -- when the signal was generated (UTC)
    symbol    TEXT NOT NULL,
    strategy  TEXT NOT NULL,       -- e.g. sma_crossover, llm_news
    target    REAL NOT NULL,       -- target position 0..1 (fraction of max size)
    reason    TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    account    TEXT NOT NULL,      -- alpaca_paper | alpaca_live | fidelity_manual
    symbol     TEXT NOT NULL,
    side       TEXT NOT NULL CHECK (side IN ('buy','sell')),
    qty        REAL,
    notional   REAL,
    price      REAL,
    fees       REAL DEFAULT 0,
    strategy   TEXT,
    followed_ai INTEGER,           -- manual trades: 1 = followed AI signal, 0 = own call
    order_id   TEXT,
    notes      TEXT
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts      TEXT NOT NULL,
    account TEXT NOT NULL,
    equity  REAL NOT NULL,
    cash    REAL,
    PRIMARY KEY (ts, account)
);
"""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


# ---------- prices ----------
def upsert_prices(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """df columns: symbol, date, open, high, low, close, volume."""
    rows = df[["symbol", "date", "open", "high", "low", "close", "volume"]].itertuples(index=False)
    cur = conn.executemany(
        "INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?)", list(rows)
    )
    conn.commit()
    return cur.rowcount


def load_close(conn: sqlite3.Connection, symbol: str) -> pd.Series:
    df = pd.read_sql_query(
        "SELECT date, close FROM prices WHERE symbol = ? ORDER BY date",
        conn, params=(symbol,), parse_dates=["date"],
    )
    return df.set_index("date")["close"].rename(symbol)


def last_price_date(conn: sqlite3.Connection, symbol: str) -> str | None:
    row = conn.execute("SELECT MAX(date) FROM prices WHERE symbol = ?", (symbol,)).fetchone()
    return row[0] if row else None


# ---------- signals / trades / equity ----------
def log_signal(conn, symbol, strategy, target, reason=None) -> bool:
    """Record a signal. Skips (returns False) if it repeats the latest signal
    for this symbol/strategy, so re-running `signals` doesn't add duplicates."""
    last = conn.execute(
        "SELECT target, reason FROM signals WHERE symbol = ? AND strategy = ? "
        "ORDER BY id DESC LIMIT 1", (symbol, strategy),
    ).fetchone()
    if last is not None and last[0] == float(target) and last[1] == reason:
        return False
    conn.execute(
        "INSERT INTO signals (ts, symbol, strategy, target, reason) VALUES (?,?,?,?,?)",
        (now_utc(), symbol, strategy, float(target), reason),
    )
    conn.commit()
    return True


def log_trade(conn, account, symbol, side, qty=None, notional=None, price=None,
              fees=0.0, strategy=None, followed_ai=None, order_id=None, notes=None,
              ts=None) -> int:
    cur = conn.execute(
        """INSERT INTO trades (ts, account, symbol, side, qty, notional, price, fees,
                               strategy, followed_ai, order_id, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (ts or now_utc(), account, symbol.upper(), side, qty, notional, price, fees,
         strategy, followed_ai, order_id, notes),
    )
    conn.commit()
    return cur.lastrowid


def log_equity(conn, account, equity, cash=None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO equity_snapshots VALUES (?,?,?,?)",
        (now_utc(), account, float(equity), cash),
    )
    conn.commit()


def trades_df(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM trades ORDER BY ts", conn)
