"""SQLite storage: prices, signals, trades, equity snapshots.

Everything the system sees or does is recorded here, so results are
reproducible and can be analyzed later (and shown on a resume/dashboard).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
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
    notes      TEXT,
    status     TEXT                -- broker orders: accepted, filled, canceled, ...
);

CREATE TABLE IF NOT EXISTS stop_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    symbol  TEXT NOT NULL,
    date    TEXT NOT NULL,         -- latest price date when the stop fired
    pl_pct  REAL NOT NULL          -- return from entry, e.g. -0.08
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts      TEXT NOT NULL,
    account TEXT NOT NULL,
    equity  REAL NOT NULL,
    cash    REAL,
    PRIMARY KEY (ts, account)
);

CREATE TABLE IF NOT EXISTS news (
    id         TEXT NOT NULL,      -- the provider's article id
    symbol     TEXT NOT NULL,
    created_at TEXT NOT NULL,      -- UTC
    headline   TEXT NOT NULL,
    summary    TEXT,
    source     TEXT,
    url        TEXT,
    PRIMARY KEY (id, symbol)
);

-- One row per research attempt, kept or not, so every LLM answer can be audited.
CREATE TABLE IF NOT EXISTS research (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    as_of         TEXT NOT NULL,   -- latest price date the model was shown
    model         TEXT NOT NULL,
    cache_key     TEXT NOT NULL,   -- same inputs => same key, so a repeat run reuses the answer
    status        TEXT NOT NULL,   -- ok | no_news | rejected | error
    detail        TEXT,            -- why it was rejected or failed
    stance        TEXT,            -- bullish | neutral | bearish
    confidence    REAL,
    target        REAL,
    summary       TEXT,
    reasoning     TEXT,
    risks         TEXT,            -- JSON list
    sources       TEXT,            -- JSON list of the cited headlines
    news_ids      TEXT,            -- JSON list of every article shown to the model
    input_tokens  INTEGER,
    output_tokens INTEGER
);
"""


def now_utc(timespec: str = "seconds") -> str:
    return datetime.now(timezone.utc).isoformat(timespec=timespec)


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # Databases made before a column existed get it added here.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    if "status" not in columns:
        conn.execute("ALTER TABLE trades ADD COLUMN status TEXT")
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
        "SELECT date, close FROM prices WHERE symbol = ? AND close IS NOT NULL ORDER BY date",
        conn, params=(symbol,), parse_dates=["date"],
    )
    return df.set_index("date")["close"].rename(symbol)


def load_ohlcv(conn: sqlite3.Connection, symbols: list[str]) -> pd.DataFrame:
    """Long table of daily bars for several symbols: symbol, date, open, close, volume."""
    marks = ",".join("?" * len(symbols))
    return pd.read_sql_query(
        f"""SELECT symbol, date, open, close, volume FROM prices
            WHERE symbol IN ({marks}) AND close IS NOT NULL AND open IS NOT NULL
            ORDER BY date, symbol""", conn, params=list(symbols), parse_dates=["date"])


def last_price_date(conn: sqlite3.Connection, symbol: str) -> str | None:
    row = conn.execute("SELECT MAX(date) FROM prices WHERE symbol = ? AND close IS NOT NULL",
                       (symbol,)).fetchone()
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
              ts=None, status=None) -> int:
    cur = conn.execute(
        """INSERT INTO trades (ts, account, symbol, side, qty, notional, price, fees,
                               strategy, followed_ai, order_id, notes, status)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (ts or now_utc(), account, symbol.upper(), side, qty, notional, price, fees,
         strategy, followed_ai, order_id, notes, status),
    )
    conn.commit()
    return cur.lastrowid


SETTLED = ("filled", "canceled", "expired", "rejected", "replaced")


def open_trades(conn, account) -> list[tuple]:
    """Broker orders whose outcome isn't recorded yet: (id, ts, symbol, side, order_id)."""
    marks = ",".join("?" * len(SETTLED))
    return conn.execute(
        f"""SELECT id, ts, symbol, side, order_id FROM trades
            WHERE account = ? AND order_id IS NOT NULL AND order_id != ''
              AND (status IS NULL OR status NOT IN ({marks}))
            ORDER BY id""", (account, *SETTLED)).fetchall()


def update_trade(conn, trade_id, status, filled_qty=None, filled_price=None) -> None:
    """Record an order's outcome. A fill replaces the planned dollar amount
    with what was actually traded."""
    if filled_qty and filled_price:
        conn.execute("UPDATE trades SET status = ?, qty = ?, price = ?, notional = ? WHERE id = ?",
                     (status, filled_qty, filled_price, filled_qty * filled_price, trade_id))
    else:
        conn.execute("UPDATE trades SET status = ? WHERE id = ?", (status, trade_id))
    conn.commit()


def log_stop(conn, symbol, date, pl_pct) -> None:
    conn.execute("INSERT INTO stop_events (ts, symbol, date, pl_pct) VALUES (?,?,?,?)",
                 (now_utc(), symbol, date, float(pl_pct)))
    conn.commit()


def last_stop_date(conn, symbol) -> str | None:
    row = conn.execute("SELECT MAX(date) FROM stop_events WHERE symbol = ?", (symbol,)).fetchone()
    return row[0] if row else None


def log_equity(conn, account, equity, cash=None, ts=None) -> None:
    """Add one snapshot. The time is kept to the microsecond and bumped if it
    is already taken, so two snapshots can never overwrite each other."""
    ts = ts or now_utc("microseconds")
    while conn.execute("SELECT 1 FROM equity_snapshots WHERE ts = ? AND account = ?",
                       (ts, account)).fetchone():
        ts = (datetime.fromisoformat(ts) + timedelta(microseconds=1)).isoformat(timespec="microseconds")
    conn.execute("INSERT INTO equity_snapshots VALUES (?,?,?,?)", (ts, account, float(equity), cash))
    conn.commit()


def trades_df(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM trades ORDER BY ts", conn)


# ---------- news / research ----------
def save_news(conn, items) -> None:
    """items: objects with id, symbol, created_at, headline, summary, source, url."""
    conn.executemany(
        "INSERT OR REPLACE INTO news VALUES (?,?,?,?,?,?,?)",
        [(i.id, i.symbol, i.created_at.isoformat(timespec="seconds"), i.headline, i.summary,
          i.source, i.url) for i in items],
    )
    conn.commit()


RESEARCH_FIELDS = ("symbol", "as_of", "model", "cache_key", "status", "detail", "stance",
                   "confidence", "target", "summary", "reasoning", "risks", "sources",
                   "news_ids", "input_tokens", "output_tokens")


def save_research(conn, **fields) -> int:
    unknown = set(fields) - set(RESEARCH_FIELDS)
    if unknown:
        raise ValueError(f"Unknown research fields: {sorted(unknown)}")
    names = ["ts", *fields]
    cur = conn.execute(
        f"INSERT INTO research ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
        (now_utc(), *fields.values()),
    )
    conn.commit()
    return cur.lastrowid


def find_research(conn, cache_key: str) -> dict | None:
    """The latest kept answer for exactly these inputs, if there is one."""
    cur = conn.execute(
        "SELECT * FROM research WHERE cache_key = ? AND status = 'ok' ORDER BY id DESC LIMIT 1",
        (cache_key,))
    row = cur.fetchone()
    return dict(zip([c[0] for c in cur.description], row)) if row else None


def research_df(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM research ORDER BY id", conn)
