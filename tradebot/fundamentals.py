"""Company fundamentals from SEC filings, each number dated by when it became public.

Source: SEC EDGAR's XBRL "company facts", one free JSON file per company. For
every number a company has put in a 10-K or 10-Q, it lists each filing that
reported it, with that filing's date. That makes a strict point-in-time rule
possible:

* When a number counts as known. A filing dated D can come out as late as
  10 pm New York time, after that day's close. So it is first used for the
  decision after the close of the next trading day: on day t, only filings
  dated before t count.
* Restatements. When a later filing reports a different value for a period,
  the old value stays in force until the later filing's date.
* Changing tags. Companies report the same line under different XBRL tags
  over the years (revenue moved to a new tag with the 2018 accounting change).
  The choice between tags is made among the tags filed by that day, never
  with a later filing in view.
* Stale numbers. A number for a period that ended more than `max_age_days`
  before the day counts as unknown, so a company that stopped filing does not
  carry old figures for years.
* Which company. Each 2015 ticker starts with the SEC number (CIK) listed for
  it at the time. When the company moved its filings to a new registrant (a
  holding-company reorganisation such as Google to Alphabet, a move abroad, or
  a merger the ticker's price series follows), the new CIK takes over from that
  date (universes/sp500_2016_cik_successors.csv, see read_links).

Market value comes from the public float, the dollar value of the shares held
by outside investors, which each 10-K reports as of the middle of the fiscal
year. It is carried forward by the stock's own return since that date:
float x close(t) / close(float date). Both closes are adjusted ones, so their
ratio is the stock's total return between the two days and needs no knowledge
of later splits. (Dividing a reported per-share number by today's
split-adjusted price would quietly use future splits.) The float leaves out
shares held by insiders, so for founder-controlled companies it is smaller
than the full market value.

Nine inputs come out of this (FEATURES), each a long-published measure, chosen
before the first test run. Download and refresh: `python main.py
fundamentals-fetch`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from tradebot import db
from tradebot.retry import call_with_retry

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
# Every company's total assets for one calendar quarter, with the filing each came from.
FRAME_URL = "https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/CY{year}Q{quarter}I.json"
# SEC asks automated clients to say who they are. Set SEC_USER_AGENT in .env
# to your name and e-mail address to follow its fair-access policy.
DEFAULT_AGENT = "trading-assistant research script"
FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A", "10-QT", "10-QT/A",
         "20-F", "20-F/A", "40-F", "40-F/A"}
# A float outside this multiple of the company's total assets is a scale error, not a value: the
# errors found were 1000x or more (Waters, Garmin, M&T, Duke, Mattel), while real values run from
# about 0.01 (a bank) to about 30 (Nvidia in 2024).
FLOAT_TO_ASSETS = (0.002, 500.0)


@dataclass(frozen=True)
class Item:
    kind: str                  # "duration" (over a period: income, cash flow) or "instant" (on a day: balance sheet)
    concepts: tuple[str, ...]  # XBRL tags, "taxonomy:name" (us-gaap when no taxonomy), most preferred first
    combine: str = "first"     # "first": the most preferred tag filed for the period; "max": the largest


ITEMS = {
    # Some revenue tags are parts of the total (contract revenue leaves out a bank's interest
    # or a landlord's rent), so the largest revenue figure filed for a period is the total.
    "revenue": Item("duration", ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                                 "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"),
                    "max"),
    "net_income": Item("duration", ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic")),
    "cfo": Item("duration", ("NetCashProvidedByUsedInOperatingActivities",
                             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")),
    "assets": Item("instant", ("Assets",)),
    "equity": Item("instant", ("StockholdersEquity",
                               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
    "public_float": Item("instant", ("dei:EntityPublicFloat",)),
}

# name -> (group, what it measures). The order here is the column order everywhere.
FEATURES = {
    "earnings_yield": ("value", "net income over the last four quarters, per dollar of market value"),
    "book_to_market": ("value", "shareholders' equity per dollar of market value (left out when equity is negative)"),
    "sales_to_price": ("value", "revenue over the last four quarters, per dollar of market value"),
    "roa": ("quality", "net income over the last four quarters, per dollar of assets"),
    "accruals": ("quality", "net income minus operating cash flow, per dollar of assets (profit not yet seen as cash)"),
    "equity_to_assets": ("quality", "shareholders' equity as a share of assets (low means more debt)"),
    "asset_growth": ("growth", "growth of total assets over a year"),
    "sales_growth": ("growth", "revenue over the last four quarters against the four before"),
    "earnings_change": ("growth", "change in net income over the last four quarters, per dollar of assets"),
}
NAMES = list(FEATURES)

FACT_COLUMNS = ["item", "concept", "start", "end", "value", "form", "filed", "accn"]
HISTORY_COLUMNS = ["item", "filed", "end", "value", "prev_end", "prev_value"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    cik        INTEGER PRIMARY KEY,
    name       TEXT,
    fetched    TEXT NOT NULL,        -- when its facts were last downloaded (UTC)
    facts      INTEGER NOT NULL,     -- rows kept; 0 when SEC has no XBRL data for it
    last_filed TEXT
);
-- Every reported value this module uses, once per filing that reported it.
CREATE TABLE IF NOT EXISTS facts (
    cik INTEGER NOT NULL, item TEXT NOT NULL, concept TEXT NOT NULL,
    start TEXT NOT NULL,             -- '' for balance-sheet numbers, which are on a day
    end TEXT NOT NULL, value REAL NOT NULL, form TEXT, filed TEXT NOT NULL, accn TEXT NOT NULL,
    PRIMARY KEY (cik, concept, start, end, accn)
);
-- What was known about each item right after each filing day (derived from facts).
CREATE TABLE IF NOT EXISTS history (
    cik INTEGER NOT NULL, item TEXT NOT NULL, filed TEXT NOT NULL,
    end TEXT NOT NULL, value REAL NOT NULL,     -- the latest period, and its value (four-quarter total for flows)
    prev_end TEXT, prev_value REAL,             -- the same, a year earlier
    PRIMARY KEY (cik, item, filed)
);
-- Filings already acted on when checking for new ones (see new_filings).
CREATE TABLE IF NOT EXISTS frame_seen (cik INTEGER NOT NULL, accn TEXT NOT NULL, PRIMARY KEY (cik, accn));
"""


def connect(path: str) -> "db.sqlite3.Connection":
    conn = db.connect(path)
    conn.executescript(SCHEMA)
    return conn


Link = tuple[int, str | None, str | None]     # (cik, filed from, filed before); None = open-ended


def read_ciks(path: str | Path) -> dict[str, int]:
    """symbol -> SEC company number, from universes/<name>_ciks.csv."""
    table = pd.read_csv(path, comment="#", dtype={"symbol": str})
    return {row.symbol.strip().upper(): int(row.cik) for row in table.itertuples()}


def read_links(ciks_path: str | Path, successors_path: str | Path | None = None) -> dict[str, list[Link]]:
    """symbol -> the SEC numbers whose filings speak for it, each with the filing dates it covers.

    The 2015 CIK covers everything before the first successor's `from` date;
    each successor covers its own dates up to the next one's."""
    links = {sym: [(cik, None, None)] for sym, cik in read_ciks(ciks_path).items()}
    if successors_path and Path(successors_path).exists():
        table = pd.read_csv(successors_path, comment="#", dtype={"symbol": str, "from": str})
        for sym, rows in table.groupby(table["symbol"].str.strip().str.upper(), sort=False):
            if sym not in links:
                continue
            chain = [(links[sym][0][0], None)] + [(int(r.cik), r[3]) for r in rows.sort_values("from").itertuples()]
            links[sym] = [(cik, start, chain[i + 1][1] if i + 1 < len(chain) else None)
                          for i, (cik, start) in enumerate(chain)]
    return links


def _as_links(symbol_ciks: dict) -> dict[str, list[Link]]:
    return {sym: value if isinstance(value, list) else [(int(value), None, None)]
            for sym, value in symbol_ciks.items()}


def all_ciks(links: dict[str, list[Link]]) -> list[int]:
    return sorted({cik for chain in links.values() for cik, _, _ in chain})


# ---------------------------------------------------------------- reading SEC's files

def parse_facts(doc: dict | None) -> pd.DataFrame:
    """The numbers this module uses from one company-facts file: one row per filing that reported each."""
    facts = (doc or {}).get("facts") or {}
    rows = []
    for item, spec in ITEMS.items():
        for concept in spec.concepts:
            taxonomy, name = concept.split(":") if ":" in concept else ("us-gaap", concept)
            units = ((facts.get(taxonomy) or {}).get(name) or {}).get("units") or {}
            for v in units.get("USD", []):
                start = v.get("start") or ""
                if v.get("form") not in FORMS or not v.get("filed") or not v.get("end") or v.get("val") is None:
                    continue
                if (spec.kind == "duration") != bool(start):
                    continue
                rows.append((item, concept, start, v["end"], float(v["val"]), v["form"], v["filed"],
                             v.get("accn") or ""))
    return pd.DataFrame(rows, columns=FACT_COLUMNS)


def recent_quarters(today: date, count: int = 3) -> list[tuple[int, int]]:
    """The `count` most recent calendar quarters that have ended before `today`, newest first."""
    year, quarter = today.year, (today.month - 1) // 3 + 1
    out = []
    for _ in range(count):
        year, quarter = (year, quarter - 1) if quarter > 1 else (year - 1, 4)
        out.append((year, quarter))
    return out


# ---------------------------------------------------------------- point in time

DURATIONS = {3: (80, 100), 6: (170, 195), 9: (260, 290), 12: (350, 380)}   # months -> length in days
YEAR = timedelta(days=365)


def months(start: date | None, end: date) -> int | None:
    """3, 6, 9 or 12 for a period of that many months (52/53-week years allowed); None otherwise."""
    if start is None:
        return None
    days = (end - start).days
    return next((m for m, (low, high) in DURATIONS.items() if low <= days <= high), None)


def _day(text: str) -> date | None:
    return date.fromisoformat(text) if text else None


class Known:
    """Everything filed so far about one item of one company: period -> {tag: latest value}.

    Fed one filing day at a time, oldest first, so it never holds a value that
    was not yet public."""

    def __init__(self, spec: Item):
        self.spec = spec
        self.periods: dict[tuple[date | None, date], dict[str, float]] = {}
        self.by_end: dict[date, list[tuple[date | None, date]]] = {}

    def add(self, concept: str, start: date | None, end: date, value: float) -> None:
        key = (start, end)
        if key not in self.periods:
            if self.spec.kind == "duration" and months(start, end) is None:
                return                                   # odd period lengths (one month, 15 weeks) are not used
            self.periods[key] = {}
            self.by_end.setdefault(end, []).append(key)
        self.periods[key][concept] = value               # a later filing replaces the earlier value

    def value(self, key) -> float | None:
        found = self.periods.get(key)
        if not found:
            return None
        if self.spec.combine == "max":
            return max(found.values())
        return next(found[c] for c in self.spec.concepts if c in found)

    def ends_near(self, day: date, tolerance: int) -> list[date]:
        """Period ends within `tolerance` days of `day`, closest first."""
        near = (day + timedelta(days=d) for d in sorted(range(-tolerance, tolerance + 1), key=abs))
        return [end for end in near if end in self.by_end]

    def find(self, end: date, tolerance: int, length: int, start: date | None = None) -> tuple | None:
        """A period of `length` months ending near `end` (and, if given, starting near `start`)."""
        for e in self.ends_near(end, tolerance):
            for key in self.by_end[e]:
                if months(*key) == length and (start is None or abs((key[0] - start).days) <= 10):
                    return key
        return None

    def ttm(self, end: date) -> float | None:
        """Total over the twelve months to `end`: a full-year figure, or else the year so far
        plus the last full year minus the same part of that year."""
        keys = sorted(self.by_end.get(end, ()), key=lambda k: k[0])          # longest period first
        for key in keys:
            if months(*key) == 12:
                return self.value(key)
        for start, _ in keys:
            length = months(start, end)
            if length not in (3, 6, 9):
                continue
            last_year = self.find(start - timedelta(days=1), 7, 12)
            same_part = self.find(end - YEAR, 10, length, start - YEAR)
            if last_year and same_part:
                return self.value((start, end)) + self.value(last_year) - self.value(same_part)
        return None

    def at(self, end: date) -> float | None:
        return self.ttm(end) if self.spec.kind == "duration" else self.value((None, end))

    def latest(self) -> tuple | None:
        """(end, value, prev_end, prev_value) for the newest period that can be valued, with the
        same a year earlier (prev_* are None when that is not known)."""
        for end in sorted(self.by_end, reverse=True)[:4]:
            value = self.at(end)
            if value is None:
                continue
            for prev_end in self.ends_near(end - YEAR, 20):
                prev = self.at(prev_end)
                if prev is not None:
                    return end, value, prev_end, prev
            return end, value, None, None
        return None


def item_history(rows: pd.DataFrame, spec: Item) -> list[tuple]:
    """What was known about one item after each filing day: (filed, end, value, prev_end, prev_value).

    rows: one company's facts for the item. A filing day where nothing can be
    valued adds no row, so the last known value stays in force."""
    known = Known(spec)
    out = []
    for filed, day in rows.sort_values(["filed", "accn"]).groupby("filed", sort=True):
        for r in day.itertuples():
            known.add(r.concept, _day(r.start), _day(r.end), r.value)
        snap = known.latest()
        if snap is not None:
            end, value, prev_end, prev = snap
            out.append((filed, end.isoformat(), value, prev_end.isoformat() if prev_end else None, prev))
    return out


def company_history(facts: pd.DataFrame) -> pd.DataFrame:
    rows = [(item, *row) for item, spec in ITEMS.items()
            for row in item_history(facts[facts["item"] == item], spec)]
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


# ---------------------------------------------------------------- storage

def store_company(conn, cik: int, name: str | None, facts: pd.DataFrame, fetched: str) -> None:
    """Replace one company's facts and derived history."""
    conn.execute("DELETE FROM facts WHERE cik = ?", (cik,))
    conn.execute("DELETE FROM history WHERE cik = ?", (cik,))
    conn.executemany("INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?)",
                     [(cik, *row) for row in facts[FACT_COLUMNS].itertuples(index=False)])
    hist = company_history(facts)
    conn.executemany("INSERT OR REPLACE INTO history VALUES (?,?,?,?,?,?,?)",
                     [(cik, *row) for row in hist[HISTORY_COLUMNS].itertuples(index=False)])
    last = facts["filed"].max() if len(facts) else None
    conn.execute("INSERT OR REPLACE INTO companies VALUES (?,?,?,?,?)", (cik, name, fetched, len(facts), last))
    conn.commit()


def load_history(conn) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT cik, {', '.join(HISTORY_COLUMNS)} FROM history", conn)


# ---------------------------------------------------------------- downloading

class SecError(Exception):
    def __init__(self, status_code: int, url: str):
        super().__init__(f"SEC answered {status_code} for {url}")
        self.status_code = status_code


class Sec:
    """SEC EDGAR, politely: a declared User-Agent and at most 8 requests a second (SEC allows 10)."""

    def __init__(self, user_agent: str | None = None, spacing_s: float = 0.125, session=None):
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.session.headers.update({"User-Agent": user_agent or DEFAULT_AGENT,
                                     "Accept-Encoding": "gzip, deflate"})
        self.spacing_s = spacing_s
        self._last = 0.0

    def _get(self, url: str):
        wait = self._last + self.spacing_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        reply = self.session.get(url, timeout=60)
        if reply.status_code == 404:
            return None
        if reply.status_code != 200:
            raise SecError(reply.status_code, url)
        return reply

    def get(self, url: str):
        return call_with_retry(self._get, url, what=f"SEC {url.rsplit('/', 1)[-1]}")

    def company_facts(self, cik: int) -> dict | None:
        """The company's facts, or None if SEC has no XBRL data for that number."""
        reply = self.get(FACTS_URL.format(cik=cik))
        return None if reply is None else reply.json()

    def assets_frame(self, year: int, quarter: int) -> list[tuple[int, str]] | None:
        """(cik, filing accession number) of every company's total assets at the end of a calendar quarter."""
        reply = self.get(FRAME_URL.format(year=year, quarter=quarter))
        return None if reply is None else [(int(r["cik"]), r["accn"]) for r in reply.json().get("data", [])]


def new_filings(conn, sec, today: date) -> set[tuple[int, str]]:
    """(cik, accession) of filings that report total assets for one of the last three quarters and
    have not been seen yet.

    Every 10-K and 10-Q reports total assets, so a new one shows up here. Some
    other filings carry the number too (a bank's prospectus supplements); once
    a company has been downloaded after one, it is remembered and not
    downloaded again for it. The lookup needs only data.sec.gov, which (unlike
    EDGAR's filing index) works without an e-mail address in the User-Agent."""
    known = set(conn.execute("SELECT DISTINCT cik, accn FROM facts WHERE item = 'assets'"))
    known |= set(conn.execute("SELECT cik, accn FROM frame_seen"))
    found = set()
    for year, quarter in recent_quarters(today):
        found |= {pair for pair in (sec.assets_frame(year, quarter) or []) if pair not in known}
    return found


def update(conn, ciks: list[int], sec, say: Callable[[str], None] = print, full: bool = False,
           today: date | None = None) -> dict:
    """Bring the stored filings up to date.

    The first run (or full=True) downloads every company. Later runs download
    again only the companies with a 10-K or 10-Q that is not stored yet (see
    new_filings), plus any never downloaded."""
    today = today or pd.Timestamp.now(tz="America/New_York").date()
    wanted = sorted(set(ciks))
    have = {row[0] for row in conn.execute("SELECT cik FROM companies")}
    seen: set[tuple[int, str]] = set()
    if full or not have:
        todo = wanted
        say(f"Downloading the filings of all {len(todo)} companies from SEC EDGAR.")
    else:
        seen = {(cik, accn) for cik, accn in new_filings(conn, sec, today) if cik in set(wanted)}
        filed = {cik for cik, _ in seen}
        new = set(wanted) - have
        todo = sorted(filed | new)
        say(f"{len(filed)} of our companies have a 10-K or 10-Q not stored yet"
            + (f"; {len(new)} were never downloaded" if new else "") + ".")
    failed, empty = [], []
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for n, cik in enumerate(todo, 1):
        try:
            doc = sec.company_facts(cik)
        except Exception as exc:  # noqa: BLE001 - reported; the next run tries again
            failed.append(cik)
            say(f"  CIK {cik}: {type(exc).__name__}: {exc}")
            continue
        facts = parse_facts(doc)
        if facts.empty:
            empty.append(cik)
        store_company(conn, cik, (doc or {}).get("entityName"), facts, fetched)
        if n % 100 == 0:
            say(f"  {n} of {len(todo)} companies")
    conn.executemany("INSERT OR IGNORE INTO frame_seen VALUES (?, ?)",
                     [pair for pair in seen if pair[0] not in failed])
    conn.commit()
    newest = conn.execute("SELECT MAX(last_filed) FROM companies").fetchone()[0]
    say(f"Company filings: {len(todo) - len(failed)} downloaded, {len(failed)} failed. Newest filing: {newest}.")
    if empty:
        say(f"SEC has none of the needed numbers for {len(empty)}: CIK {', '.join(map(str, empty))}")
    return {"downloaded": len(todo) - len(failed), "failed": failed, "empty": empty}


# ---------------------------------------------------------------- daily inputs

def item_panels(history: pd.DataFrame, item: str, symbol_ciks: dict, days: pd.DatetimeIndex,
                symbols: list[str], max_age_days: int) -> dict[str, pd.DataFrame]:
    """What was known about `item` after each day's close: 'value', 'prev' and 'end' panels (days x symbols).

    symbol_ciks: symbol -> CIK, or symbol -> links (see read_links). A filing
    dated D counts from the first day after D. A value for a period that ended
    more than `max_age_days` before the day is left out."""
    part = history[history["item"] == item]
    by_cik = {cik: rows for cik, rows in part.groupby("cik")}
    links = _as_links(symbol_ciks)
    columns: dict[str, dict[str, pd.Series]] = {"value": {}, "prev": {}, "end": {}}
    day_series = pd.Series(days, index=days)
    for sym in symbols:
        pieces = []
        for cik, start, until in links.get(sym, []):
            rows = by_cik.get(cik)
            if rows is not None:
                keep = pd.Series(True, index=rows.index)
                if start:
                    keep &= rows["filed"] >= start
                if until:
                    keep &= rows["filed"] < until
                pieces.append(rows[keep])
        if not pieces:
            continue
        rows = pd.concat(pieces).sort_values("filed", kind="stable")    # same day: the later link wins
        if rows.empty:
            continue
        known_from = pd.DatetimeIndex(pd.to_datetime(rows["filed"]) + pd.Timedelta(days=1))
        frame = pd.DataFrame({"value": rows["value"].to_numpy(float),
                              "prev": pd.to_numeric(rows["prev_value"], errors="coerce").to_numpy(float),
                              "end": pd.to_datetime(rows["end"]).to_numpy()}, index=known_from)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        on_days = frame.reindex(days, method="ffill")
        fresh = (day_series - on_days["end"]).dt.days <= max_age_days
        for key in columns:
            columns[key][sym] = on_days[key].where(fresh)
    out = {key: pd.DataFrame(columns[key], index=days).reindex(columns=symbols) for key in ("value", "prev")}
    out["end"] = pd.DataFrame(columns["end"], index=days).reindex(columns=symbols).astype("datetime64[ns]")
    return out


def market_value(floats: dict[str, pd.DataFrame], close: pd.DataFrame, max_age_days: int,
                 held_back: list | None = None, assets: pd.DataFrame | None = None) -> pd.DataFrame:
    """The float carried forward by the stock's return: float x close(t) / close(float date).

    A float outside FLOAT_TO_ASSETS times the total assets known on the day it
    arrives (from the same 10-K) is held back as a scale error, and the last
    accepted float stays in use. Waters, for one, filed its float a thousand
    times too large in two 10-Ks. Each float is judged against its own filing,
    so a company that repeats the same mistake cannot confirm it."""
    days, out = close.index, {}
    day_number = days.to_numpy().astype("datetime64[D]").astype(np.int64)
    low, high = FLOAT_TO_ASSETS
    for sym in close.columns:
        prices = close[sym]
        if not prices.notna().any():
            continue
        ends = floats["end"][sym].to_numpy().astype("datetime64[D]")
        values, price_now = floats["value"][sym].to_numpy(float), prices.to_numpy(float)
        balance = (assets[sym].to_numpy(float) if assets is not None and sym in assets
                   else np.full(len(days), np.nan))
        result = np.full(len(days), np.nan)
        current, accepted = None, None                 # accepted: (float date as a day number, shares-equivalent)
        for i in range(len(days)):
            end, value = ends[i], values[i]
            if np.isnat(end) or not value > 0:
                continue
            if (end, value) != current:
                current = (end, value)
                if np.isfinite(balance[i]) and balance[i] > 0 and not low <= value / balance[i] <= high:
                    if held_back is not None:
                        held_back.append((sym, str(end), float(value)))
                else:
                    price = prices.asof(pd.Timestamp(end))
                    if np.isfinite(price) and price > 0:
                        accepted = (int(end.astype(np.int64)), value / price)
            if accepted is not None and day_number[i] - accepted[0] <= max_age_days:
                result[i] = accepted[1] * price_now[i]
        out[sym] = result
    return pd.DataFrame(out, index=days).reindex(columns=close.columns)


def compute(history: pd.DataFrame, symbol_ciks: dict, close: pd.DataFrame, max_age_days: int = 365,
            float_max_age_days: int = 730, notes: dict | None = None) -> dict[str, pd.DataFrame]:
    """The nine inputs, each a (days x symbols) table like close's, NaN where unknown.

    notes: if given, gets 'float_held_back': [(symbol, float date, value), ...]."""
    days, symbols = close.index, list(close.columns)
    p = {item: item_panels(history, item, symbol_ciks, days, symbols,
                           10 ** 6 if item == "public_float" else max_age_days)    # the float's age is checked below
         for item in ITEMS}
    held_back: list = []
    positive = lambda frame: frame.where(frame > 0)  # noqa: E731
    market = positive(market_value(p["public_float"], close, float_max_age_days, held_back, p["assets"]["value"]))
    if notes is not None:
        notes["float_held_back"] = held_back

    ni, ni_prev = p["net_income"]["value"], p["net_income"]["prev"]
    revenue, revenue_prev = p["revenue"]["value"], p["revenue"]["prev"]
    cfo = p["cfo"]["value"]
    assets, assets_prev = positive(p["assets"]["value"]), positive(p["assets"]["prev"])
    equity = p["equity"]["value"]
    out = {
        "earnings_yield": ni / market,
        "book_to_market": positive(equity) / market,
        "sales_to_price": revenue / market,
        "roa": ni / assets,
        "accruals": (ni - cfo) / assets,
        "equity_to_assets": equity / assets,
        "asset_growth": assets / assets_prev - 1,
        "sales_growth": revenue / positive(revenue_prev) - 1,
        "earnings_change": (ni - ni_prev) / assets,
    }
    traded = close.notna()
    return {name: out[name].replace([np.inf, -np.inf], np.nan).where(traded) for name in NAMES}
