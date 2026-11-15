"""Write down the S&P 500 members as they were listed on a past date.

    python scripts/build_universe.py 2016-01-01

Why a past date: a test that picks stocks from today's index only sees the
companies that survived and grew. The honest list is the one a person could
have written down before the test period began, including companies that
were later bought, dropped from the index or went bankrupt.

How: Wikipedia keeps every old version of its "List of S&P 500 companies"
page. This reads the last version saved before the date and takes the
tickers from its table. They are the tickers of that day (FB, not META);
Alpaca's `asof` option follows each one to the company's later names.

The result goes to universes/sp500_<year>.txt, one ticker per line, in
Alpaca's format (BRK.B, not BRK-B).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

TITLE = "List of S&P 500 companies"
API = "https://en.wikipedia.org/w/api.php"
PAGE = "https://en.wikipedia.org/w/index.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (universe-builder)"}
ROOT = Path(__file__).resolve().parent.parent


def clean(ticker) -> str | None:
    if not isinstance(ticker, str) or not ticker.strip():
        return None
    return ticker.strip().upper().replace("-", ".")


def tickers_from(html: str) -> list[str]:
    """The tickers in the page's member table (the first table with a ticker column)."""
    for table in pd.read_html(io.StringIO(html)):
        column = next((c for c in table.columns if str(c).lower() in ("symbol", "ticker symbol", "ticker")), None)
        if column is not None and len(table) > 400:
            return sorted({t for t in map(clean, table[column]) if t})
    raise ValueError("No member table found on the page.")


def last_revision_before(date: str) -> dict:
    reply = requests.get(API, headers=HEADERS, timeout=30, params={
        "action": "query", "prop": "revisions", "titles": TITLE, "rvlimit": 1, "rvdir": "older",
        "rvstart": f"{date}T00:00:00Z", "rvprop": "ids|timestamp", "format": "json"})
    reply.raise_for_status()
    return next(iter(reply.json()["query"]["pages"].values()))["revisions"][0]


def main(as_of: str) -> None:
    revision = last_revision_before(as_of)
    reply = requests.get(PAGE, headers=HEADERS, timeout=30, params={"title": TITLE, "oldid": revision["revid"]})
    reply.raise_for_status()
    members = tickers_from(reply.text)
    target = ROOT / "universes" / f"sp500_{as_of[:4]}.txt"
    target.parent.mkdir(exist_ok=True)
    header = [f"# S&P 500 members as listed on {revision['timestamp'][:10]}: {len(members)} tickers.",
              "# These are the tickers of that day, in Alpaca's format (FB, not META; BRK.B, not BRK-B).",
              f"# Source: Wikipedia, \"{TITLE}\", revision {revision['revid']} (the last one before {as_of}).",
              f"# Rebuild with: python scripts/build_universe.py {as_of}"]
    target.write_text("\n".join(header + members) + "\n", encoding="utf-8")
    print(f"{len(members)} members as listed on {revision['timestamp'][:10]}. Written to {target}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2016-01-01")
