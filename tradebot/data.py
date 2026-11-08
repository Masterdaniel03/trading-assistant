"""Market data ingestion (Yahoo Finance via yfinance)."""
from __future__ import annotations

import pandas as pd


def fetch_prices(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily, split/dividend-adjusted OHLCV for one symbol.

    Returns long-format DataFrame: symbol, date, open, high, low, close, volume.
    """
    import yfinance as yf  # imported lazily so tests don't need network libs

    hist = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=True)
    if hist.empty:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    hist = hist.reset_index()
    hist["date"] = pd.to_datetime(hist["Date"]).dt.strftime("%Y-%m-%d")
    out = hist.rename(columns=str.lower)[["date", "open", "high", "low", "close", "volume"]]
    out.insert(0, "symbol", symbol)
    return out


def is_crypto(symbol: str) -> bool:
    return symbol.upper().endswith("-USD")


def periods_per_year(symbol: str) -> int:
    """Crypto trades every day; stocks ~252 days/year."""
    return 365 if is_crypto(symbol) else 252
