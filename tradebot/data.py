"""Market data ingestion.

* Daily history: Yahoo Finance via yfinance (free, split/dividend adjusted).
* Real-time quotes: Alpaca market data API. The free plan serves stocks from
  the IEX feed (one exchange, ~2-3% of volume, so the bid/ask can be wider
  than the national best). Crypto quotes are free and need no keys.

Every network call goes through tradebot.retry (backoff on 429s and
transient errors).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from tradebot.broker import disable_builtin_retry
from tradebot.retry import call_with_retry

PRICE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]

# Pause between per-symbol Yahoo requests so a watchlist refresh doesn't
# trip Yahoo's (undocumented) rate limiter in the first place.
YF_REQUEST_SPACING_S = 0.5


def _yf_history(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so tests don't need network libs

    # yfinance swallows errors (incl. rate limits) and returns an empty frame
    # by default; we want them raised so the retry layer can see them.
    yf.config.debug.hide_exceptions = False
    return yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=True)


def fetch_prices(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily, split/dividend-adjusted OHLCV for one symbol.

    Returns long-format DataFrame: symbol, date, open, high, low, close, volume.
    Empty DataFrame means Yahoo has no data for that range (e.g. bad ticker).
    """
    hist = call_with_retry(_yf_history, symbol, start, end, what=f"yfinance {symbol}")
    if hist is None or hist.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    hist = hist.reset_index()
    date_col = "Date" if "Date" in hist.columns else hist.columns[0]
    dates = pd.to_datetime(hist.pop(date_col)).dt.strftime("%Y-%m-%d")
    out = hist.rename(columns=str.lower)[PRICE_COLUMNS[2:]]
    out.insert(0, "date", dates)
    out.insert(0, "symbol", symbol)
    return out


def fetch_many(symbols: list[str], start: str | dict[str, str], end: str | None = None,
               spacing_s: float = YF_REQUEST_SPACING_S):
    """Yield (symbol, DataFrame | Exception) for each symbol, paced politely.

    `start` is one date for all symbols, or a {symbol: date} dict.
    One bad symbol doesn't abort the batch; its exception is yielded instead.
    """
    for i, sym in enumerate(symbols):
        if i and spacing_s:
            time.sleep(spacing_s)
        try:
            sym_start = start[sym] if isinstance(start, dict) else start
            yield sym, fetch_prices(sym, sym_start, end)
        except Exception as exc:  # noqa: BLE001 - reported to caller
            yield sym, exc


# ---------------------------------------------------------------- incremental

INCREMENTAL_OVERLAP_DAYS = 10


def incremental_start(last_date: str | None, full_start: str,
                      overlap_days: int = INCREMENTAL_OVERLAP_DAYS) -> str:
    """Where to resume downloading: a few days before the last stored date
    (the overlap lets us detect re-adjusted history), or full_start if empty."""
    if not last_date:
        return full_start
    resume = (pd.Timestamp(last_date) - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")
    return max(full_start, resume)


def history_was_readjusted(stored: pd.Series, fresh: pd.DataFrame, rel_tol: float = 1e-4) -> bool:
    """True if overlapping closes differ, meaning Yahoo re-adjusted the whole
    history (a split or dividend) and the stored series must be refetched.

    `stored` is a date-indexed close series; its last bar is ignored because it
    may have been an in-progress intraday bar when it was saved. No overlap to
    compare also counts as True, since consistency can't be verified.
    """
    stored = stored.iloc[:-1]
    if fresh.empty or stored.empty:
        return not fresh.empty
    new = fresh.set_index(pd.to_datetime(fresh["date"]))["close"].astype(float)
    common = stored.index.intersection(new.index)
    if len(common) == 0:
        return True
    return bool(((new[common] / stored[common]) - 1).abs().gt(rel_tol).any())


# ---------------------------------------------------------------- real-time

@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp: datetime
    source: str

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.bid or self.ask

    @property
    def spread_bps(self) -> float:
        return (self.ask - self.bid) / self.mid * 1e4 if self.mid else float("nan")


def to_alpaca_symbol(symbol: str) -> str:
    """Yahoo 'BTC-USD' -> Alpaca 'BTC/USD'; stocks unchanged."""
    return symbol.upper().replace("-", "/") if is_crypto(symbol) else symbol.upper()


class AlpacaQuotes:
    """Latest bid/ask from Alpaca's free market-data endpoints."""

    def __init__(self, key: str | None, secret: str | None, stock_feed: str = "iex"):
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

        self._stock = StockHistoricalDataClient(key, secret) if key and secret else None
        self._crypto = CryptoHistoricalDataClient(key, secret) if key and secret else CryptoHistoricalDataClient()
        for c in (self._stock, self._crypto):
            if c is not None:
                disable_builtin_retry(c)
        self.stock_feed = stock_feed

    def latest(self, symbols: list[str]) -> dict[str, Quote]:
        """Yahoo-style symbols in, {yahoo_symbol: Quote} out."""
        stocks = [s for s in symbols if not is_crypto(s)]
        crypto = [s for s in symbols if is_crypto(s)]
        out: dict[str, Quote] = {}
        if stocks:
            out.update(self._stock_quotes(stocks))
        if crypto:
            out.update(self._crypto_quotes(crypto))
        return out

    def _stock_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        if self._stock is None:
            raise RuntimeError("Stock quotes need ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestQuoteRequest

        req = StockLatestQuoteRequest(symbol_or_symbols=[s.upper() for s in symbols],
                                      feed=DataFeed(self.stock_feed))
        raw = call_with_retry(self._stock.get_stock_latest_quote, req, what="alpaca stock quotes")
        return {s: _to_quote(s, raw[s.upper()], f"alpaca:{self.stock_feed}")
                for s in symbols if s.upper() in raw}

    def _crypto_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        from alpaca.data.requests import CryptoLatestQuoteRequest

        mapping = {to_alpaca_symbol(s): s for s in symbols}
        req = CryptoLatestQuoteRequest(symbol_or_symbols=list(mapping))
        raw = call_with_retry(self._crypto.get_crypto_latest_quote, req, what="alpaca crypto quotes")
        return {mapping[a]: _to_quote(mapping[a], q, "alpaca:crypto") for a, q in raw.items() if a in mapping}


def _to_quote(symbol: str, q, source: str) -> Quote:
    return Quote(symbol=symbol, bid=float(q.bid_price or 0), ask=float(q.ask_price or 0),
                 bid_size=float(q.bid_size or 0), ask_size=float(q.ask_size or 0),
                 timestamp=q.timestamp, source=source)


# ---------------------------------------------------------------- helpers

def is_crypto(symbol: str) -> bool:
    return symbol.upper().endswith("-USD")


def periods_per_year(symbol: str) -> int:
    """Crypto trades every day; stocks ~252 days/year."""
    return 365 if is_crypto(symbol) else 252
