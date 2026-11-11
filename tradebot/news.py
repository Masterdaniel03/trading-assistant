"""Recent news per symbol, from Alpaca's news feed (Benzinga articles).

It comes free with the Alpaca keys and every article has a timestamp, so we
always know what was public at the time a research note was written.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from tradebot.broker import disable_builtin_retry
from tradebot.data import is_crypto
from tradebot.retry import call_with_retry

# An article tagged with more tickers than this is usually a market round-up
# ("10 stocks moving today") rather than news about the one we asked for.
FOCUSED_MAX_TICKERS = 3


@dataclass(frozen=True)
class NewsItem:
    id: str
    symbol: str            # our symbol, e.g. AAPL or BTC-USD
    created_at: datetime   # UTC
    headline: str
    summary: str
    source: str
    url: str
    n_tickers: int         # how many tickers the article is tagged with


def to_news_symbol(symbol: str) -> str:
    """Our 'BTC-USD' is 'BTCUSD' in the news feed; stocks are unchanged."""
    return symbol.upper().replace("-", "") if is_crypto(symbol) else symbol.upper()


def clean_text(text: str | None, limit: int) -> str:
    """Plain one-line text: no HTML tags or entities, trimmed to `limit` characters."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def select_items(items: list[NewsItem], limit: int) -> list[NewsItem]:
    """The `limit` most useful items: no repeated headlines, articles focused
    on few tickers first, newest first within each group."""
    seen, unique = set(), []
    for item in sorted(items, key=lambda i: i.created_at, reverse=True):
        key = item.headline.casefold()
        if item.headline and key not in seen:
            seen.add(key)
            unique.append(item)
    unique.sort(key=lambda i: i.n_tickers > FOCUSED_MAX_TICKERS)   # stable: keeps newest first
    return unique[:limit]


class AlpacaNews:
    def __init__(self, key: str | None, secret: str | None):
        if not key or not secret:
            raise RuntimeError("News needs ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")
        from alpaca.data.historical.news import NewsClient

        self._client = NewsClient(key, secret)
        disable_builtin_retry(self._client)

    def recent(self, symbol: str, since: datetime, limit: int = 50) -> list[NewsItem]:
        """Articles tagged with `symbol` published since `since`, newest first."""
        from alpaca.data.requests import NewsRequest

        req = NewsRequest(symbols=to_news_symbol(symbol), start=since, limit=limit)
        result = call_with_retry(self._client.get_news, req, what=f"alpaca news {symbol}")
        return [to_item(symbol, raw) for raw in result.data.get("news", [])]


def to_item(symbol: str, raw) -> NewsItem:
    created = raw.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return NewsItem(id=str(raw.id), symbol=symbol, created_at=created.astimezone(timezone.utc),
                    headline=clean_text(raw.headline, 200), summary=clean_text(raw.summary, 400),
                    source=str(raw.source or ""), url=str(raw.url or ""),
                    n_tickers=len(raw.symbols or []))
