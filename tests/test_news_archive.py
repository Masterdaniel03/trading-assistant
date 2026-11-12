"""The historical news download, with a fake HTTP session. No network."""
import numpy as np
import pandas as pd
import pytest

from tradebot import data, news_archive


class Response:
    def __init__(self, body=None, status=200):
        self.body, self.status_code, self.headers = body or {}, status, {}

    def raise_for_status(self):
        if self.status_code != 200:
            error = RuntimeError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self):
        return self.body


def row(id, when):
    return {"id": id, "created_at": when, "updated_at": when, "headline": f"Headline {id}",
            "summary": "", "source": "benzinga", "symbols": ["AAPL"]}


class Session:
    """Serves two pages for January, one for February, one for March."""
    def __init__(self, fail_first=False):
        self.calls, self.fail_first = [], fail_first

    def get(self, url, headers, params, timeout):
        self.calls.append(dict(params))
        if self.fail_first and len(self.calls) == 1:
            return Response(status=429)
        month, token = params["start"][:7], params.get("page_token")
        if month == "2026-01" and token is None:
            return Response({"news": [row(1, "2026-01-05T14:00:00Z")], "next_page_token": "p2"})
        if month == "2026-01":
            return Response({"news": [row(2, "2026-01-20T14:00:00Z")], "next_page_token": None})
        return Response({"news": [row(int(month[-2:]) * 10, f"{month}-03T14:00:00Z")]})


NOW = pd.Timestamp("2026-03-10 12:00")


@pytest.fixture
def conn(tmp_path):
    return news_archive.connect(tmp_path / "archive.db")


def fetch(conn, session):
    return news_archive.download(conn, "k", "s", ["MSFT", "AAPL"], "2026-01-01", say=lambda _: None,
                                 session=session, sleep=lambda _: None, now=NOW)


def test_download_follows_pages_and_stores_every_article(conn):
    session = Session()
    assert fetch(conn, session) == 4
    assert [c.get("page_token") for c in session.calls] == [None, "p2", None, None]
    assert session.calls[0]["symbols"] == "AAPL,MSFT" and session.calls[0]["sort"] == "asc"
    assert session.calls[0]["start"] == "2026-01-01T00:00:00Z"
    assert session.calls[0]["end"] == "2026-01-31T23:59:59Z"
    assert session.calls[-1]["end"] == "2026-03-10T11:59:59Z"      # never asks beyond "now"

    stored = news_archive.load_articles(conn).sort_values("id")
    assert stored["id"].tolist() == [1, 2, 20, 30]
    assert str(stored["updated_at"].dt.tz) == "UTC" and stored["symbols"].iloc[0] == ["AAPL"]


def test_second_run_skips_finished_months_but_refreshes_the_current_one(conn):
    fetch(conn, Session())
    again = Session()
    assert fetch(conn, again) == 1
    assert [c["start"][:7] for c in again.calls] == ["2026-03"]    # March was not over yet
    assert len(news_archive.load_articles(conn)) == 4              # same article, not duplicated


def test_a_different_ticker_list_is_downloaded_from_scratch(conn):
    fetch(conn, Session())
    other = Session()
    news_archive.download(conn, "k", "s", ["IBM"], "2026-01-01", say=lambda _: None,
                          session=other, sleep=lambda _: None, now=NOW)
    assert [c["start"][:7] for c in other.calls] == ["2026-01", "2026-01", "2026-02", "2026-03"]


def test_rate_limit_is_retried(conn):
    session = Session(fail_first=True)
    assert fetch(conn, session) == 4
    assert len(session.calls) == 5


def test_month_starts():
    months = news_archive.month_starts("2025-11-20", "2026-02-03")
    assert [m.strftime("%Y-%m") for m in months] == ["2025-11", "2025-12", "2026-01", "2026-02"]


# ---------------------------------------------------------------- price sources

def test_alpaca_bars_are_converted_to_new_york_dates():
    bars = pd.DataFrame({"timestamp": pd.to_datetime(["2025-08-26 04:00", "2025-08-27 04:00"], utc=True),
                         "open": [12.0, 11.9], "high": [12.1, 12.0], "low": [11.8, 11.9],
                         "close": [11.95, np.nan], "volume": [1000, 2000], "vwap": [12.0, 12.0]})
    out = data.bars_to_prices("WBA", bars)
    assert out.columns.tolist() == data.PRICE_COLUMNS
    assert out["date"].tolist() == ["2025-08-26"]                  # the row with no close is left out
    assert out["symbol"].tolist() == ["WBA"]
    assert data.bars_to_prices("WBA", bars.iloc[:0]).empty


def test_fetch_prices_reports_how_many_blank_rows_it_left_out(monkeypatch):
    idx = pd.DatetimeIndex(["2026-11-10", "2026-11-11"], name="Date")
    raw = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": [10.0, np.nan], "Volume": 5.0},
                       index=idx)
    monkeypatch.setattr(data, "_yf_history", lambda *a: raw)
    out = data.fetch_prices("SPY", "2026-11-01")
    assert out["date"].tolist() == ["2026-11-10"] and out.attrs["missing_close"] == 1
