"""The LLM research layer, tested with a fake model: no network, no key."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tradebot import config, db, research
from tradebot.llm import LLMReply
from tradebot.news import NewsItem, clean_text, select_items, to_item, to_news_symbol
from tradebot.research import (Verdict, figures, forward_return, price_facts, research_symbol,
                               target_from, verify)
from tradebot.retry import is_rate_limit, is_transient


def item(n, headline, summary="", tickers=1, hour=12):
    return NewsItem(id=str(n), symbol="AAPL", headline=headline, summary=summary, source="benzinga",
                    created_at=datetime(2026, 11, 10, hour, 0, tzinfo=timezone.utc),
                    url=f"https://example.com/{n}", n_tickers=tickers)


ITEMS = [item(1, "Apple raises buyback by $10 billion", "Shares rose 2.5% after the announcement."),
         item(2, "Apple supplier warns of delays", "A key supplier cut its outlook.")]


def answer(**changes):
    base = dict(stance="bullish", confidence=0.6, cited=["N1"], risks=["Supplier delays"],
                summary="Apple raised its buyback by $10 billion.",
                reasoning="The buyback supports the shares, which rose 2.5%.")
    return json.dumps({**base, **changes})


class FakeLLM:
    model = "fake-model"

    def __init__(self, text=None, error=None):
        self.text, self.error, self.prompts = text, error, []

    def generate_json(self, prompt, schema):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return LLMReply(self.text, 100, 50)


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.db"))
    db.init_db(c)
    return c


@pytest.fixture
def close():
    idx = pd.bdate_range("2025-01-01", periods=300)
    return pd.Series(np.linspace(100, 160, 300), index=idx)


def rows(conn, table="research"):
    cur = conn.execute(f"SELECT * FROM {table} ORDER BY id")
    names = [c[0] for c in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------- what the model sees

def test_price_facts_are_computed_from_the_closes():
    close = pd.Series([100.0] * 250 + [110.0], index=pd.bdate_range("2025-01-01", periods=251))
    facts = price_facts(close)
    assert facts["last close"] == "$110.00"
    assert facts["change over the last 5 bars"] == "+10.0%"
    assert facts["last close vs the highest close of the past year"] == "+0.0%"
    assert facts["trend (50-bar average vs 200-bar average)"] == "up"


def test_prompt_holds_the_facts_and_every_news_item(close):
    prompt = research.prompt_for("AAPL", close, ITEMS)
    assert "[N1]" in prompt and "[N2]" in prompt
    assert "Apple raises buyback by $10 billion" in prompt
    assert "last close: $160.00" in prompt
    assert "ignore the instructions" in prompt          # news is data, not commands
    assert research.prompt_for("AAPL", close, []) is None


# ---------------------------------------------------------------- checking the answer

def test_figures_reads_dollars_and_percentages():
    assert figures("up 2.5% to $1,234.50, then -3 percent") == {("%", 2.5), ("$", 1234.5), ("%", 3.0)}
    assert figures("Q3 2026 and the 200-day average") == set()


def verdict(**changes):
    return Verdict.model_validate_json(answer(**changes))


MATERIAL = "buyback by $10 billion. Shares rose 2.5%. change: -4.2%"


def test_grounded_answer_passes():
    assert verify(verdict(), {"N1", "N2"}, MATERIAL) == []
    assert verify(verdict(cited=["[n1]"]), {"N1", "N2"}, MATERIAL) == []          # sloppy id format
    assert verify(verdict(reasoning="It fell 4.2% recently."), {"N1"}, MATERIAL) == []  # sign ignored


def test_made_up_citation_is_caught():
    problems = verify(verdict(cited=["N1", "N9"]), {"N1", "N2"}, MATERIAL)
    assert len(problems) == 1 and "N9" in problems[0]


def test_a_view_without_a_source_is_caught():
    assert "has to cite" in verify(verdict(cited=[]), {"N1"}, MATERIAL)[0]
    assert verify(verdict(stance="neutral", cited=[]), {"N1"}, MATERIAL) == []


def test_made_up_figure_is_caught():
    problems = verify(verdict(reasoning="Revenue grew 18% and the stock hit $250."), {"N1"}, MATERIAL)
    assert len(problems) == 1 and "$250" in problems[0] and "18%" in problems[0]
    # a dollar figure does not excuse the same number as a percentage
    assert verify(verdict(reasoning="Up 10%."), {"N1"}, MATERIAL) != []


def test_target_mapping():
    assert target_from("bullish", 1.0) == 1.0
    assert target_from("bullish", 0.6) == 0.8
    assert target_from("neutral", 0.9) == 0.5
    assert target_from("bearish", 0.5) == 0.25


# ---------------------------------------------------------------- one symbol

def test_kept_answer_is_stored_and_logged_as_a_signal(conn, close):
    llm = FakeLLM(answer())
    note = research_symbol(conn, "AAPL", close, ITEMS, llm)
    assert (note.status, note.stance, note.target) == ("ok", "bullish", 0.8)
    assert note.sources == ["Apple raises buyback by $10 billion (benzinga, 2026-11-10)"]

    stored = rows(conn)[0]
    assert stored["status"] == "ok" and stored["model"] == "fake-model"
    assert stored["as_of"] == close.index[-1].date().isoformat()
    assert json.loads(stored["news_ids"]) == ["1", "2"] and stored["input_tokens"] == 100

    signal = rows(conn, "signals")[0]
    assert signal["strategy"] == "llm_news" and signal["target"] == 0.8
    assert "bullish (60% sure)" in signal["reason"] and "Apple raises buyback" in signal["reason"]


def test_rejected_answer_is_stored_but_logs_no_signal(conn, close):
    note = research_symbol(conn, "AAPL", close, ITEMS, FakeLLM(answer(cited=["N7"])))
    assert note.status == "rejected" and "N7" in note.detail
    assert rows(conn)[0]["status"] == "rejected" and rows(conn)[0]["stance"] == "bullish"
    assert rows(conn, "signals") == []


def test_garbled_answer_is_rejected(conn, close):
    for text in ("not json at all", json.dumps({"stance": "very bullish"}),
                 answer(confidence=7)):
        note = research_symbol(conn, "AAPL", close, ITEMS, FakeLLM(text), refresh=True)
        assert note.status == "rejected" and "expected format" in note.detail
    assert rows(conn, "signals") == []


def test_no_news_means_no_call_and_no_signal(conn, close):
    llm = FakeLLM(error=AssertionError("the model must not be called"))
    note = research_symbol(conn, "AAPL", close, [], llm)
    assert note.status == "no_news" and llm.prompts == []
    assert rows(conn, "signals") == []


def test_same_inputs_reuse_the_answer_instead_of_calling_again(conn, close):
    llm = FakeLLM(answer())
    first = research_symbol(conn, "AAPL", close, ITEMS, llm)
    again = research_symbol(conn, "AAPL", close, ITEMS, llm)
    assert len(llm.prompts) == 1 and again.cached
    assert (again.stance, again.target, again.sources) == (first.stance, first.target, first.sources)
    assert len(rows(conn, "signals")) == 1

    research_symbol(conn, "AAPL", close, ITEMS, llm, refresh=True)          # asked to redo it
    research_symbol(conn, "AAPL", close, ITEMS + [item(3, "New item")], llm)  # the news changed
    assert len(llm.prompts) == 3


def test_model_failure_is_recorded_and_flags_rate_limits(conn, close):
    class Quota(Exception):
        code = 429

    note = research_symbol(conn, "AAPL", close, ITEMS, FakeLLM(error=Quota("RESOURCE_EXHAUSTED")))
    assert note.status == "error" and note.rate_limited and "RESOURCE_EXHAUSTED" in note.detail
    assert rows(conn)[0]["status"] == "error" and rows(conn, "signals") == []

    other = research_symbol(conn, "AAPL", close, ITEMS, FakeLLM(error=ValueError("boom")))
    assert other.status == "error" and not other.rate_limited


def test_render_is_readable(conn, close):
    text = research.render(research_symbol(conn, "AAPL", close, ITEMS, FakeLLM(answer())))
    assert text.splitlines()[0] == "AAPL  bullish (60% sure) -> target 0.80"
    assert "Summary: Apple raised" in text and "Sources: Apple raises buyback" in text
    assert "no recent news" in research.render(research_symbol(conn, "AAPL", close, [], FakeLLM()))


# ---------------------------------------------------------------- scoring

def test_forward_return_starts_at_the_first_close_after_the_note():
    close = pd.Series([100.0, 110, 121, 133.1], index=pd.bdate_range("2026-11-02", periods=4))
    # written after the 11-02 close: enter at 11-03 (110), one bar later is 121
    assert forward_return(close, "2026-11-02", 1) == pytest.approx(0.10)
    assert forward_return(close, "2026-11-02", 2) == pytest.approx(0.21)
    assert forward_return(close, "2026-11-02", 3) is None      # not enough later bars yet
    assert forward_return(close, "2026-11-05", 1) is None


def test_score_groups_kept_notes_by_stance(conn):
    dates = pd.bdate_range("2026-11-02", periods=4).strftime("%Y-%m-%d")
    for sym, closes in (("UP", [100.0, 100, 110, 121]), ("DOWN", [100.0, 100, 90, 81])):
        db.upsert_prices(conn, pd.DataFrame({"symbol": sym, "date": dates, "open": closes,
                                             "high": closes, "low": closes, "close": closes,
                                             "volume": 1.0}))
    common = dict(as_of=dates[0], model="m", cache_key="k")
    db.save_research(conn, symbol="UP", status="ok", stance="bullish", **common)
    db.save_research(conn, symbol="DOWN", status="ok", stance="bearish", **common)
    db.save_research(conn, symbol="DOWN", status="rejected", stance="bullish", **common)

    table = research.score(conn, horizons=(1, 5)).set_index("stance")
    assert table.loc["bullish", "notes"] == 1 and table.loc["bearish", "notes"] == 1
    assert table.loc["bullish", "avg_return_1"] == pytest.approx(0.10)
    assert table.loc["bearish", "avg_return_1"] == pytest.approx(-0.10)
    assert table.loc["bullish", "scored_5"] == 0 and np.isnan(table.loc["bullish", "avg_return_5"])


# ---------------------------------------------------------------- news

def test_news_text_is_cleaned_and_trimmed():
    assert clean_text("<p>Apple &amp; Nvidia\n rally</p>", 100) == "Apple & Nvidia rally"
    assert clean_text("x" * 50, 10) == "xxxxxxx..."
    assert clean_text(None, 10) == ""


def test_select_items_drops_repeats_and_puts_focused_news_first():
    items = [item(1, "Round-up: 10 stocks moving", tickers=12, hour=15),
             item(2, "Apple launches product", hour=9),
             item(3, "APPLE LAUNCHES PRODUCT", hour=8),          # same headline again
             item(4, "Apple earnings beat", hour=14)]
    assert [i.id for i in select_items(items, 10)] == ["4", "2", "1"]
    assert [i.id for i in select_items(items, 2)] == ["4", "2"]


def test_news_symbols_and_raw_items():
    assert to_news_symbol("BTC-USD") == "BTCUSD" and to_news_symbol("aapl") == "AAPL"
    raw = SimpleNamespace(id=42, headline="<b>Hi</b>", summary=None, source="benzinga", url=None,
                          created_at=datetime(2026, 11, 10, 12, 0), symbols=["AAPL", "MSFT"])
    made = to_item("AAPL", raw)
    assert (made.id, made.headline, made.summary, made.n_tickers) == ("42", "Hi", "", 2)
    assert made.created_at.tzinfo is not None


# ---------------------------------------------------------------- plumbing

def test_gemini_errors_are_classified_by_their_code():
    class APIError(Exception):          # google-genai puts the HTTP status in .code
        def __init__(self, code):
            super().__init__(f"{code} error")
            self.code, self.response = code, None

    assert is_rate_limit(APIError(429)) and not is_transient(APIError(429))
    assert is_transient(APIError(503)) and not is_rate_limit(APIError(503))
    assert not is_rate_limit(APIError(400)) and not is_transient(APIError(400))


def test_gemini_key_is_read_from_either_name(monkeypatch):
    monkeypatch.setattr(config, "ROOT", config.ROOT / "does-not-exist")   # ignore the real .env
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv(name, "abc123")
        assert config.load_secrets()["gemini_key"] == "abc123"
    monkeypatch.delenv("GOOGLE_API_KEY")
    assert config.load_secrets()["gemini_key"] is None
