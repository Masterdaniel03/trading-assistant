"""The unattended daily run, with fake steps. Nothing is fetched or traded."""
from tradebot.daily import DAILY, daily_run, is_last_trading_day_of_week

# Mon 9 .. Fri 13 November 2026, then the next week; Fri 27th is the day after Thanksgiving
WEEK = ["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-16"]


def recorder(codes=None):
    calls = []

    def run(root, step):
        calls.append(step.args[0])
        return (codes or {}).get(step.args[0], 0), f"output of {step.args[0]}"
    return run, calls


def test_a_normal_weekday_runs_fetch_research_paper_in_that_order(tmp_path):
    run, calls = recorder()
    results = daily_run(tmp_path, WEEK, "2026-11-11", run)
    assert calls == ["fetch", "research", "paper"]
    assert all(code == 0 for _, code in results)
    text = (tmp_path / "logs" / "daily_run.log").read_text(encoding="utf-8")
    assert "daily run for 2026-11-11" in text and "output of paper" in text and "all steps ok" in text


def test_the_paper_step_sends_orders():
    assert next(s for s in DAILY if s.args[0] == "paper").args == ["paper", "--execute"]


def test_weekends_and_holidays_do_nothing(tmp_path):
    run, calls = recorder()
    assert daily_run(tmp_path, WEEK, "2026-11-14", run) == []        # Saturday
    assert daily_run(tmp_path, ["2026-11-25", "2026-11-27"], "2026-11-26", run) == []   # Thanksgiving
    assert calls == []
    assert "not a trading day" in (tmp_path / "logs" / "daily_run.log").read_text(encoding="utf-8")


def test_weekly_re_tests_run_on_the_last_trading_day_of_the_week(tmp_path):
    run, calls = recorder()
    daily_run(tmp_path, WEEK, "2026-11-13", run)                     # Friday
    assert calls == ["fetch", "research", "paper", "news-archive", "walk-forward", "news-test"]
    assert is_last_trading_day_of_week(["2026-11-25", "2026-11-26"], "2026-11-26")   # short week
    assert not is_last_trading_day_of_week(WEEK, "2026-11-12")


def test_a_failed_step_does_not_stop_the_next_ones(tmp_path):
    run, calls = recorder({"fetch": 1, "research": 1})
    results = daily_run(tmp_path, WEEK, "2026-11-11", run)
    assert calls == ["fetch", "research", "paper"]                   # paper's own guards decide
    assert dict(results) == {"fetch prices": 1, "research notes (Gemini)": 1, "paper trading": 0}
    assert "problems in: fetch prices, research notes (Gemini)" in (
        tmp_path / "logs" / "daily_run.log").read_text(encoding="utf-8")
