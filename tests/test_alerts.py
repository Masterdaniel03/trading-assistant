"""Alerts: stored, logged, sent to each channel, and never able to stop a run."""
import json

import pytest

from tradebot import alerts, db
from tradebot.broker import PositionInfo
from tradebot.daily import daily_run, missed_session

WEEK = ["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-16"]


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "a.db"))
    db.init_db(c)
    return c


def collector():
    got = []
    return got, (lambda level, title, detail: got.append((level, title, detail)))


def fake_run(codes=None):
    def run(root, step):
        return (codes or {}).get(step.args[0], 0), f"line one\nlast line of {step.args[0]}"
    return run


# ---------------------------------------------------------------- raise_alert

def test_an_alert_is_stored_logged_and_sent_to_every_channel(conn, tmp_path):
    sent = []
    channels = {"phone": lambda title, body, level: sent.append((title, body, level))}
    alerts.raise_alert(conn, "error", "Daily run failed", "fetch prices: exit code 1", "daily",
                       channels=channels, log_path=tmp_path / "logs" / "alerts.log")
    assert sent == [("Daily run failed", "fetch prices: exit code 1", "error")]
    row = db.alerts_df(conn).iloc[0]
    assert (row["level"], row["title"], row["delivered"], row["read"]) == ("error", "Daily run failed", "phone", 0)
    text = (tmp_path / "logs" / "alerts.log").read_text(encoding="utf-8")
    assert "ERROR   Daily run failed" in text and "fetch prices: exit code 1" in text and "sent to: phone" in text


def test_a_channel_that_fails_is_recorded_and_the_others_still_get_it(conn):
    def broken(title, body, level):
        raise ConnectionError("no internet")
    got = []
    alerts.raise_alert(conn, "warning", "Stop-loss", channels={"phone": broken,
                                                               "desktop": lambda *a: got.append(a)})
    row = db.alerts_df(conn).iloc[0]
    assert len(got) == 1 and row["delivered"] == "desktop"
    assert row["failed"] == "phone: ConnectionError: no internet"


def test_unknown_level_is_refused(conn):
    with pytest.raises(ValueError):
        alerts.raise_alert(conn, "panic", "x", channels={})


def test_marking_alerts_as_read(conn):
    alerts.raise_alert(conn, "error", "one", channels={})
    alerts.raise_alert(conn, "warning", "two", channels={})
    assert len(db.alerts_df(conn, unread_only=True)) == 2
    db.mark_alerts_read(conn)
    assert db.alerts_df(conn, unread_only=True).empty and len(db.alerts_df(conn)) == 2


def test_channels_come_from_the_config_and_the_environment(monkeypatch):
    monkeypatch.setattr(alerts.sys, "platform", "win32")
    assert list(alerts.configured_channels({}, env={})) == ["desktop"]
    assert list(alerts.configured_channels({"alerts": {"desktop": False}}, env={})) == []
    both = alerts.configured_channels({"alerts": {"desktop": False}},
                                      env={"NTFY_TOPIC": "my-topic", "DISCORD_WEBHOOK_URL": "https://x"})
    assert list(both) == ["ntfy", "discord"]
    assert alerts.configured_channels({}, env={"TRADEBOT_ALERTS": "off", "NTFY_TOPIC": "t"}) == {}


def test_ntfy_message_goes_to_the_topic_with_a_priority(monkeypatch):
    calls = []

    class Reply:
        def raise_for_status(self):
            pass

    import requests
    monkeypatch.setattr(requests, "post", lambda url, **kw: calls.append((url, kw)) or Reply())
    alerts.send_ntfy("my-topic", "Daily run failed", "details", "error")
    url, kw = calls[0]
    assert url == "https://ntfy.sh/my-topic" and kw["data"] == b"details"
    assert kw["headers"]["Priority"] == "urgent" and kw["headers"]["Title"] == "Daily run failed"


# ---------------------------------------------------------------- the daily run

def test_a_failed_step_raises_an_error_alert_naming_the_step(conn, tmp_path):
    got, alert = collector()
    daily_run(tmp_path, WEEK, "2026-11-11", fake_run({"fetch": 1}), conn=conn, alert=alert, summary=True)
    assert len(got) == 1
    level, title, detail = got[0]
    assert level == "error" and "problem in fetch prices" in title
    assert "fetch prices (exit code 1)" in detail and "last line of fetch" in detail


def test_a_good_run_sends_a_summary_only_when_asked(conn, tmp_path):
    db.log_equity(conn, "alpaca_paper", 100_726.67, 50.0)
    got, alert = collector()
    daily_run(tmp_path, WEEK, "2026-11-11", fake_run(), conn=conn, alert=alert, summary=False)
    assert got == []
    daily_run(tmp_path, WEEK, "2026-11-12", fake_run(), conn=conn, alert=alert, summary=True)
    level, title, detail = got[0]
    assert level == "info" and title == "Daily run 2026-11-12: all ok"
    assert "Account value $100,726.67" in detail and "Orders sent in this run: 0" in detail


def test_every_run_is_recorded_with_its_steps(conn, tmp_path):
    daily_run(tmp_path, WEEK, "2026-11-11", fake_run({"research": 1}), conn=conn)
    run = db.runs_df(conn).iloc[0]
    assert (run["session"], run["status"]) == ("2026-11-11", "failed") and run["finished"]
    steps = json.loads(run["steps"])
    assert [s["name"] for s in steps] == ["fetch prices", "research notes (Gemini)", "paper trading",
                                          "model prices", "model scores", "bot league"]
    assert [s["code"] for s in steps] == [0, 1, 0, 0, 0, 0] and "last line of research" in steps[1]["output"]


def test_a_day_that_is_not_a_trading_day_records_nothing(conn, tmp_path):
    got, alert = collector()
    daily_run(tmp_path, WEEK, "2026-11-14", fake_run(), conn=conn, alert=alert, summary=True)
    assert got == [] and db.runs_df(conn).empty


def test_a_missed_previous_day_is_reported_once_the_automation_is_in_use(conn, tmp_path):
    assert missed_session(conn, WEEK, "2026-11-11") is None            # first run ever: nothing was missed
    daily_run(tmp_path, WEEK, "2026-11-10", fake_run(), conn=conn)
    assert missed_session(conn, WEEK, "2026-11-11") is None            # the 10th ran
    assert missed_session(conn, WEEK, "2026-11-12") == "2026-11-11"    # the 11th did not
    got, alert = collector()
    daily_run(tmp_path, WEEK, "2026-11-12", fake_run(), conn=conn, alert=alert)
    assert [(level, title) for level, title, _ in got] == [("warning", "No daily run finished on 2026-11-11")]
    assert missed_session(conn, WEEK, "2026-11-16") == "2026-11-13"    # Friday missed, seen on Monday


def test_a_run_that_was_cut_off_counts_as_missed(conn):
    db.start_run(conn, "2026-11-10")                                   # started, never finished
    assert missed_session(conn, WEEK, "2026-11-11") == "2026-11-10"


# ---------------------------------------------------------------- the paper bot

def test_a_stop_loss_sale_raises_a_warning(bot):
    bot.store("AAA"), bot.store("BBB")
    bot.broker.held = {"AAA": PositionInfo(20_000.0, -0.09)}
    bot.run()
    titles = list(db.alerts_df(bot.conn)["title"])
    assert titles == ["Stop-loss: selling AAA (-9.0%)"]


def test_stale_prices_raise_a_warning_but_a_dry_run_stays_quiet(bot):
    bot.store("AAA", days_old=3), bot.store("BBB")
    bot.run(execute=False)
    assert db.alerts_df(bot.conn).empty
    bot.run()
    assert list(db.alerts_df(bot.conn)["title"]) == ["Paper bot did not trade: prices are out of date"]


def test_pausing_blocks_buys_but_not_sells(bot):
    bot.store("AAA"), bot.store("BBB")
    db.set_setting(bot.conn, "paused", "1")
    bot.broker.held = {"ZZZ": PositionInfo(5_000.0, 0.01)}            # not managed: must still be closed
    broker = bot.run()
    assert broker.bought == [] and broker.closed == ["ZZZ"]
    assert "Paused" in bot.log_text()
    db.set_setting(bot.conn, "paused", "0")
    bot.broker.held = {}
    assert bot.run().bought == ["AAA", "BBB"]


def test_a_stop_loss_still_sells_while_paused(bot):
    bot.store("AAA"), bot.store("BBB")
    db.set_setting(bot.conn, "paused", "1")
    bot.broker.held = {"AAA": PositionInfo(20_000.0, -0.09)}
    broker = bot.run()
    assert broker.closed == ["AAA"] and broker.bought == []
