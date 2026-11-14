"""The unattended daily run: fetch -> research -> paper --execute, plus a weekly re-test.

Started by Windows Task Scheduler (see scripts/install_schedule.ps1), or by
hand with `python main.py daily`. Each step runs as its own process, so one
failing step can never take the others down, and everything is appended to
logs/daily_run.log. The steps that trade write their own logs too
(paper_trading.log, research.log).

Given a database, each run is also recorded there (the dashboard's run history),
and given an `alert` function it reports problems: a failed step, a run that
never happened on the previous trading day, and, if wanted, a short "all ok"
after a good run, so that silence means trouble.

Safety does not depend on this file: `paper` still refuses to buy on stale
prices, still applies the stop-loss and position caps, and still skips symbols
with an order waiting. A missed or failed fetch therefore means "no new
trades today", not wrong trades.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from tradebot import db
from tradebot.runlog import RunLog


@dataclass(frozen=True)
class Step:
    name: str
    args: list[str]
    weekly: bool = False      # only on the last trading day of the week


DAILY = [
    Step("fetch prices", ["fetch"]),
    Step("research notes (Gemini)", ["research"]),
    Step("paper trading", ["paper", "--execute"]),
    Step("bot league", ["league"]),
    Step("update the news archive", ["news-archive"], weekly=True),
    Step("walk-forward re-test", ["walk-forward"], weekly=True),
    Step("news test", ["news-test"], weekly=True),
]


def run_step(root: Path, step: Step, timeout_s: int = 1800) -> tuple[int, str]:
    """Run `python main.py <args>` and return (exit code, its output)."""
    try:
        done = subprocess.run([sys.executable, str(root / "main.py"), *step.args], cwd=root,
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout_s, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        return done.returncode, (done.stdout + done.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, f"stopped after {timeout_s // 60} minutes without finishing"


def is_last_trading_day_of_week(sessions: list[str], today: str) -> bool:
    """True if no later session falls in the same Monday-to-Sunday week."""
    week = datetime.fromisoformat(today).isocalendar()[:2]
    return not any(d > today and datetime.fromisoformat(d).isocalendar()[:2] == week for d in sessions)


def missed_session(conn, sessions: list[str], today: str) -> str | None:
    """The previous trading day, if the automation was already in use and no run finished for it."""
    earlier = [d for d in sessions if d < today]
    if not earlier:
        return None
    previous = max(earlier)
    first = conn.execute("SELECT MIN(session) FROM daily_runs").fetchone()[0]
    if first is None or first > previous:
        return None                                   # the automation did not exist yet
    done = conn.execute("SELECT 1 FROM daily_runs WHERE session = ? AND status IN ('ok', 'failed')",
                        (previous,)).fetchone()
    return None if done else previous


def _account_line(conn, started: str) -> str:
    row = conn.execute("SELECT equity FROM equity_snapshots ORDER BY ts DESC LIMIT 1").fetchone()
    sent = conn.execute("SELECT COUNT(*) FROM trades WHERE order_id IS NOT NULL AND order_id != '' "
                        "AND ts >= ?", (started,)).fetchone()[0]
    value = f"Account value ${row[0]:,.2f}. " if row else ""
    return f"{value}Orders sent in this run: {sent}."


def daily_run(root: Path, sessions: list[str], today: str,
              run: Callable[[Path, Step], tuple[int, str]] = run_step, *,
              conn=None, alert: Callable[[str, str, str], object] | None = None,
              summary: bool = False) -> list[tuple[str, int]]:
    """Run today's steps. `sessions` are exchange trading days around today (from the
    broker's calendar); `today` is New York's date. Returns (step, exit code) pairs.

    conn:    record the run in the database (table daily_runs).
    alert:   called as alert(level, title, detail) when something needs attention.
    summary: also send a short "all ok" after a good run.
    """
    results = []
    tell = alert or (lambda level, title, detail: None)
    with RunLog(root / "logs" / "daily_run.log", f"daily run for {today}") as say:
        if today not in sessions:
            say(f"{today} is not a trading day (weekend or holiday). Nothing to do.")
            return results
        if conn is not None:
            missed = missed_session(conn, sessions, today)
            if missed:
                say(f"Note: no run finished for the previous trading day, {missed}.")
                tell("warning", f"No daily run finished on {missed}",
                     "The PC was probably off or signed out. No orders were sent that day; "
                     "today's run continues as normal.")
        run_id = db.start_run(conn, today) if conn is not None else None
        started = db.now_utc()
        weekly = is_last_trading_day_of_week(sessions, today)
        record = []
        for step in DAILY:
            if step.weekly and not weekly:
                continue
            say(f"--- {step.name}: python main.py {' '.join(step.args)}")
            began = time.monotonic()
            code, output = run(root, step)
            lines = output.splitlines()
            for line in lines[-40:]:                        # the tail is what matters; full logs exist
                say(f"    {line}")
            say(f"    -> {'ok' if code == 0 else f'exit code {code}'}")
            results.append((step.name, code))
            record.append({"name": step.name, "code": code, "seconds": round(time.monotonic() - began, 1),
                           "output": "\n".join(lines[-200:])})
        failed = [name for name, code in results if code != 0]
        say("Done: all steps ok." if not failed else f"Done with problems in: {', '.join(failed)}. "
            "The bot's own guards decide what is safe to trade; see the step output above.")
        if conn is not None:
            db.finish_run(conn, run_id, "failed" if failed else "ok", json.dumps(record))
        if failed:
            detail = []
            for item in record:
                if item["code"] != 0:
                    tail = [line.strip() for line in item["output"].splitlines() if line.strip()][-3:]
                    detail.append(f"{item['name']} (exit code {item['code']}): " + " | ".join(tail))
            tell("error", f"Daily run {today}: problem in {', '.join(failed)}", "\n".join(detail))
        elif summary:
            ok = ", ".join(name for name, _ in results)
            line = _account_line(conn, started) if conn is not None else ""
            tell("info", f"Daily run {today}: all ok", f"{line}\nSteps: {ok}.".strip())
    return results
