"""The unattended daily run: fetch -> research -> paper --execute, plus a weekly re-test.

Started by Windows Task Scheduler (see scripts/install_schedule.ps1), or by
hand with `python main.py daily`. Each step runs as its own process, so one
failing step can never take the others down, and everything is appended to
logs/daily_run.log. The steps that trade write their own logs too
(paper_trading.log, research.log).

Safety does not depend on this file: `paper` still refuses to buy on stale
prices, still applies the stop-loss and position caps, and still skips symbols
with an order waiting. A missed or failed fetch therefore means "no new
trades today", not wrong trades.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

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


def daily_run(root: Path, sessions: list[str], today: str,
              run: Callable[[Path, Step], tuple[int, str]] = run_step) -> list[tuple[str, int]]:
    """Run today's steps. `sessions` are exchange trading days around today (from the
    broker's calendar); `today` is New York's date. Returns (step, exit code) pairs."""
    results = []
    with RunLog(root / "logs" / "daily_run.log", f"daily run for {today}") as say:
        if today not in sessions:
            say(f"{today} is not a trading day (weekend or holiday). Nothing to do.")
            return results
        weekly = is_last_trading_day_of_week(sessions, today)
        for step in DAILY:
            if step.weekly and not weekly:
                continue
            say(f"--- {step.name}: python main.py {' '.join(step.args)}")
            code, output = run(root, step)
            for line in output.splitlines()[-40:]:          # the tail is what matters; full logs exist
                say(f"    {line}")
            say(f"    -> {'ok' if code == 0 else f'exit code {code}'}")
            results.append((step.name, code))
        failed = [name for name, code in results if code != 0]
        say("Done: all steps ok." if not failed else f"Done with problems in: {', '.join(failed)}. "
            "The bot's own guards decide what is safe to trade; see the step output above.")
    return results
