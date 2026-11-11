"""A plain-text record of each run, written as it happens.

    with RunLog(path, "paper --execute") as say:
        say("equity $100,000")

Every line goes to the screen and to the log file. Lines are flushed one by
one, so a crash part-way through still leaves a record of what was already
sent. Each run starts with a ruled, timestamped header.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

RULE = "=" * 72


class RunLog:
    def __init__(self, path: str | Path, title: str, now: datetime | None = None):
        self.path = Path(path)
        self.title = title
        self.started = now or datetime.now().astimezone()
        self._fh = None

    def __enter__(self) -> "RunLog":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        for line in (RULE, f"{self.started:%Y-%m-%d %H:%M:%S %z}  {self.title}", RULE):
            self._write(line)
        return self

    def __call__(self, message: str = "") -> None:
        print(message)
        self._write(message)

    def _write(self, line: str) -> None:
        self._fh.write(line + "\n")
        self._fh.flush()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is SystemExit:
            if exc.code not in (None, 0):
                self._write(f"Stopped: {exc.code}")   # Python prints the message itself
        elif exc_type is not None:
            self._write(f"ERROR: {exc_type.__name__}: {exc}")
        self._write("")
        self._fh.close()
        return False
