"""Run records: which code, settings and stocks produced a test report.

A test's settings are written into config.yaml before its first run and never
tuned against its result. A run record makes that checkable afterwards. Each
test report (model-test, news-test, walk-forward) gets one, as a "Run record"
section at the end of a Markdown report and as a JSON file next to the report
(`<report>.run.json`; the only record for the walk-forward CSV). It holds:

* the git commit the code was at, and whether tracked files had uncommitted
  changes (an uncommitted config.yaml shows up here too);
* a fingerprint of the test's own settings, read from config.yaml as written,
  so an unrelated edit elsewhere in the file leaves it unchanged;
* a fingerprint of the whole config.yaml file;
* a fingerprint of the stock list, and of each data file the test reads;
* when the test started, and the Python and package versions.

A fingerprint is a SHA-256 hash: the same input always gives the same one, and
any change gives a different one. The record is taken when the test starts, so
it describes what the test ran with. Nothing here changes a result.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

from tradebot.config import ROOT

# The parts of config.yaml each test reads. A dotted name is a single key inside a section.
SETTINGS = {
    "model-test": ("model", "fundamentals", "backtest.cost_bps"),
    "news-test": ("news_test", "backtest.cost_bps"),
    "walk-forward": ("walk_forward", "backtest", "risk.max_risk_per_position_pct", "risk.max_position_pct",
                     "risk.stop_cooldown_bars"),
}
PACKAGES = ("numpy", "pandas", "scikit-learn")
SHOWN = 16            # characters of a fingerprint shown in a report; the JSON file keeps all 64


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(value) -> str:
    """Fingerprint of a value written as sorted JSON, so the order settings are written in does not matter."""
    return sha256(json.dumps(value, sort_keys=True, default=str).encode())


def pick(raw: dict, names) -> dict:
    """The named sections, or dotted keys, of a config as written. One that is missing is recorded as None."""
    out = {}
    for name in names:
        value = raw
        for part in name.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        out[name] = value
    return out


def git_state(root: Path = ROOT) -> dict:
    """The commit the code is at, and whether tracked files have uncommitted changes. None where unknown."""
    def git(*args):
        done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return done.stdout.strip() if done.returncode == 0 else None

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=no") if commit else None
    except (OSError, subprocess.SubprocessError):       # git missing or stuck: the record says "unknown"
        commit = status = None
    return {"commit": commit, "uncommitted_changes": None if status is None else bool(status)}


def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _shown_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def build(test: str, symbols, files=(), options: dict | None = None,
          config_path: Path | None = None, root: Path = ROOT) -> dict:
    """The record for one run of `test` (a key of SETTINGS), taken before the test starts."""
    config_path = Path(config_path or root / "config.yaml")
    text = config_path.read_bytes()
    settings = pick(yaml.safe_load(text) or {}, SETTINGS[test])
    stocks = sorted(set(symbols))
    data_files = {_shown_path(Path(f), root): sha256(Path(f).read_bytes()) if Path(f).exists() else None
                  for f in files}
    return {"test": test, "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **git_state(root),
            "settings_keys": list(SETTINGS[test]), "settings": settings, "settings_sha256": fingerprint(settings),
            "config_sha256": sha256(text),
            "stocks": len(stocks), "stocks_sha256": fingerprint(stocks), "data_files": data_files,
            "options": options or {}, "versions": versions()}


def json_path(report: Path) -> Path:
    """model_test_report.md -> model_test_report.run.json, next to it."""
    report = Path(report)
    return report.with_name(report.stem + ".run.json")


def write(record: dict, report: Path) -> Path:
    path = json_path(report)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def markdown(record: dict, report: Path) -> list[str]:
    """The "Run record" section for the end of a Markdown report."""
    commit = record.get("commit")
    if not commit:
        code = "unknown (git could not be read)"
    else:
        code = f"commit {commit[:10]}" + (", with uncommitted changes to tracked files"
                                           if record.get("uncommitted_changes") else "")
    keys = ", ".join(f"`{key}`" for key in record["settings_keys"])
    lines = ["## Run record", "",
             "What this report was produced with, recorded when the test started. Fingerprints are SHA-256 "
             f"hashes (first {SHOWN} characters): the same input gives the same fingerprint, and any change "
             "gives a different one. Two reports with the same settings fingerprint ran with the same settings, "
             "whatever else changed in config.yaml.", "",
             f"- Code: {code}",
             f"- This test's settings ({keys} in config.yaml): {record['settings_sha256'][:SHOWN]}",
             f"- All of config.yaml: {record['config_sha256'][:SHOWN]}",
             f"- Stock list ({record['stocks']} tickers): {record['stocks_sha256'][:SHOWN]}"]
    for name, digest in record["data_files"].items():
        lines.append(f"- {name}: {digest[:SHOWN] if digest else 'missing'}")
    v = record["versions"]
    packages = ", ".join(f"{name} {v[name]}" for name in PACKAGES if v.get(name))
    lines += [f"- Started {record['started']} with Python {v['python']}, {packages}",
              f"- Full record, with the settings themselves: {json_path(report).name}"]
    return lines


def append(report_text: str, record: dict, report: Path) -> str:
    return report_text.rstrip("\n") + "\n\n" + "\n".join(markdown(record, report))
