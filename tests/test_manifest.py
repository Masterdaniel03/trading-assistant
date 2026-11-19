"""Run records: which code, settings and stocks produced a test report (tradebot/manifest.py)."""
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from tradebot import config, db, manifest

CONFIG = """
watchlist: [SPY]
backtest: {initial_capital: 4000, cost_bps: 5}
model: {universe: wide, horizon_days: 5, models: {ridge: {alpha: 100.0}}}
fundamentals: {max_age_days: 365}
news_test: {start: "2020-01-01", universe: [AAPL, MSFT]}
walk_forward: {train_years: 5, test_years: 1, grids: {sma_crossover: {fast: [20], slow: [100]}}}
risk: {max_position_pct: 0.2, max_risk_per_position_pct: 0.015, stop_cooldown_bars: 20, daily_loss_limit_pct: 0.03}
"""


def write_config(folder, text=CONFIG):
    path = folder / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def record(folder, test="model-test", text=CONFIG, symbols=("AAPL", "MSFT"), **kw):
    return manifest.build(test, symbols, config_path=write_config(folder, text), root=folder, **kw)


def test_a_fingerprint_ignores_the_order_settings_are_written_in():
    assert manifest.fingerprint({"a": 1, "b": [1, 2]}) == manifest.fingerprint({"b": [1, 2], "a": 1})
    assert manifest.fingerprint({"a": 1}) != manifest.fingerprint({"a": 2})


def test_a_tests_settings_fingerprint_changes_only_with_its_own_settings(tmp_path):
    before = record(tmp_path)
    unrelated = record(tmp_path, text=CONFIG.replace("watchlist: [SPY]", "watchlist: [SPY, QQQ]")
                       .replace("initial_capital: 4000", "initial_capital: 9000"))
    own = record(tmp_path, text=CONFIG.replace("alpha: 100.0", "alpha: 10.0"))
    cost = record(tmp_path, text=CONFIG.replace("cost_bps: 5", "cost_bps: 10"))
    assert unrelated["settings_sha256"] == before["settings_sha256"]      # the model test reads neither
    assert unrelated["config_sha256"] != before["config_sha256"]          # but the file did change
    assert own["settings_sha256"] != before["settings_sha256"]
    assert cost["settings_sha256"] != before["settings_sha256"]           # backtest.cost_bps is read
    assert before["settings"]["backtest.cost_bps"] == 5 and before["settings"]["model"]["horizon_days"] == 5
    news = record(tmp_path, test="news-test")
    assert set(news["settings"]) == {"news_test", "backtest.cost_bps"}


def test_the_stock_list_fingerprint_ignores_order_and_repeats(tmp_path):
    a = record(tmp_path, symbols=["MSFT", "AAPL"])
    b = record(tmp_path, symbols=["AAPL", "MSFT", "AAPL"])
    c = record(tmp_path, symbols=["AAPL"])
    assert a["stocks_sha256"] == b["stocks_sha256"] != c["stocks_sha256"]
    assert a["stocks"] == 2


def test_data_files_are_fingerprinted_by_their_content(tmp_path):
    (tmp_path / "ciks.csv").write_text("ticker,cik\nAAPL,320193\n", encoding="utf-8")
    first = record(tmp_path, files=[tmp_path / "ciks.csv", tmp_path / "gone.csv"])
    assert set(first["data_files"]) == {"ciks.csv", "gone.csv"} and first["data_files"]["gone.csv"] is None
    (tmp_path / "ciks.csv").write_text("ticker,cik\nAAPL,1\n", encoding="utf-8")
    again = record(tmp_path, files=[tmp_path / "ciks.csv"])
    assert again["data_files"]["ciks.csv"] != first["data_files"]["ciks.csv"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_the_commit_and_uncommitted_changes_are_recorded(tmp_path):
    assert manifest.git_state(tmp_path) == {"commit": None, "uncommitted_changes": None}   # not a repository

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=test", "-c", "user.email=test@example.com",
                        *args], check=True, capture_output=True)

    write_config(tmp_path)
    git("init", "-q")
    git("add", "config.yaml")
    git("commit", "-q", "-m", "settings")
    clean = manifest.git_state(tmp_path)
    assert len(clean["commit"]) == 40 and clean["uncommitted_changes"] is False
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")            # a file git does not track: no change
    assert manifest.git_state(tmp_path)["uncommitted_changes"] is False
    write_config(tmp_path, CONFIG.replace("alpha: 100.0", "alpha: 10.0"))  # an uncommitted setting
    assert manifest.git_state(tmp_path) == {"commit": clean["commit"], "uncommitted_changes": True}


def test_a_missing_git_is_recorded_as_unknown(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(manifest.subprocess, "run", missing)
    assert manifest.git_state(tmp_path) == {"commit": None, "uncommitted_changes": None}
    assert "- Code: unknown (git could not be read)" in manifest.markdown(record(tmp_path), tmp_path / "r.md")


def test_the_record_is_saved_next_to_the_report_and_summed_up_at_its_end(tmp_path):
    rec = {**record(tmp_path), "commit": "0123456789" * 4, "uncommitted_changes": True}
    report = tmp_path / "model_test_report.md"
    text = manifest.append("# Report\n\nbody\n", rec, report)
    assert text.startswith("# Report\n\nbody\n\n## Run record\n")
    assert "- Code: commit 0123456789, with uncommitted changes to tracked files" in text
    assert ("- This test's settings (`model`, `fundamentals`, `backtest.cost_bps` in config.yaml): "
            + rec["settings_sha256"][:16]) in text
    assert "model_test_report.run.json" in text
    saved = manifest.write(rec, report)
    assert saved == tmp_path / "model_test_report.run.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["settings"] == rec["settings"]
    assert manifest.json_path(tmp_path / "walk_forward_summary.csv").name == "walk_forward_summary.run.json"


def test_every_test_reads_settings_that_exist_in_the_real_config():
    raw = yaml.safe_load((config.ROOT / "config.yaml").read_text(encoding="utf-8"))
    for test, names in manifest.SETTINGS.items():
        missing = [name for name, value in manifest.pick(raw, names).items() if value is None]
        assert not missing, f"{test}: {missing}"


def test_walk_forward_saves_its_run_record_next_to_its_summary(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "REPORTS", tmp_path / "reports")
    metrics = {"sharpe": 1.0, "cagr": 0.1, "max_drawdown": -0.1}
    fake = SimpleNamespace(folds=[{}], mean_train_sharpe=1.0, oos=SimpleNamespace(metrics=metrics),
                           benchmark=SimpleNamespace(metrics=metrics), folds_beating_benchmark=1)
    monkeypatch.setattr(main, "walk_forward", lambda *args, **kwargs: fake)
    cfg = config.load_config()
    cfg["db_path"] = str(tmp_path / "prices.db")
    db.init_db(db.connect(cfg["db_path"]))
    main.cmd_walk_forward(cfg, main.build_parser().parse_args(["walk-forward", "SPY", "--strategy", "momentum"]))
    saved = json.loads((tmp_path / "reports" / "walk_forward_summary.run.json").read_text(encoding="utf-8"))
    assert saved["test"] == "walk-forward" and saved["stocks"] == 1
    assert saved["options"] == {"symbols": ["SPY"], "strategy": "momentum", "no_stop": False}
    assert (tmp_path / "reports" / "walk_forward_summary.csv").exists()
