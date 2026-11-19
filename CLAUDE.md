# CLAUDE.md

Guidance for Claude Code in this repo. README.md is the full user guide; read it for what each part does.

## What this is
A research and paper-trading system: price data → backtests and walk-forward tests → an Alpaca **paper** bot with hard risk limits, a simulated bot league, an LLM news-research layer (Gemini), a stock-ranking model (price inputs plus point-in-time SEC fundamentals), alerts and a Streamlit dashboard. The user's goal is a strategy of their own that predicts trends from fundamentals plus capital-flow / candle (k-line) data.

## Environment
- Windows 11, PowerShell. Python lives in `.venv` (`.venv\Scripts\python`).
- Some package versions are blocked by Windows Smart App Control; `requirements.txt` excludes them. Don't upgrade past those pins without checking.
- Keys are in `.env` (see `.env.example`): Alpaca paper keys, `GEMINI_API_KEY`, `SEC_USER_AGENT`, optional `NTFY_TOPIC` / `DISCORD_WEBHOOK_URL`. Never commit `.env` and never print key values.

## Commands
- Tests: `.venv\Scripts\python -m pytest -q` (about 90 seconds). Tests set `TRADEBOT_ALERTS=off`, so they never send real alerts. A fake broker is in `tests/conftest.py`.
- Everything else is `python main.py <command>`. See the README's Commands table. `paper` is a dry run unless given `--execute`.
- Long data jobs (`model-fetch --full`, `fundamentals-fetch --full`, `news-archive`, `model-test --rebuild`) take minutes: run them in the background.

## Layout
- `main.py`: the CLI (one `cmd_*` function per command). `config.yaml`: all settings, loaded by `tradebot/config.py`, which also resolves paths.
- `tradebot/`: `data.py` prices, `strategies.py`, `backtest.py`, `walkforward.py`, `rebalance.py` + `risk.py` + `broker.py` (paper bot), `league.py`, `research.py` + `llm.py` + `news.py` (LLM notes), `newstest.py` + `news_archive.py`, `modeldata.py` + `features.py` + `fundamentals.py` + `rankmodel.py` (ranking model), `daily.py` (scheduled run), `alerts.py`, `dashboard_data.py` + `dashboard_charts.py`, `manifest.py` (run records on test reports).
- `dashboard.py`: the Streamlit page. `universes/`: the no-hindsight S&P 500 list as of end 2015, plus its SEC CIKs and CIK successors. `scripts/`: `build_universe.py`, `install_schedule.ps1`, `league_attribution.py` (what the stop-loss and costs take from the rank bots; read-only).
- Generated and not committed: `data/*.db`, `data/models/`, `logs/`, most of `reports/`.

## Rules that matter here
- **No look-ahead.** Every input must be known at decision time: decide after the close, trade at the next open; a filing counts from the trading day after its filing date; restatements only from the restating filing's date. Each test has a point-in-time audit; keep it passing.
- **Settings are fixed before a test runs.** Never tune a model, horizon, input list or universe against a test result. A new idea is a new test, with its settings written into `config.yaml` (with the date) before its first run. Stored model scores are never recomputed. Each test report carries a run record (`tradebot/manifest.py`); a new test lists the config sections it reads in `manifest.SETTINGS`. A changed league bot is added as a new bot, so the old one keeps its live record.
- **Paper only.** Live trading stays blocked (`risk.allow_live` plus `ALPACA_PAPER`). Don't weaken a risk guard (position cap, stop-loss, daily loss limit, stale-data check, pause switch) without being asked.
- Paper runs may be done without asking, as long as each run leaves its saved log (`logs/paper_trading.log`).
- Report results honestly, including null results. Compare against buy and hold and against SPY.

## Style
- Write user-facing text (README, reports, CLI output, dashboard captions, alerts) in plain English: short sentences, terms explained, no unexplained jargon. Match the existing README tone.
- Code comments explain *why*. Each module opens with a docstring on what it does and its point-in-time rules.
- Keep the README up to date when commands, settings or results change: the Commands table, the relevant section and the Roadmap.
- Add or update tests with each change. The daily-run step order is tested in `tests/test_daily.py`, `tests/test_alerts.py` and `tests/test_rank_league.py`.

## Git
- Work on feature branches, never force-push. Commit and push each tested checkpoint.
- Editor and agent-tool files (`.vscode/`, `.agents/`, `.claude/skills/`) are ignored; don't stage them.
