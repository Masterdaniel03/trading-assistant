# AI Trading Assistant

A research and paper-trading system for stocks and crypto. It ingests market data, backtests strategies with transaction costs and no look-ahead bias, generates daily signals, runs an automated bot on an Alpaca paper account with hard risk limits, and logs manual (Fidelity) trades so AI-assisted and discretionary decisions can be compared against a buy-and-hold benchmark.

## Architecture

```
 yfinance ──► data.py ──► SQLite (prices, signals, trades, equity_snapshots)
                               │
          strategies.py ◄──────┤
                │              │
          backtest.py ──► reports/ (metrics, equity + drawdown charts)
                │
          rebalance.py + risk.py ──► broker.py ──► Alpaca (paper)
                                                    
  manual Fidelity trades ──► `log-trade` ──► trades table (followed_ai flag)

  Alpaca news ──► news.py ──► research.py ──► Gemini ──► answer checked ──► signals (llm_news)
```

**Two tracks**
- **Track A (automated, paper):** the bot trades an Alpaca paper account (currently $100,000).
- **Track B (assisted, real):** `signals` prints suggestions; you place trades yourself in Fidelity and record them with `log-trade --followed-ai yes|no`. (Fidelity has no retail trading API, and automating its website breaks its terms of service, so the human stays in the loop.)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your Alpaca PAPER keys, and a Gemini key for `research`
python main.py init
python main.py fetch
```

## Commands

| Command | What it does |
|---|---|
| `python main.py fetch [SYMBOLS...] [--full] [--source alpaca]` | Download/refresh daily prices (resumes from the last stored date; refetches full history automatically after a split/dividend re-adjustment). `--source alpaca` gets them from Alpaca instead of Yahoo, for stocks Yahoo no longer serves, such as delisted ones |
| `python main.py quote [SYMBOLS...]` | Real-time bid/ask from Alpaca (free IEX feed for stocks; crypto needs no keys) |
| `python main.py backtest SPY sma_crossover` | Backtest one strategy vs buy & hold, save chart to `reports/` |
| `python main.py backtest-all` | Every watchlist symbol × strategy → `reports/backtest_summary.csv` |
| `python main.py walk-forward [SYMBOLS...] [--strategy NAME] [--verbose]` | Out-of-sample test: picks parameters on each training window, trades them on the next window → `reports/walk_forward_summary.csv` |
| `python main.py signals` | Today's target positions (also logged to the DB) |
| `python main.py research [SYMBOLS...] [--refresh] [--show-prompt]` | An LLM reads the last few days of news and writes a stance with reasons (advisory; see below). `--show-prompt` prints what would be sent and sends nothing |
| `python main.py research-score` | Average return after bullish / neutral / bearish notes, once they are old enough to score |
| `python main.py news-archive` | Download historical news for the news test (resumable; about 10 minutes the first time) |
| `python main.py news-test` | Walk-forward test: does past news predict the next session? → `reports/news_test_report.md` (see below) |
| `python main.py paper` | Dry run: shows orders the bot would send |
| `python main.py paper --execute` | Sends those orders to Alpaca paper. Every run, dry or not, is appended to `logs/paper_trading.log`; the next run records what each earlier order filled at |
| `python main.py log-trade buy AAPL 2 228.50 --followed-ai yes --notes "..."` | Record a manual Fidelity trade |
| `python main.py report` | Trade counts, % of manual trades that followed the AI, latest equity |
| `pytest` | Run tests (indicators, signals, look-ahead, costs, risk limits, paper runs, research checks, DB) |

Settings (watchlist, strategy parameters, risk limits) are in `config.yaml`.

## LLM research layer
`python main.py research` gives the model (Gemini, set in `config.yaml`) two things per symbol: a few price facts computed from the stored closes, and up to 12 recent news items from Alpaca's news feed. It answers with a stance (bullish, neutral or bearish), a confidence, a short summary, its reasoning, risks, and the news items it relied on.

The answer is kept only if it passes these checks:
- it has the expected shape;
- every news item it cites was really supplied;
- a bullish or bearish view cites at least one item;
- every dollar amount or percentage it quotes appears in the supplied material.

A kept answer is logged as a signal with strategy `llm_news`, with the reasoning in `signals.reason`. A failed answer is stored as rejected with the reason, and logs no signal. With no recent news the model is not called. Every attempt is in the `research` table and every run in `logs/research.log`. The same inputs reuse the stored answer, which saves free-tier quota.

**The bot does not trade on these notes.** An LLM's view can't be backtested honestly, because the model may already know what happened next. The only fair test is forward: log the notes now and compare them with later prices using `research-score`.

## News test
`python main.py news-test` asks one question: do headlines that were already public before the open predict how a stock does that day, relative to the others? It is research only and places no trades. The universe is the 30 Dow stocks as of 2020-01-01, fixed up front so the choice uses no hindsight.

No prediction may use anything that was not yet known:
- **When a headline is known.** The archive serves each article's latest text, so a headline counts as known from its last revision. Articles revised more than an hour after publication are dropped.
- **Which session it can affect.** The decision for a trading day is made at 09:00 New York. News after that waits for the next session.
- **What is predicted.** The day's open-to-close return minus the universe average, which starts after the decision.
- **What the model learns from.** The text model's vocabulary and weights are refitted each quarter on past sessions only.
- **Controls** (momentum, volatility, trading volume) use prices up to the previous close.

The report includes the point-in-time audit, rank IC, a long-short return before and after costs, a shuffled-dates test, and a deliberately leaky version that shows what look-ahead would look like. All settings were fixed before the first run; changing them to improve the result would be its own form of look-ahead.

## Safety rails
- Paper trading by default. Live trading raises an error unless `risk.allow_live: true` **and** `ALPACA_PAPER=false`.
- Max 20% of equity in any one symbol, no margin, and no new buys after a 3% daily loss (sells that reduce risk still go through).
- Running `paper --execute` twice is safe: a symbol with an order still waiting to fill gets no second order. Sells are sent before buys.
- Each position risks at most 1.5% of equity: a 7.5% stop-loss (1.5% / 20%), checked each time `paper` runs, followed by a 20-bar cooldown. It is not a resting order, so a price gap or a long gap between runs can lose more. Stops still execute when the daily-loss or stale-data guards block buying.
- `backtest` and `walk-forward` include the stop; pass `--no-stop` to compare without it.
- If any bot symbol's prices are missing or older than `risk.max_data_age_days` (default 4), `paper` sends stop-loss exits only.
- `fetch` stores finished daily bars only: today's stock bar is left out until 16:15 New York time, and today's crypto bar until the UTC day ends.
- Positions in the paper account that aren't in `paper_bot.symbols` are closed (`paper_bot.close_unlisted`).
- `paper` is a dry run unless you pass `--execute`.
- Never commit `.env`. Never give any key withdrawal permissions.

## Daily routine (once it runs)
After the day's prices are final (1:15 pm Pacific, 15 minutes after the close): `fetch` → `signals` → `paper --execute`. Orders sent then fill at the next open. Log any Fidelity trades the same day. A cron job or GitHub Actions schedule can automate the first three.

## Roadmap
- [x] Phase 1: data pipeline, SQLite logging, backtester, 2 baseline strategies, Alpaca paper bot, manual trade log
- [x] Phase 2a: walk-forward / out-of-sample testing; more strategies (momentum, volatility targeting)
- [ ] Phase 2b: crypto on Alpaca (`BTC/USD`)
- [x] Phase 3a: LLM research layer (recent news → a checked stance with written reasoning, stored in `signals.reason`; forward scoring)
- [ ] Phase 3b: earnings dates and filings as extra research inputs; decide from the scores whether the notes should influence the bot
- [x] Phase 3c: point-in-time news test (walk-forward, headlines → next session)
- [ ] Phase 4: Streamlit dashboard: bot vs. manual vs. SPY, Sharpe, drawdown, AI-follow rate
- [ ] Phase 5: write-up of results (including what didn't work)

## Notes on honest evaluation
- Always compare against buy & hold of the same asset **and** SPY.
- Tune parameters on one period, then evaluate on a later period you didn't touch.
- Paper fills are optimistic (no market impact). Treat results as an upper bound.
- This is an educational project, not investment advice.
