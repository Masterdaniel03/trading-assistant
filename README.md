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
```

**Two tracks**
- **Track A (automated, paper):** the bot trades an Alpaca paper account sized to $4,000.
- **Track B (assisted, real):** `signals` prints suggestions; you place trades yourself in Fidelity and record them with `log-trade --followed-ai yes|no`. (Fidelity has no retail trading API, and automating its website breaks its terms of service, so the human stays in the loop.)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your Alpaca PAPER keys
python main.py init
python main.py fetch
```

## Commands

| Command | What it does |
|---|---|
| `python main.py fetch [SYMBOLS...] [--full]` | Download/refresh daily prices (resumes from the last stored date; refetches full history automatically after a split/dividend re-adjustment) |
| `python main.py quote [SYMBOLS...]` | Real-time bid/ask from Alpaca (free IEX feed for stocks; crypto needs no keys) |
| `python main.py backtest SPY sma_crossover` | Backtest one strategy vs buy & hold, save chart to `reports/` |
| `python main.py backtest-all` | Every watchlist symbol × strategy → `reports/backtest_summary.csv` |
| `python main.py walk-forward [SYMBOLS...] [--strategy NAME] [--verbose]` | Out-of-sample test: picks parameters on each training window, trades them on the next window → `reports/walk_forward_summary.csv` |
| `python main.py signals` | Today's target positions (also logged to the DB) |
| `python main.py paper` | Dry run: shows orders the bot would send |
| `python main.py paper --execute` | Sends those orders to Alpaca paper |
| `python main.py log-trade buy AAPL 2 228.50 --followed-ai yes --notes "..."` | Record a manual Fidelity trade |
| `python main.py report` | Trade counts, % of manual trades that followed the AI, latest equity |
| `pytest` | Run tests (indicators, signals, look-ahead, costs, risk limits, DB) |

Settings (watchlist, strategy parameters, risk limits) are in `config.yaml`.

## Safety rails
- Paper trading by default. Live trading raises an error unless `risk.allow_live: true` **and** `ALPACA_PAPER=false`.
- Max 20% of equity in any one symbol, no margin, and no new orders after a 3% daily loss.
- `paper` refuses to run if any bot symbol's prices are missing or older than `risk.max_data_age_days` (default 4).
- Positions in the paper account that aren't in `paper_bot.symbols` are closed (`paper_bot.close_unlisted`).
- `paper` is a dry run unless you pass `--execute`.
- Never commit `.env`. Never give any key withdrawal permissions.

## Daily routine (once it runs)
After the market closes (1 pm Pacific): `fetch` → `signals` → `paper --execute`. Log any Fidelity trades the same day. A cron job or GitHub Actions schedule can automate the first three.

## Roadmap
- [x] Phase 1: data pipeline, SQLite logging, backtester, 2 baseline strategies, Alpaca paper bot, manual trade log
- [x] Phase 2a: walk-forward / out-of-sample testing; more strategies (momentum, volatility targeting)
- [ ] Phase 2b: crypto on Alpaca (`BTC/USD`)
- [ ] Phase 3: LLM research layer (summarize news/earnings → a signal with written reasoning, stored in `signals.reason`)
- [ ] Phase 4: Streamlit dashboard: bot vs. manual vs. SPY, Sharpe, drawdown, AI-follow rate
- [ ] Phase 5: write-up of results (including what didn't work)

## Notes on honest evaluation
- Always compare against buy & hold of the same asset **and** SPY.
- Tune parameters on one period, then evaluate on a later period you didn't touch.
- Paper fills are optimistic (no market impact). Treat results as an upper bound.
- This is an educational project, not investment advice.
