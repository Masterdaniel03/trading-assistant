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
                                                    
  SQLite ──► league.py ──► simulated bots (same rules, own strategy and symbols each)

  Alpaca (503 stocks, 2016 on) ──► modeldata.py ──► features.py ──► rankmodel.py ──► scores
                                                     (walk-forward)        │
                                         reports/model_test_report.md ◄────┴────► league rank bots

  SQLite ──► dashboard_data.py ──► dashboard.py (Streamlit: charts, controls, run history)

  daily.py ──► alerts.py ──► Windows notification / phone (ntfy, Discord) / logs/alerts.log
                                                    
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
| `python main.py model-fetch [--full]` | Download or refresh daily prices for the ranking model's 503-stock universe from Alpaca, into their own database |
| `python main.py model-score [--rebuild]` | Score every stock for each day that has no score yet (the daily step) |
| `python main.py model-test [--rebuild]` | Walk-forward test of the ranking model → `reports/model_test_report.md` (see below). `--rebuild` deletes stored scores and saved models first; needed after changing `model` settings |
| `python main.py league` | Replay every league bot on the stored prices and print the leaderboard (see below) |
| `python main.py pause` / `resume` | Stop the paper bot from buying, or let it buy again. Sells and stop-loss exits always go through |
| `python main.py alerts` / `alert-test` | List recent alerts, or send a test alert through every channel that is switched on |
| `python main.py log-trade buy AAPL 2 228.50 --followed-ai yes --notes "..."` | Record a manual Fidelity trade (the dashboard has a form for this too) |
| `python main.py dashboard` | Open the dashboard in your browser (see below) |
| `python main.py report` | Trade counts, % of manual trades that followed the AI, latest equity |
| `pytest` | Run tests (indicators, signals, look-ahead, costs, risk limits, paper runs, research checks, DB) |

Settings (watchlist, strategy parameters, risk limits, league bots, alerts) are in `config.yaml`.
With no symbols given, `fetch` downloads the watchlist plus every symbol a league bot trades.

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

## Ranking model
The first step towards a strategy of your own: a model that scores every stock each day and tries to say which will do best over the next week. It uses price and volume history only for now (fundamentals from company filings are the planned next input).

- **Universe: 503 stocks, with no hindsight.** The S&P 500 as it was listed at the end of 2015, read from an old version of Wikipedia's list (`scripts/build_universe.py`, saved in `universes/sp500_2016.txt`). Alpaca follows each 2015 ticker through later renames and up to the day the company was bought or delisted, so about 100 companies that later disappeared stay in the test. A list of today's members would only contain the survivors.
- **Inputs: 25 numbers per stock per day** (`tradebot/features.py`): trend (returns over a day to a year, distance from the high, moving-average gaps, RSI), risk (volatility, range), candle shape (where the close sat in the day's range, body and shadow sizes, overnight against intraday return) and volume and money flow (volume surge, money flow index, Chaikin money flow, on-balance volume, liquidity). Each is replaced by its rank among that day's stocks.
- **Two models**, a straight-line one (`ridge`) and boosted trees (`gbm`), with settings fixed in `config.yaml` before the first test run.
- **Walk-forward.** The scores for each year come from a model trained only on days whose outcome was known before that year began. A decision uses prices up to the close; its outcome is measured from the next open. The report prints a point-in-time audit and stops with an error if a model could have seen its own test year.
- **Scores are written once.** A stored score is never recomputed, so what the league's rank bots saw stays put.
- **Data breaks.** A close more than 3 times the previous one, or less than a third of it, is treated as a break in the data (a bankrupt company's new shares taking over the ticker, or an unadjusted split or spin-off), and the stock's history ends the day before. The report lists every series cut this way.

**First result (2026-11-14; scored out of sample from 2019 to 2026, 1,973 days):** both models rank stocks a little better than chance, but not by enough to rule out luck. Rank IC is +0.011 for `ridge` (t = 1.25) and +0.013 for `gbm` (t = 1.69); `gbm` was positive in all eight years. Holding the top 25 beat holding every stock by +0.6% a year (`ridge`) and +4.9% a year (`gbm`) after costs, with t-statistics of 0.1 and 1.0. None of the 25 inputs clears |t| = 2 on its own. In the league's fuller simulation, which adds the stop-loss and position limits, both rank bots trail `hold-sp500` over the same years (+13.2% and +14.9% a year against +15.9%). There is no usable edge yet. The rank bots' live record will show whether the small positive ranking skill is real.

A first run on uncleaned prices gave different dollar returns (`ridge` +4.2%, `gbm` 0.0%), because a dozen impossible price jumps were being counted as returns. The rank IC barely moved, since it only looks at the order of the stocks. That is why the data-break rule exists, and why the report lists what it cut.

## Bot league
One paper account can only follow one strategy on a handful of stocks, which produces evidence slowly. The league runs several simulated bots side by side, each with its own strategy and list of symbols (`league.bots` and `universes` in `config.yaml`): eight rule bots across 52 symbols, plus two rank bots and their baseline on the ranking model's 503 stocks.

- **Same rules as the real bot.** Each bot decides after the close and trades at the next open, with the same position cap, order planner, stop-loss, cooldown and trading cost. Only the fills are simulated, from stored prices, so no extra broker accounts are needed.
- **Live and history are kept apart.** The day a bot's settings are first seen is recorded. Days after it are the bot's **live** record: nothing was chosen knowing those prices. Days before it are **history**, an ordinary backtest. Changing a bot's line in the config restarts its live record.
- **History flatters every bot.** The strategies were picked knowing the past, and the stock lists hold today's large companies, which are the ones that did well. The `hold-large30` bot (buy and hold the same 30 stocks) is the baseline to beat, not SPY.
- **Rank bots** (`rank-ridge`, `rank-gbm`) hold the 25 stocks their model scores highest and re-pick every 5 trading days, under the same position cap and stop-loss. Their history starts in 2019, the first year the model was scored without hindsight, and it is a walk-forward result, not an ordinary backtest. `hold-sp500` holds all 503 stocks and is the baseline they have to beat. A holding whose stock stops trading is paid out at its last price after five days.
- Each run replays every bot from its start date, so a missed day changes nothing. `python main.py league` takes about a minute and writes `logs/league.log`.

Picking the bot with the best history and trusting it is the overfitting this project tries to avoid. Use the live column, and give it months.

## Alerts
The system tells you when something needs attention:

| Alert | When |
|---|---|
| Daily run: problem in ... | A step of the scheduled run failed. The message names the step and its last lines of output |
| Daily run could not start | The exchange calendar could not be read, so nothing ran |
| No daily run finished on ... | The previous trading day has no finished run (the PC was off or signed out). Reported by the next run |
| Stop-loss: selling ... | A position hit its stop and the sell order was sent |
| Paper bot did not trade / Daily loss limit hit | A guard blocked trading for the day |
| Daily run: all ok | After each good run (`alerts.daily_summary`), so that silence means trouble |

Every alert is stored in the database, shown at the top of the dashboard until marked as read, and appended to `logs/alerts.log`. It is also sent as a Windows notification (`alerts.desktop`). To get alerts on a phone, install the free **ntfy** app, subscribe to a long random topic name, put the same name in `.env` as `NTFY_TOPIC`, and run `python main.py alert-test`. A Discord webhook (`DISCORD_WEBHOOK_URL`) works too. A channel that fails is recorded on the alert and never stops a run.

An alert cannot be sent while the PC is off. The daily "all ok" message covers that: if it does not arrive, look.

## Dashboard
`python main.py dashboard` opens a local page with:
- **Controls (sidebar):** pause or resume buying; fetch prices; update the league; preview the bot's orders; and, after ticking a box, send the orders or start the full daily run. Each button runs the same `python main.py ...` command you could type, so the bot's limits, checks and logs apply.
- **Unread alerts** at the top of the page, with a button to mark them as read.
- **Bot league:** the leaderboard (live record first, then backtest history, each next to SPY over the bot's own days), a chart of any bot against SPY and the other bots over the live period, 1 year, 3 years or all history, and that bot's holdings and latest trades. The chart starts where the chosen bot starts.
- **Runs and alerts:** every daily run with each step's result and output, and the list of alerts with where each was sent.
- **Headline numbers:** account value, return since the first trade, SPY over the same days, the worst drop from a peak, and the Sharpe ratio (shown only after 20 trading days, because before that it is noise).
- **Account value vs SPY buy and hold:** both lines start from the same dollar amount on the day of the first order, so they share one axis.
- **Holdings and orders:** each holding's share of the account against its cap, its return since entry and how far it is from its stop-loss; every order with its status and fill price.
- **Signals and research:** the bot's latest signal next to the latest research note for each symbol, whether they agree, and how past notes turned out.
- **Your own trades:** a form to log a trade, and manual trades valued at the latest close, next to what the same dollars would have made in SPY, split by whether you followed the AI.
- **Backtests:** the walk-forward summary, the ranking model's test report and the news test report.

Everything shown comes from the local database. The broker-side numbers (balance, holdings, order results, daily account value) are refreshed by every `python main.py paper` run and by the page's **Refresh from Alpaca** button, which sends no orders. Nothing is sent to the broker unless you tick the box under **Send orders**.

## Safety rails
- Paper trading by default. Live trading raises an error unless `risk.allow_live: true` **and** `ALPACA_PAPER=false`.
- Max 20% of equity in any one symbol, no margin, and no new buys after a 3% daily loss (sells that reduce risk still go through).
- Running `paper --execute` twice is safe: a symbol with an order still waiting to fill gets no second order. Sells are sent before buys.
- Each position risks at most 1.5% of equity: a 7.5% stop-loss (1.5% / 20%), checked each time `paper` runs, followed by a 20-bar cooldown. It is not a resting order, so a price gap or a long gap between runs can lose more. Stops still execute when the daily-loss or stale-data guards block buying.
- `backtest` and `walk-forward` include the stop; pass `--no-stop` to compare without it.
- If any bot symbol is missing the close of the latest finished session, `paper` sends stop-loss exits only. Which session that is comes from Alpaca's exchange calendar, so holidays and your PC's timezone don't matter.
- `fetch` stores finished daily bars only: today's stock bar is left out until 16:15 New York time, and today's crypto bar until the UTC day ends.
- Positions in the paper account that aren't in `paper_bot.symbols` are closed (`paper_bot.close_unlisted`).
- `paper` is a dry run unless you pass `--execute`.
- `python main.py pause` (or the dashboard's Pause button) stops all buying until `resume`. Sells and stop-loss exits still go through.
- The league bots are simulations. They never send an order to any broker.
- Never commit `.env`. Never give any key withdrawal permissions.

## Daily routine (automatic)
`python main.py daily` runs the whole day in one go: `fetch`, then `research`, then `paper --execute`, then `model-fetch`, `model-score` and `league`. Trading comes first, so a problem in the model steps can never delay an order. On the last trading day of each week it also updates the news archive and re-runs `walk-forward`, `news-test` and `model-test`. It skips weekends and market holidays (using Alpaca's calendar). Each step runs separately, so one failure doesn't stop the rest. Everything is appended to `logs/daily_run.log`, each run is stored for the dashboard's run history, and a problem raises an alert.

**Why 2:30 pm when the market opens at 6:30 am (Pacific)?** The strategies decide from daily closing prices, and the close is at 1:00 pm Pacific. The run happens after that, once the day's prices are final, and its orders wait at the broker and fill by themselves at the next open, 6:30 am. Running at 6:30 am instead would use the same closing prices, only 17 hours later.

To run it automatically on weekdays at 2:30 pm (after the day's prices are final):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1          # install
powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1 -Remove  # remove
```
The PC has to be on (or asleep, which Windows can wake it from). If it was off at 2:30 pm, the run starts as soon as it's back on. Orders fill at the next market open by themselves; nobody needs to be awake for that. Log any Fidelity trades yourself with `log-trade`.

## Roadmap
- [x] Phase 1: data pipeline, SQLite logging, backtester, 2 baseline strategies, Alpaca paper bot, manual trade log
- [x] Phase 2a: walk-forward / out-of-sample testing; more strategies (momentum, volatility targeting)
- [ ] Phase 2b: crypto on Alpaca (`BTC/USD`)
- [x] Phase 3a: LLM research layer (recent news → a checked stance with written reasoning, stored in `signals.reason`; forward scoring)
- [ ] Phase 3b: earnings dates and filings as extra research inputs; decide from the scores whether the notes should influence the bot
- [x] Phase 3c: point-in-time news test (walk-forward, headlines → next session)
- [x] Phase 4: Streamlit dashboard: bot vs. manual vs. SPY, Sharpe, drawdown, AI-follow rate
- [x] Phase 4b: failure alerts, pause switch, bot league (8 simulated bots, 52 symbols), dashboard controls and run history
- [x] Phase 6a: ranking model on a no-hindsight 503-stock universe (25 price, candle and money-flow inputs; walk-forward test; rank bots in the league)
- [ ] Phase 6b: fundamentals from SEC filings, dated by when each number became public, as extra model inputs
- [ ] Phase 5: write-up of results (including what didn't work)

## Notes on honest evaluation
- Always compare against buy & hold of the same asset **and** SPY.
- Tune parameters on one period, then evaluate on a later period you didn't touch.
- Paper fills are optimistic (no market impact). Treat results as an upper bound.
- This is an educational project, not investment advice.
