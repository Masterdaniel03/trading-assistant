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

  Alpaca (503 stocks, 2016 on) ──► modeldata.py ──► features.py ──┐
  SEC EDGAR (10-K/10-Q facts) ──► fundamentals.py (point in time) ─┴► rankmodel.py ──► scores
                                                                     (walk-forward)     │
                                         reports/model_test_report.md ◄─────────────────┴──► league rank bots

  SQLite ──► dashboard_data.py ──► dashboard.py (Streamlit: charts, controls, run history)

  daily.py ──► alerts.py ──► Windows notification / phone (ntfy, Discord) / logs/alerts.log
                                                    
  manual Fidelity trades ──► `log-trade` ──► trades table (followed_ai flag)

  Alpaca news ──► news.py ──► research.py ──► Gemini ──► answer checked ──► signals (llm_news)
```

**Where things are**
- `main.py`: every command below. `config.yaml`: all settings. `.env`: keys (never committed).
- `tradebot/`: the code, one file per box in the diagram above, plus `db.py` (SQLite tables), `daily.py` (the scheduled run), `manifest.py` (each test report's run record), `runlog.py`, `retry.py` and `account_sync.py`.
- `universes/`: the no-hindsight 503-stock list and its SEC numbers. `scripts/`: building that list, installing the scheduled task, and checking what the stop-loss and costs take from the rank bots.
- `tests/`: `pytest`. `data/`, `logs/` and most of `reports/` are made by running the commands and are not committed.

**Two tracks**
- **Track A (automated, paper):** the bot trades an Alpaca paper account (currently $100,000).
- **Track B (assisted, real):** `signals` prints suggestions; you place trades yourself in Fidelity and record them with `log-trade --followed-ai yes|no`. (Fidelity has no retail trading API, and automating its website breaks its terms of service, so the human stays in the loop.)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your Alpaca PAPER keys, a Gemini key for `research`, and SEC_USER_AGENT
python main.py init
python main.py fetch
```

## Commands

| Command | What it does |
|---|---|
| `python main.py init` | Create the database tables (safe to run again) |
| `python main.py daily` | The whole day in one go, as the scheduled task runs it (see [Daily routine](#daily-routine-automatic)) |
| `python main.py fetch [SYMBOLS...] [--full] [--source alpaca]` | Download/refresh daily prices (resumes from the last stored date; refetches full history automatically after a split/dividend re-adjustment). `--source alpaca` gets them from Alpaca instead of Yahoo, for stocks Yahoo no longer serves, such as delisted ones |
| `python main.py quote [SYMBOLS...]` | Real-time bid/ask from Alpaca (free IEX feed for stocks; crypto needs no keys) |
| `python main.py backtest SPY sma_crossover` | Backtest one strategy vs buy & hold, save chart to `reports/` |
| `python main.py backtest-all` | Every watchlist symbol × strategy → `reports/backtest_summary.csv` |
| `python main.py walk-forward [SYMBOLS...] [--strategy NAME] [--verbose]` | Out-of-sample test: picks parameters on each training window, trades them on the next window → `reports/walk_forward_summary.csv` |
| `python main.py signals` | Today's target positions (also logged to the DB) |
| `python main.py research [SYMBOLS...] [--refresh] [--show-prompt]` | An LLM reads the last few days of news and writes a stance with reasons (advisory; see below). `--show-prompt` prints what would be sent and sends nothing |
| `python main.py research-score` | Average return after bullish / neutral / bearish notes, once they are old enough to score, both plain and relative to SPY over the same days |
| `python main.py news-archive` | Download historical news for the news test (resumable; about 10 minutes the first time) |
| `python main.py news-test` | Walk-forward test: does past news predict the next session? → `reports/news_test_report.md` (see below) |
| `python main.py paper` | Dry run: shows orders the bot would send |
| `python main.py paper --execute` | Sends those orders to Alpaca paper. Every run, dry or not, is appended to `logs/paper_trading.log`; the next run records what each earlier order filled at |
| `python main.py model-fetch [--full]` | Download or refresh daily prices for the ranking model's 503-stock universe from Alpaca, into their own database |
| `python main.py fundamentals-fetch [--full]` | Download the company filings (SEC EDGAR) for the model's companies, about 6 minutes the first time. Later runs download again only the companies with a 10-K or 10-Q not stored yet |
| `python main.py model-score [--rebuild]` | Score every stock for each day that has no score yet (the daily step) |
| `python main.py model-test [--rebuild]` | Walk-forward test of the ranking model → `reports/model_test_report.md` (see below). `--rebuild` deletes stored scores and saved models first; needed after changing `model` settings |
| `python main.py league` | Replay every league bot on the stored prices and print the leaderboard (see below) |
| `python scripts/league_attribution.py [--end DATE]` | Replay each rank bot as set, without its stop-loss, and without stop-loss or costs, against `hold-sp500` (research only; nothing is stored). See [Why the rank bots trail](#why-the-rank-bots-trail-hold-sp500) |
| `python main.py pause` / `resume` | Stop the paper bot from buying, or let it buy again. Sells and stop-loss exits always go through |
| `python main.py alerts` / `alert-test` | List recent alerts, or send a test alert through every channel that is switched on |
| `python main.py log-trade buy AAPL 2 228.50 --followed-ai yes --notes "..."` | Record a manual Fidelity trade (the dashboard has a form for this too) |
| `python main.py dashboard` | Open the dashboard in your browser (see below) |
| `python main.py report` | Trade counts, % of manual trades that followed the AI, latest equity |
| `pytest` | Run tests (indicators, signals, look-ahead, costs, risk limits, paper runs, research checks, company filings' dating, run records, DB) |

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

`research-score` reports each note's return two ways: plain, and minus SPY's return over the same days. The watchlist moves with the market, so the plain return mostly measures the market: in the first week of notes (November 2026) SPY fell, and bullish, neutral and bearish notes were all followed by losses. Only the return relative to SPY says whether a note knew something about the stock. Notes on the same symbol a few days apart cover mostly the same days, so they are not independent evidence.

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
The first step towards a strategy of your own: a model that scores every stock each day and tries to say which will do best over the next week. `ridge` and `gbm` use price and volume history only. `ridge-fund` and `gbm-fund` also read nine numbers from company filings (see [Company filings](#company-filings-phase-6b) below).

- **Universe: 503 stocks, with no hindsight.** The S&P 500 as it was listed at the end of 2015, read from an old version of Wikipedia's list (`scripts/build_universe.py`, saved in `universes/sp500_2016.txt`). Alpaca follows each 2015 ticker through later renames and up to the day the company was bought or delisted, so about 100 companies that later disappeared stay in the test. A list of today's members would only contain the survivors.
- **Inputs: 25 numbers per stock per day** (`tradebot/features.py`): trend (returns over a day to a year, distance from the high, moving-average gaps, RSI), risk (volatility, range), candle shape (where the close sat in the day's range, body and shadow sizes, overnight against intraday return) and volume and money flow (volume surge, money flow index, Chaikin money flow, on-balance volume, liquidity). Each is replaced by its rank among that day's stocks.
- **Two kinds of model**, a straight-line one (`ridge`) and boosted trees (`gbm`), with settings fixed in `config.yaml` before the first test run. Each also has a `-fund` twin with identical settings that also reads company filings (Phase 6b).
- **Walk-forward.** The scores for each year come from a model trained only on days whose outcome was known before that year began. A decision uses prices up to the close; its outcome is measured from the next open. The report prints a point-in-time audit and stops with an error if a model could have seen its own test year.
- **Scores are written once.** A stored score is never recomputed, so what the league's rank bots saw stays put.
- **Data breaks.** A close more than 3 times the previous one, or less than a third of it, is treated as a break in the data (a bankrupt company's new shares taking over the ticker, or an unadjusted split or spin-off), and the stock's history ends the day before. The report lists every series cut this way.

**First result (2026-11-14; scored out of sample from 2019 to 2026, 1,973 days):** both models rank stocks a little better than chance, but not by enough to rule out luck. Rank IC is +0.011 for `ridge` (t = 1.25) and +0.013 for `gbm` (t = 1.69); `gbm` was positive in all eight years. Holding the top 25 beat holding every stock by +0.6% a year (`ridge`) and +4.9% a year (`gbm`) after costs, with t-statistics of 0.1 and 1.0. None of the 25 inputs clears |t| = 2 on its own. In the league's fuller simulation, which adds the stop-loss and position limits, both rank bots trail `hold-sp500` over the same years (+13.2% and +14.9% a year against +15.9%), mostly because of the stop-loss (see [Why the rank bots trail](#why-the-rank-bots-trail-hold-sp500)). There is no usable edge yet. The rank bots' live record will show whether the small positive ranking skill is real.

A first run on uncleaned prices gave different dollar returns (`ridge` +4.2%, `gbm` 0.0%), because a dozen impossible price jumps were being counted as returns. The rank IC barely moved, since it only looks at the order of the stocks. That is why the data-break rule exists, and why the report lists what it cut.

## Company filings (Phase 6b)
The price-only model found no usable edge, so the next inputs come from what companies report: nine numbers from their 10-K and 10-Q filings, taken from SEC EDGAR's XBRL "company facts" (`tradebot/fundamentals.py`).

**Which company.** Each ticker starts with the SEC number (CIK) listed for it on the same end-2015 Wikipedia page as the universe (`universes/sp500_2016_ciks.csv`). 21 tickers later moved to a new SEC number, and their filings would otherwise stop: Google became Alphabet, Medtronic moved to Ireland, Disney and Cigna formed new holding companies. Some old numbers kept filing as subsidiaries. Alpaca's price series for DOW follows DowDuPont and today's DuPont, not today's Dow Inc. (daily return correlation 0.995 against 0.53). So `universes/sp500_2016_cik_successors.csv` hands each ticker to the new number from the date of the change. Each row was found by listing the tickers whose prices outlived their CIK's filings, or whose CIK stopped reporting a public float, and was checked against SEC's records.

| Group | Input | What it measures |
|---|---|---|
| value | `earnings_yield` | net income over the last four quarters, per dollar of market value |
| value | `book_to_market` | shareholders' equity per dollar of market value |
| value | `sales_to_price` | revenue over the last four quarters, per dollar of market value |
| quality | `roa` | net income over the last four quarters, per dollar of assets |
| quality | `accruals` | net income minus operating cash flow, per dollar of assets |
| quality | `equity_to_assets` | equity as a share of assets (low means more debt) |
| growth | `asset_growth` | growth of total assets over a year |
| growth | `sales_growth` | revenue over the last four quarters against the four before |
| growth | `earnings_change` | change in four-quarter net income, per dollar of assets |

Each is a long-published measure, and the list was fixed before the first test run.

**Point in time.** EDGAR lists every filing that reported each number, with its filing date, so the model sees only what was public:
- A filing dated D can come out after that day's close, so it is first used for the decision after the close of the next trading day.
- When a later filing restates a number, the old value stays in force until the restating filing's date.
- Companies move the same line between XBRL tags over the years (revenue moved to a new tag with the 2018 accounting change). The choice between tags is made among the tags already filed by that day.
- A number for a period that ended more than a year earlier counts as unknown. A company that stopped filing (bought, or moved to a new SEC number that the successor list misses) does not carry old figures forward.
- Four-quarter totals are built from what each filing reports: the year so far, plus the last full year, minus the same part of that year. Checked by hand against Apple's filings: four-quarter revenue of $259.034 billion to June 2019.

**Market value without look-ahead.** Dividing a reported per-share number by today's split-adjusted price would quietly use splits that came later. Instead, market value is the public float each 10-K reports (as of mid-year), carried forward by the stock's own return since that date. The float leaves out insiders' shares, so founder-controlled companies look smaller than they are. Filed numbers can be wrong. Waters reported its float a thousand times too large in two 10-Ks, and Garmin, Host Hotels, M&T, Zimmer Biomet, Mattel and Electronic Arts made the same kind of mistake in two or three years running. A float below 0.002 or above 500 times the total assets in the same 10-K is held back as a scale error, and the last accepted float stays in use for up to two years. Real values run from about 0.01 (a bank) to about 30 (Nvidia in 2024); every error found was off by a factor of 1,000 or more. Each float is judged against its own filing, so a repeated mistake cannot confirm itself. 25 floats are held back this way.

**A controlled comparison.** `ridge-fund` and `gbm-fund` have exactly the settings of `ridge` and `gbm`, so the filing inputs are the only difference. The model test report prints each pair's difference in daily rank IC with its t-statistic. The price-only models keep their stored scores and saved models, so their live record is unaffected.

**Result (2026-11-18; out of sample from 2019, 1,976 days): the filings added nothing measurable.** Each day's rank IC of `ridge-fund` minus `ridge` averages +0.0001 (t = 0.02); `gbm-fund` minus `gbm`, +0.0012 (t = 0.29). On their own, `gbm-fund` ranks at +0.014 (t = 1.72) and `ridge-fund` at +0.011 (t = 1.23), the same as their price-only twins. Their top 25 after costs beat holding every stock by +1.1% and -0.7% a year (t = 0.23 and -0.16). None of the nine inputs clears |t| = 2 by itself. Asset growth comes closest, at t = 2.00, and with the opposite sign to the published effect (fast-growing companies did slightly *better* over the next week). With 34 inputs tested, one near the line is what luck alone produces. In the league's fuller simulation, `rank-ridge-fund` and `rank-gbm-fund` made +6.5% and +9.5% a year from 2019, against +15.9% for `hold-sp500`.

The likely reason is the horizon, not the data. Value, quality and growth effects were found over months to years. These numbers change four times a year, while the model is asked about the next five days, a horizon fixed before Phase 6a. Testing a longer horizon would be a new test with its own settings written down first. Re-running this one with a different horizon until something works would be the overfitting this project avoids.

`fundamentals-fetch` runs every day after `model-fetch`. Every 10-K and 10-Q reports total assets, so it asks SEC for each company's total assets over the last three quarters, with the filing each number came from. It downloads again only the companies with a filing it has not seen. This uses only data.sec.gov. EDGAR's filing index on www.sec.gov refuses clients that give no e-mail address. SEC asks automated clients to identify themselves: put your name and e-mail in `.env` as `SEC_USER_AGENT`.

**Gaps.** Coca-Cola Europacific Partners (CCE's successor) files under IFRS, not US GAAP, so it has no data. Yahoo became Altaba, an investment fund that filed no 10-K. In November 2026, SEC's company-facts files held no 10-Q from the second or third quarter of 2026 for about ten companies, including Citigroup, NextEra and Southern. Their numbers drop out once they are a year old. The model test report lists the share of stock-days that have each input.

## Bot league
One paper account can only follow one strategy on a handful of stocks, which produces evidence slowly. The league runs several simulated bots side by side, each with its own strategy and list of symbols (`league.bots` and `universes` in `config.yaml`): eight rule bots across 52 symbols, plus eight rank bots and their baseline on the ranking model's 503 stocks.

- **Same rules as the real bot.** Each bot decides after the close and trades at the next open, with the same position cap, order planner, stop-loss, cooldown and trading cost. Only the fills are simulated, from stored prices, so no extra broker accounts are needed.
- **Live and history are kept apart.** The day a bot's settings are first seen is recorded. Days after it are the bot's **live** record: nothing was chosen knowing those prices. Days before it are **history**, an ordinary backtest. Changing a bot's line in the config restarts its live record.
- **History flatters every bot.** The strategies were picked knowing the past, and the stock lists hold today's large companies, which are the ones that did well. The `hold-large30` bot (buy and hold the same 30 stocks) is the baseline to beat, not SPY.
- **Rank bots** (`rank-ridge`, `rank-gbm`, and `rank-ridge-fund`, `rank-gbm-fund` for the models that also read company filings) hold the 25 stocks their model scores highest and re-pick every 5 trading days, under the same position cap and stop-loss. Their history starts in 2019, the first year the model was scored without hindsight, and it is a walk-forward result, not an ordinary backtest. `hold-sp500` holds all 503 stocks and is the baseline they have to beat. A holding whose stock stops trading is paid out at its last price after five days.
- **Rank bots with a stop sized to their positions** (`rank-ridge-sized`, `rank-gbm-sized`, `rank-ridge-fund-sized`, `rank-gbm-fund-sized`, added 2026-11-18) are copies of the four rank bots with a 37.5% stop instead of 7.5%. Everything else is the same. See below for why.
- Each run replays every bot from its start date, so a missed day changes nothing. `python main.py league` takes about a minute and writes `logs/league.log`.

Picking the bot with the best history and trusting it is the overfitting this project tries to avoid. Use the live column, and give it months.

### Why the rank bots trail hold-sp500
`hold-sp500` has no stop-loss, while the rank bots do. `scripts/league_attribution.py` replays each rank bot with the league's own simulator three ways: as set, without the stop-loss, and without the stop-loss or trading costs. Over the leaderboard's history, from 2019 to 2026-11-13 (`--end 2026-11-13`), the yearly returns were as follows, with the Sharpe ratio in brackets:

| Bot | 7.5% stop (as set) | 37.5% stop (`-sized` copy) | No stop | No stop, no costs | Return left after beta, no stop |
|---|---|---|---|---|---|
| `rank-ridge` | +13.2% (0.68) | +14.9% (0.69) | +16.4% (0.73) | +20.0% (0.85) | +1.2% a year (t = 0.26) |
| `rank-gbm` | +14.9% (0.74) | +18.0% (0.78) | +19.8% (0.80) | +23.5% (0.91) | +3.3% a year (t = 0.71) |
| `rank-ridge-fund` | +7.0% (0.43) | +13.6% (0.66) | +14.3% (0.68) | +17.3% (0.79) | -0.1% a year (t = -0.02) |
| `rank-gbm-fund` | +9.9% (0.55) | +15.4% (0.72) | +16.3% (0.72) | +19.1% (0.81) | +1.2% a year (t = 0.26) |
| `hold-sp500` | +15.9% (0.82), no stop | | | | |

- **The 7.5% stop is the main reason they trail.** It was hit 824 to 892 times per bot and cost 3 to 7 points a year. It lowered every bot's Sharpe ratio, so it did not simply trade return for safety.
- **Why it fires so often.** The stop is the 1.5% risk budget divided by the 20% position cap, which fits the paper bot's five symbols. A rank bot holds 25 stocks at 4% each, so the same rule gives 1.5% / 4% = 37.5%. The `-sized` copies use that, by capping positions at 4% (what they hold anyway), and were stopped out only 10 to 18 times.
- **What the 7.5% stop does buy.** A smaller worst drop for the price-only models: −38.5% for `rank-gbm` against −44.5% without a stop, and −41.1% for `hold-sp500`. For the models that read company filings it made no difference. The 37.5% stop gave each model its deepest drop (−47.0% for `rank-gbm-sized`): in the March 2020 crash, where that drop happened, its stops fired near the bottom.
- **Without the stop there is still no edge.** Once costs are paid, no rank bot's Sharpe ratio reaches `hold-sp500`'s 0.82. Without the stop they move a little more than the market (beta 0.96 to 1.09: how much a bot moved when `hold-sp500` moved 1%). The yearly return left once that is taken out is −0.1% to +3.3%, with t at most 0.71, which could easily be luck. Even before costs it is at most +6.4% (t = 1.37).
- **Trading costs** take about 3 to 4 points a year at 5 bps per trade, because about half the holdings change each week.
- **Not the cause:** stocks that stopped trading. The rank bots held only 0 to 3 of them when they were delisted.
- **`rank-gbm` against `rank-gbm-fund`.** Their gap shrinks without the stop (+19.8% against +16.3%). In the model test, the gap between their top 25s was −3.9% a year with t = −1.01, which could be luck.

Choosing a stop after seeing this history would be hindsight. So the original rank bots keep their 7.5% stop and their live record, and the `-sized` copies were added with their settings written into `config.yaml` first. Only their live record, from 2026-11-19 on, counts as evidence.

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
`python main.py daily` runs the whole day in one go: `fetch`, then `research`, then `paper --execute`, then `model-fetch`, `fundamentals-fetch`, `model-score` and `league`. Trading comes first, so a problem in the model steps can never delay an order. On the last trading day of each week it also updates the news archive and re-runs `walk-forward`, `news-test` and `model-test`. It skips weekends and market holidays (using Alpaca's calendar). Each step runs separately, so one failure doesn't stop the rest. Everything is appended to `logs/daily_run.log`, each run is stored for the dashboard's run history, and a problem raises an alert.

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
- [ ] Phase 3b: earnings dates and filings as extra research inputs; run research after events (earnings, 8-K filings, unusually large price moves or news volume) instead of on the same list every day, so notes overlap less and `research-score` gets more independent evidence; decide from the scores whether the notes should influence the bot
- [x] Phase 3c: point-in-time news test (walk-forward, headlines → next session)
- [x] Phase 4: Streamlit dashboard: bot vs. manual vs. SPY, Sharpe, drawdown, AI-follow rate
- [x] Phase 4b: failure alerts, pause switch, bot league (8 simulated bots, 52 symbols), dashboard controls and run history
- [x] Phase 6a: ranking model on a no-hindsight 503-stock universe (25 price, candle and money-flow inputs; walk-forward test; rank bots in the league)
- [x] Phase 6b: fundamentals from SEC filings, dated by when each number became public, as extra model inputs (9 inputs; `ridge-fund` and `gbm-fund`, each compared with its price-only twin)
- [x] What the stop-loss and costs take from the rank bots (`scripts/league_attribution.py`); rank bot copies with a stop sized to their positions
- [x] A run record (commit, settings and stock-list fingerprints) on every test report
- [ ] Phase 5: write-up of results (including what didn't work). Waits for a few months of live league and research-note records
- [ ] Next test to pre-register: the same filing inputs against a one-to-three-month horizon, the scale at which value and quality effects were found

## Run records
A test's settings are written into `config.yaml` before its first run and never tuned against its result. Each test report records what it ran with, so that can be checked later. `model-test` and `news-test` end their report with a **Run record** section, and `model-test`, `news-test` and `walk-forward` each save a JSON file next to their report (for example `reports/model_test_report.run.json`). The record holds:
- the git commit, and whether tracked files (`config.yaml` included) had uncommitted changes;
- a fingerprint of the test's own settings in `config.yaml`, which changes only when those settings change;
- a fingerprint of the whole `config.yaml`, of the stock list, and of each data file the test reads (such as the SEC numbers);
- when the test started, and the Python and package versions.

A fingerprint is a SHA-256 hash: the same input always gives the same one, and any change gives a different one. Two reports with the same settings fingerprint ran with the same settings. The record never changes a result (`tradebot/manifest.py`).

## Notes on honest evaluation
- Always compare against buy & hold of the same asset **and** SPY.
- Tune parameters on one period, then evaluate on a later period you didn't touch.
- Paper fills are optimistic (no market impact). Treat results as an upper bound.
- This is an educational project, not investment advice.
