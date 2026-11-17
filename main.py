"""Command-line entry point.

    python main.py init                      # create database
    python main.py fetch                     # download/refresh prices for watchlist
    python main.py quote [SYMBOLS...]        # real-time bid/ask from Alpaca (free IEX feed)
    python main.py backtest SPY sma_crossover
    python main.py backtest-all              # every watchlist symbol x every strategy
    python main.py walk-forward [SYMBOLS...] # out-of-sample test, parameters re-picked each fold
    python main.py signals                   # today's signals (what the AI suggests)
    python main.py research [SYMBOLS...]     # LLM reads recent news, writes a stance with reasons
    python main.py research-score            # how past research notes compare with later prices
    python main.py news-archive              # download historical news for the news test
    python main.py news-test                 # does past news predict the next session? (walk-forward)
    python main.py paper                     # dry run: show orders the bot WOULD place
    python main.py paper --execute           # actually send them to Alpaca paper
    python main.py model-fetch               # prices for the ranking model's wide universe (Alpaca)
    python main.py model-score               # score every stock for the days not scored yet
    python main.py model-test                # walk-forward test of the ranking model, with a report
    python main.py league                    # replay every league bot and print the leaderboard
    python main.py pause | resume            # stop the paper bot from buying / let it buy again
    python main.py alerts                    # recent alerts; `alert-test` sends a test one
    python main.py dashboard                 # open the web dashboard
    python main.py log-trade buy AAPL 2 228.50 --followed-ai yes --notes "SMA signal"
    python main.py report                    # summary of trades logged
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from tradebot import db
from tradebot.alerts import raise_alert
from tradebot.backtest import compare, format_metrics, plot_equity, run_backtest, with_stop_loss
from tradebot.config import ROOT, load_config, load_secrets
from tradebot.data import (AlpacaQuotes, drop_incomplete_bars, fetch_many, fetch_prices,
                           fetch_prices_alpaca, history_was_readjusted, incremental_start,
                           periods_per_year)
from tradebot.rebalance import plan_orders
from tradebot.account_sync import sync_account
from tradebot.runlog import RunLog
from tradebot.risk import (RiskViolation, check_daily_loss, check_data_fresh, check_live_allowed,
                           stop_loss_pct, triggered_stops)
from tradebot.strategies import STRATEGIES, get_strategy
from tradebot.walkforward import walk_forward

REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"


def _alert(cfg, conn, level, title, detail="", source=""):
    """Store an alert and send it to every channel that is switched on. Never raises."""
    try:
        raise_alert(conn, level, title, detail, source, cfg=cfg, log_path=LOGS / "alerts.log")
    except Exception as e:  # noqa: BLE001 - an alert must never stop the run that raised it
        print(f"Note: could not record the alert ({type(e).__name__}: {e})")


def _fetch_symbols(cfg):
    """The watchlist plus every symbol a league bot trades, without repeats."""
    from tradebot.league import league_symbols
    return list(dict.fromkeys([*cfg["watchlist"], *cfg["paper_bot"]["symbols"], *league_symbols(cfg)]))


def cmd_init(cfg, args):
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    print(f"Database ready at {cfg['db_path']}")


def cmd_fetch(cfg, args):
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    symbols = args.symbols or _fetch_symbols(cfg)
    full_start = cfg["start_date"]
    if args.source == "alpaca":
        _fetch_from_alpaca(conn, symbols, full_start)
        return
    starts = ({s: full_start for s in symbols} if args.full else
              {s: incremental_start(db.last_price_date(conn, s), full_start) for s in symbols})
    failed = []
    for sym, result in fetch_many(symbols, start=starts):
        if not isinstance(result, Exception) and starts[sym] != full_start \
                and history_was_readjusted(db.load_close(conn, sym), result):
            print(f"{sym:<8} history re-adjusted (split/dividend); refetching from {full_start}")
            try:
                result = fetch_prices(sym, full_start)
            except Exception as exc:  # noqa: BLE001 - reported below
                result = exc
        if isinstance(result, Exception):
            failed.append(sym)
            print(f"{sym:<8} FAILED  {type(result).__name__}: {result}")
            continue
        blank = result.attrs.get("missing_close", 0)
        final = drop_incomplete_bars(result, sym)
        n = db.upsert_prices(conn, final) if not final.empty else 0
        note = "; today's bar is not final yet, so it was left out" if len(final) < len(result) else ""
        if blank:
            note += f"; Yahoo sent {blank} row(s) with no close, left out. Fetch again later"
        print(f"{sym:<8} {n:>5} rows  (latest: {db.last_price_date(conn, sym)}{note})")
    if failed:
        sys.exit(f"Fetch failed for {', '.join(failed)} (see errors above).")


def _fetch_from_alpaca(conn, symbols, start):
    """Full history from Alpaca, for stocks Yahoo no longer has (e.g. delisted ones)."""
    secrets = load_secrets()
    if not secrets["alpaca_key"] or not secrets["alpaca_secret"]:
        sys.exit("--source alpaca needs ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")
    failed = []
    for sym in symbols:
        try:
            bars = fetch_prices_alpaca(sym, start, secrets["alpaca_key"], secrets["alpaca_secret"])
        except Exception as exc:  # noqa: BLE001 - reported, and the rest carry on
            failed.append(sym)
            print(f"{sym:<8} FAILED  {type(exc).__name__}: {exc}")
            continue
        final = drop_incomplete_bars(bars, sym)
        n = db.upsert_prices(conn, final) if not final.empty else 0
        print(f"{sym:<8} {n:>5} rows  (latest: {db.last_price_date(conn, sym)}, from Alpaca)")
    if failed:
        sys.exit(f"Fetch failed for {', '.join(failed)} (see errors above).")


def cmd_quote(cfg, args):
    secrets = load_secrets()
    try:
        quotes = AlpacaQuotes(secrets["alpaca_key"], secrets["alpaca_secret"]).latest(
            args.symbols or cfg["watchlist"])
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"{'symbol':<8} {'bid':>11} {'ask':>11} {'mid':>11} {'spread':>8}  time (UTC)        source")
    for q in quotes.values():
        spread = f"{q.spread_bps:>6.1f}bp" if q.bid > 0 and q.ask > 0 else f"{'n/a':>8}"
        print(f"{q.symbol:<8} {_price(q.bid)} {_price(q.ask)} {q.mid:>11,.2f} {spread}"
              f"  {q.timestamp:%Y-%m-%d %H:%M:%S}  {q.source}")


def _price(value: float) -> str:
    """A missing bid or ask comes back as 0; show a dash rather than $0.00."""
    return f"{value:>11,.2f}" if value > 0 else f"{'-':>11}"


def _strategy_params(cfg, name, symbol):
    return {**cfg.get("strategies", {}).get(name, {}), "periods_per_year": periods_per_year(symbol)}


def _with_stop(cfg, strategy):
    risk = cfg["risk"]
    return with_stop_loss(strategy, stop_loss_pct(risk), risk["stop_cooldown_bars"])


def _backtest_one(cfg, conn, symbol, strategy_name, plot=True, use_stop=True):
    close = db.load_close(conn, symbol)
    if close.empty:
        raise LookupError(f"No prices for {symbol}. Run: python main.py fetch {symbol}")
    bt = cfg["backtest"]
    ppy = periods_per_year(symbol)
    strat = get_strategy(strategy_name)
    if use_stop and strategy_name != "buy_and_hold":
        strat = _with_stop(cfg, strat)
    res = run_backtest(close, strat(close, **_strategy_params(cfg, strategy_name, symbol)),
                       name=strategy_name, initial_capital=bt["initial_capital"],
                       cost_bps=bt["cost_bps"], periods_per_year=ppy)
    bench = run_backtest(close, STRATEGIES["buy_and_hold"](close), name="buy_and_hold",
                         initial_capital=bt["initial_capital"], cost_bps=bt["cost_bps"],
                         periods_per_year=ppy)
    if plot:
        REPORTS.mkdir(exist_ok=True)
        path = plot_equity([res, bench], str(REPORTS / f"{symbol}_{strategy_name}.png"),
                           title=f"{symbol}: {strategy_name} vs buy & hold")
        print(f"Chart saved: {path}")
    return res, bench


def cmd_backtest(cfg, args):
    conn = db.connect(cfg["db_path"])
    try:
        res, bench = _backtest_one(cfg, conn, args.symbol, args.strategy, use_stop=not args.no_stop)
    except LookupError as e:
        sys.exit(str(e))
    print("Stop-loss: off" if args.no_stop else f"Stop-loss: {stop_loss_pct(cfg['risk']):.1%}")
    print(f"\n{args.symbol}  {res.equity.index[0].date()} -> {res.equity.index[-1].date()}")
    print(format_metrics(compare([res, bench])).to_string())


def cmd_backtest_all(cfg, args):
    conn = db.connect(cfg["db_path"])
    rows, skipped = [], []
    for sym in cfg["watchlist"]:
        if db.last_price_date(conn, sym) is None:
            skipped.append(sym)
            continue
        for name in STRATEGIES:
            if name == "buy_and_hold":
                continue
            res, bench = _backtest_one(cfg, conn, sym, name, plot=False)
            rows.append({"symbol": sym, "strategy": name,
                         **{f"strat_{k}": v for k, v in res.metrics.items() if k in ("cagr", "sharpe", "max_drawdown")},
                         **{f"bh_{k}": v for k, v in bench.metrics.items() if k in ("cagr", "sharpe", "max_drawdown")}})
    if skipped:
        print(f"Skipped (no prices, run `python main.py fetch`): {', '.join(skipped)}\n")
    if not rows:
        sys.exit("Nothing to backtest.")
    df = pd.DataFrame(rows)
    df["beats_bh_sharpe"] = df["strat_sharpe"] > df["bh_sharpe"]
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "backtest_summary.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(df.to_string(index=False))
    print(f"\nSaved {REPORTS / 'backtest_summary.csv'}")


def cmd_walk_forward(cfg, args):
    conn = db.connect(cfg["db_path"])
    wf, bt = cfg["walk_forward"], cfg["backtest"]
    names = [args.strategy] if args.strategy else list(wf["grids"])
    rows, skipped = [], []
    for sym in args.symbols or cfg["watchlist"]:
        close = db.load_close(conn, sym)
        ppy = periods_per_year(sym)
        for name in names:
            try:
                strat = get_strategy(name) if args.no_stop else _with_stop(cfg, get_strategy(name))
                r = walk_forward(close, strat, wf["grids"][name],
                                 train_bars=wf["train_years"] * ppy, test_bars=wf["test_years"] * ppy,
                                 name=name, initial_capital=bt["initial_capital"],
                                 cost_bps=bt["cost_bps"], periods_per_year=ppy)
            except ValueError as e:
                skipped.append(f"{sym}/{name}: {e}")
                continue
            rows.append({"symbol": sym, "strategy": name, "folds": len(r.folds),
                         "train_sharpe": r.mean_train_sharpe,
                         "oos_sharpe": r.oos.metrics["sharpe"], "bh_sharpe": r.benchmark.metrics["sharpe"],
                         "oos_cagr": r.oos.metrics["cagr"], "bh_cagr": r.benchmark.metrics["cagr"],
                         "oos_max_dd": r.oos.metrics["max_drawdown"],
                         "bh_max_dd": r.benchmark.metrics["max_drawdown"],
                         "folds_beat_bh": r.folds_beating_benchmark})
            if args.verbose:
                print(f"\n{sym} {name}")
                print(pd.DataFrame(r.folds).to_string(index=False))
    for s in skipped:
        print(f"Skipped {s}")
    if not rows:
        sys.exit("Nothing to test. Run: python main.py fetch")
    df = pd.DataFrame(rows)
    df["beats_bh_sharpe"] = df["oos_sharpe"] > df["bh_sharpe"]
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "walk_forward_summary.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 180):
        print(df.to_string(index=False))
    print(f"\nSaved {REPORTS / 'walk_forward_summary.csv'}")


def _current_targets(cfg, conn, strategy_name, symbols, log=True):
    strat = get_strategy(strategy_name)
    targets = {}
    for sym in symbols:
        close = db.load_close(conn, sym)
        if close.empty:
            continue
        t = float(strat(close, **_strategy_params(cfg, strategy_name, sym)).iloc[-1])
        targets[sym] = t
        if log:
            db.log_signal(conn, sym, strategy_name, t,
                          reason=f"as of {close.index[-1].date()} close={close.iloc[-1]:.2f}")
    return targets


def cmd_signals(cfg, args):
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    name = args.strategy or cfg["paper_bot"]["strategy"]
    targets = _current_targets(cfg, conn, name, cfg["watchlist"])
    print(f"Signals ({name})  1 = full position, 0 = stay out")
    for sym, t in targets.items():
        print(f"  {sym:<8} {t:.2f}")


def cmd_research(cfg, args):
    """Advisory only: notes are logged and scored, and the bot does not trade on them."""
    from tradebot import research
    from tradebot.llm import DEFAULT_MODELS, GeminiLLM
    from tradebot.news import AlpacaNews, select_items

    rc = cfg.get("research", {})
    secrets = load_secrets()
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    symbols = args.symbols or cfg["watchlist"]
    since = datetime.now(timezone.utc) - timedelta(days=rc.get("news_lookback_days", 3))
    try:
        news = AlpacaNews(secrets["alpaca_key"], secrets["alpaca_secret"])
        models = rc.get("models") or ([rc["model"]] if rc.get("model") else DEFAULT_MODELS)
        llm = None if args.show_prompt else GeminiLLM(
            secrets["gemini_key"], models, rc.get("requests_per_minute", 5))
    except RuntimeError as e:
        sys.exit(str(e))

    def recent_news(sym):
        return select_items(news.recent(sym, since), rc.get("max_news_items", 12))

    if args.show_prompt:   # show exactly what the model would be sent, and call nothing
        for sym in symbols:
            close = db.load_close(conn, sym)
            prompt = None if close.empty else research.prompt_for(
                sym, close, recent_news(sym), periods_per_year(sym))
            print(prompt or f"{sym}: no prices or no recent news, so nothing would be sent.")
            print("-" * 72)
        return

    with RunLog(LOGS / "research.log", f"research ({', '.join(llm.models)})") as say:
        for sym in symbols:
            close = db.load_close(conn, sym)
            if close.empty:
                say(f"{sym}  no prices stored. Run: python main.py fetch {sym}")
                continue
            try:
                items = recent_news(sym)
            except Exception as e:  # noqa: BLE001 - one symbol's news must not stop the rest
                say(f"{sym}  could not get news: {type(e).__name__}: {e}")
                continue
            db.save_news(conn, items)
            note = research.research_symbol(conn, sym, close, items, llm,
                                            periods_per_year(sym), refresh=args.refresh)
            say(research.render(note))
            say()
            if note.stop_batch:
                say("No Gemini model could answer, so the remaining symbols were skipped to save "
                    "quota. Run it again later; answers already kept are reused, not re-asked.")
                break
        say("Advisory only: these notes are logged and scored. The bot does not trade on them.")


def cmd_research_score(cfg, args):
    from tradebot import research

    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    table = research.score(conn)
    if table["notes"].sum() == 0:
        print("No research notes kept yet. Run: python main.py research")
        return
    print("Average return after each kind of note. A note is scored from the first close")
    print("after it was written, over the next 1, 5 and 20 bars.\n")
    with pd.option_context("display.float_format", "{:+.2%}".format, "display.width", 140):
        print(table.to_string(index=False))
    if table[[c for c in table if c.startswith("scored_")]].to_numpy().sum() == 0:
        print("\nNone are old enough to score yet. Check back after a few trading days.")


def cmd_news_archive(cfg, args):
    """Download the historical news the news test needs. Safe to re-run: finished months are skipped."""
    from tradebot import news_archive

    nt = cfg["news_test"]
    secrets = load_secrets()
    if not secrets["alpaca_key"] or not secrets["alpaca_secret"]:
        sys.exit("The news archive needs ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")
    symbols = sorted(set(nt["universe"]) | set(nt.get("aliases") or {}))
    start = (pd.Timestamp(nt["start"]) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    conn = news_archive.connect(ROOT / nt["archive_path"])
    print(f"Downloading news tagged with {len(symbols)} tickers since {start}")
    fetched = news_archive.download(conn, secrets["alpaca_key"], secrets["alpaca_secret"],
                                    symbols, start)
    stored = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    print(f"Fetched {fetched} articles this run; {stored} stored in {nt['archive_path']}")


def cmd_news_test(cfg, args):
    """Research only: nothing here trades."""
    from tradebot import news_archive, newstest

    nt = cfg["news_test"]
    archive = ROOT / nt["archive_path"]
    if not archive.exists():
        sys.exit("No news archive yet. Run: python main.py news-archive")
    articles = news_archive.load_articles(news_archive.connect(archive))
    conn = db.connect(cfg["db_path"])
    prices = db.load_ohlcv(conn, nt["universe"])
    have = set(prices["symbol"])
    missing = [s for s in nt["universe"] if s not in have]
    if len(have) < 10:
        sys.exit("Too few symbols have prices. Run: python main.py fetch " + " ".join(nt["universe"]))
    result = newstest.run(articles, prices, universe=[s for s in nt["universe"] if s in have],
                          aliases=nt.get("aliases") or {}, start=nt["start"], no_prices=missing,
                          settings=newstest.Settings(cost_bps=cfg["backtest"]["cost_bps"]))
    report = newstest.render(result)
    print(report)
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "news_test_report.md"
    path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {path}")


def _in_cooldown(conn, symbol, risk) -> bool:
    stopped = db.last_stop_date(conn, symbol)
    if stopped is None:
        return False
    bars_since = int((db.load_close(conn, symbol).index > pd.Timestamp(stopped)).sum())
    return bars_since < risk["stop_cooldown_bars"]


def _record_earlier_orders(conn, broker, say) -> None:
    """Look up orders sent on earlier runs and record how they ended."""
    waiting = db.open_trades(conn, broker.account_name)
    if not waiting:
        return
    say("Earlier orders:")
    for trade_id, ts, symbol, side, order_id in waiting:
        info = broker.order_info(order_id)
        if info is None:
            say(f"  {side:<5} {symbol:<6} sent {ts[:10]}: the broker has no record of it")
            continue
        db.update_trade(conn, trade_id, info.status, info.filled_qty, info.filled_avg_price)
        if info.filled_qty and info.filled_avg_price:
            say(f"  {side:<5} {symbol:<6} sent {ts[:10]}: {info.status}: "
                f"{info.filled_qty:g} @ ${info.filled_avg_price:,.2f}")
        else:
            say(f"  {side:<5} {symbol:<6} sent {ts[:10]}: {info.status}")


def cmd_paper(cfg, args):
    from tradebot.broker import AlpacaBroker, enum_text

    secrets = load_secrets()
    risk, pb = cfg["risk"], cfg["paper_bot"]
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)

    title = "paper --execute" if args.execute else "paper (dry run)"
    with RunLog(LOGS / "paper_trading.log", title) as say:
        try:
            check_live_allowed(secrets["alpaca_paper"], risk)
            broker = AlpacaBroker(secrets["alpaca_key"], secrets["alpaca_secret"],
                                  paper=secrets["alpaca_paper"])
        except (RiskViolation, RuntimeError) as e:
            sys.exit(str(e))
        acct = broker.account()
        db.log_equity(conn, broker.account_name, acct["equity"], acct["cash"])
        say(f"[{broker.account_name}] equity ${acct['equity']:,.2f}  cash ${acct['cash']:,.2f}")
        _record_earlier_orders(conn, broker, say)
        held = broker.position_details()
        try:      # keeps the dashboard's numbers current; never allowed to stop a trading run
            sync_account(conn, broker, held)
        except Exception as e:  # noqa: BLE001
            say(f"Note: could not save account history for the dashboard ({type(e).__name__}: {e})")

        # Two guards, two strengths. Old prices make every signal untrustworthy,
        # so only stop-loss exits pass. A big down day blocks adding risk, but
        # sells that reduce it still go through.
        stale = daily_loss = None
        paused = db.is_paused(conn)
        expected = broker.last_completed_session()    # by the exchange's calendar, not this PC's
        try:
            check_data_fresh({s: db.last_price_date(conn, s) for s in pb["symbols"]}, expected)
        except RiskViolation as e:
            stale = str(e)
        try:
            check_daily_loss(acct["equity"], acct["last_equity"], risk)
        except RiskViolation as e:
            daily_loss = str(e)

        stops = triggered_stops({s: p.unrealized_plpc for s, p in held.items()}, risk)
        for sym, pl in stops.items():
            say(f"  STOP  {sym:<6} {pl:+.1%} (limit {-stop_loss_pct(risk):+.1%})")

        targets = _current_targets(cfg, conn, pb["strategy"], pb["symbols"])
        cooling = [s for s in targets if s not in stops and _in_cooldown(conn, s, risk)]
        for sym in targets:
            if sym in stops or sym in cooling:
                targets[sym] = 0.0
        say(f"Targets ({pb['strategy']}): " + ", ".join(f"{s} {t:.2f}" for s, t in targets.items()))
        if cooling:
            say(f"Held at 0 during the stop-loss cooldown: {', '.join(cooling)}")

        orders = plan_orders(targets, {s: p.market_value for s, p in held.items()},
                             acct["equity"], risk, close_unlisted=pb.get("close_unlisted", True))
        if stale:
            say(f"Blocked: {stale}")
            say("Only stop-loss exits are allowed.")
            orders = [o for o in orders if o.symbol in stops and o.side == "close"]
        elif daily_loss:
            say(f"Blocked: {daily_loss}")
            say("Sells and stop-loss exits still go through.")
            orders = [o for o in orders if o.side != "buy"]
        if paused:
            say("Paused (set from the dashboard or `python main.py pause`): no buys.")
            say("Sells and stop-loss exits still go through. `python main.py resume` lifts it.")
            orders = [o for o in orders if o.side != "buy"]
        if args.execute and stale:
            _alert(cfg, conn, "warning", "Paper bot did not trade: prices are out of date", stale, "paper")
        elif args.execute and daily_loss:
            _alert(cfg, conn, "warning", "Daily loss limit hit: no new buys today", daily_loss, "paper")

        # An order from an earlier run that hasn't filled yet (e.g. sent after
        # the close) is not in the positions, so the plan would repeat it.
        pending = broker.open_order_symbols()
        for o in orders:
            if o.symbol in pending:
                say(f"  skip  {o.symbol:<6} an earlier order is still open, so no new {o.side} order")
        orders = [o for o in orders if o.symbol not in pending]
        orders.sort(key=lambda o: o.side == "buy")   # sells first, so buys never lean on margin

        if not orders:
            say("No orders to send." if (stale or daily_loss or pending or paused)
                else "Portfolio already on target. No orders.")
            return
        say("Planned orders:")
        for o in orders:
            say(f"  {o.side:<5} {o.symbol:<6} ${o.notional:>10,.2f}   "
                f"(have ${o.current_value:,.0f} -> want ${o.target_value:,.0f})")
        if not args.execute:
            say("Dry run: nothing was sent. Use --execute to send these orders.")
            return
        if not broker.is_market_open():
            say("Market closed: orders will wait for the next open.")

        failed = []
        for o in orders:
            side = "sell" if o.side == "close" else o.side
            try:
                resp = (broker.close_position(o.symbol) if o.side == "close"
                        else broker.submit_notional(o.symbol, o.side, o.notional))
            except Exception as e:  # noqa: BLE001 - one bad order must not strand the others
                failed.append(o.symbol)
                say(f"  FAILED {side} {o.symbol}: {type(e).__name__}: {e}")
                continue
            order_id = str(getattr(resp, "id", ""))
            db.log_trade(conn, broker.account_name, o.symbol, side, notional=o.notional,
                         strategy=pb["strategy"], order_id=order_id,
                         notes="stop-loss" if o.symbol in stops else None,
                         status=enum_text(getattr(resp, "status", None)))
            if o.symbol in stops:
                db.log_stop(conn, o.symbol,
                            db.last_price_date(conn, o.symbol) or expected or db.now_utc()[:10],
                            stops[o.symbol])
                _alert(cfg, conn, "warning", f"Stop-loss: selling {o.symbol} ({stops[o.symbol]:+.1%})",
                       f"{o.symbol} fell {stops[o.symbol]:+.1%} from its entry price (limit "
                       f"{-stop_loss_pct(risk):+.1%}). The sell order was sent and fills at the next open. "
                       f"The bot stays out of {o.symbol} for {risk['stop_cooldown_bars']} trading days.",
                       "paper")
            say(f"  sent {side} {o.symbol} ${o.notional:,.2f}  (order {order_id})")
        say(f"Result: {len(orders) - len(failed)} of {len(orders)} orders sent.")
        if failed:
            sys.exit(f"{len(failed)} order(s) failed: {', '.join(failed)}")


def cmd_log_trade(cfg, args):
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    followed = None if args.followed_ai is None else int(args.followed_ai == "yes")
    tid = db.log_trade(conn, args.account, args.symbol, args.side, qty=args.qty,
                       price=args.price, notional=args.qty * args.price, fees=args.fees,
                       strategy=args.strategy, followed_ai=followed, notes=args.notes)
    print(f"Logged trade #{tid}: {args.side} {args.qty} {args.symbol.upper()} @ {args.price} ({args.account})")


def cmd_daily(cfg, args):
    """Unattended daily run (what the scheduled task starts). See tradebot/daily.py."""
    from tradebot.broker import AlpacaBroker
    from tradebot.daily import daily_run

    secrets = load_secrets()
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    try:
        broker = AlpacaBroker(secrets["alpaca_key"], secrets["alpaca_secret"], paper=secrets["alpaca_paper"])
        sessions = broker.trading_days()
    except Exception as e:  # noqa: BLE001 - recorded, then stop: without the calendar we can't know
        problem = f"Could not read the exchange calendar ({type(e).__name__}: {e}); nothing was run."
        _alert(cfg, conn, "error", "Daily run could not start", problem, "daily")
        with RunLog(LOGS / "daily_run.log", "daily run"):
            sys.exit(problem)
    today = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
    results = daily_run(ROOT, sessions, today, conn=conn,
                        alert=lambda level, title, detail: _alert(cfg, conn, level, title, detail, "daily"),
                        summary=bool((cfg.get("alerts") or {}).get("daily_summary", True)))
    if any(code != 0 for _, code in results):
        sys.exit(1)


def run_daily_guarded(args):
    """Start `cmd_daily`, recording any crash in logs/daily_run.log and as an alert.

    The scheduled task runs under pythonw: there is no console, so an error raised before
    the run's own log opens (reading config.yaml, .env, opening the database) used to vanish
    with nothing but exit code 1 in Task Scheduler. Exits with code 1 after recording it.
    """
    import traceback

    cfg = None
    try:
        cfg = load_config()
        cmd_daily(cfg, args)
    except Exception as e:  # noqa: BLE001 - record anything, then still fail the task
        with RunLog(LOGS / "daily_run.log", "daily run") as say:
            say(f"Crashed: {type(e).__name__}: {e}. Nothing after this point ran.")
            for line in traceback.format_exc().rstrip().splitlines():
                say(f"    {line}")
        try:
            conn = db.connect(cfg["db_path"] if cfg else str(ROOT / "data" / "trading.db"))
            db.init_db(conn)
        except Exception:  # noqa: BLE001 - the database may be the thing that failed
            conn = None
        detail = f"{type(e).__name__}: {e}\nFull traceback in logs\\daily_run.log."
        if conn is not None:
            _alert(cfg or {}, conn, "error", "Daily run crashed", detail, "daily")
        else:
            try:
                with open(LOGS / "alerts.log", "a", encoding="utf-8") as fh:
                    fh.write(f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}  ERROR   "
                             f"Daily run crashed (database unavailable, alert not stored)\n    {detail}\n")
            except OSError:
                pass
        sys.exit(1)


def _model_setup(cfg):
    model = cfg.get("model")
    if not model:
        sys.exit("config.yaml has no `model` section.")
    conn = db.connect(model["prices_path"])
    db.init_db(conn)
    return model, conn


def _model_dataset(model, conn, say):
    from tradebot import modeldata, rankmodel
    panels, breaks = modeldata.load_clean_panels(conn, model.get("break_ratio"))
    if not panels:
        sys.exit("No prices for the model yet. Run: python main.py model-fetch")
    data = rankmodel.build_dataset(panels, int(model["horizon_days"]))
    say(f"{len(data.symbols)} stocks, {len(data.days):,} days of prices, "
        f"{data.days[0]:%Y-%m-%d} to {data.days[-1]:%Y-%m-%d}. "
        f"{len(breaks)} series end early at a data break.")
    return panels, data, breaks


def cmd_model_fetch(cfg, args):
    """Download or refresh the wide universe's prices from Alpaca (their own database)."""
    from tradebot import modeldata

    model, conn = _model_setup(cfg)
    secrets = load_secrets()
    if not secrets["alpaca_key"] or not secrets["alpaca_secret"]:
        sys.exit("model-fetch needs ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")
    with RunLog(LOGS / "model.log", "model-fetch") as say:
        def fetch(symbols, start):
            return modeldata.fetch_bars(symbols, start, secrets["alpaca_key"], secrets["alpaca_secret"],
                                        model.get("asof"))
        modeldata.update_prices(conn, cfg["universes"][model["universe"]], model["start"], model.get("asof"),
                                fetch, say, full=args.full)


def _clear_scores(conn, say):
    import shutil
    conn.execute("DELETE FROM model_scores")
    conn.commit()
    shutil.rmtree(ROOT / "data" / "models", ignore_errors=True)
    say("Stored scores and saved models were deleted; everything is computed again.")


def cmd_model_score(cfg, args):
    """Score every stock for each day from model.first_test_year on that has no score yet."""
    from tradebot import rankmodel

    model, conn = _model_setup(cfg)
    with RunLog(LOGS / "model.log", "model-score") as say:
        if args.rebuild:
            _clear_scores(conn, say)
        _, data, _ = _model_dataset(model, conn, say)
        added = rankmodel.update_scores(conn, data, model, ROOT / "data" / "models", say)
        say("New scores: " + ", ".join(f"{name} {count:,}" for name, count in added.items()) +
            f". Scored up to {data.days[-1]:%Y-%m-%d}.")


def cmd_model_test(cfg, args):
    """Walk-forward test of the ranking model. Research only: nothing here trades."""
    from tradebot import rankmodel

    model, conn = _model_setup(cfg)
    with RunLog(LOGS / "model.log", "model-test") as say:
        if args.rebuild:
            _clear_scores(conn, say)
        panels, data, breaks = _model_dataset(model, conn, say)
        rankmodel.update_scores(conn, data, model, ROOT / "data" / "models", say)
        plan = rankmodel.folds(data, int(model["first_test_year"]))
        if not plan:
            sys.exit("Not enough history before model.first_test_year to train a model.")
        test_days = data.days[data.days.year >= int(model["first_test_year"])]
        top_n, cost_bps = int(model.get("top_n", 25)), float(cfg["backtest"]["cost_bps"])
        results = {name: rankmodel.evaluate(rankmodel.load_scores(conn, name), data, top_n, cost_bps)
                   for name in model["models"]}
        last = panels["close"].apply(lambda column: column.last_valid_index())
        info = {"symbols": len(data.symbols), "ended": int((last < data.days[-1]).sum()),
                "first_bar": f"{data.days[0]:%Y-%m-%d}", "last_bar": f"{data.days[-1]:%Y-%m-%d}",
                "horizon": data.horizon, "top_n": top_n, "cost_bps": cost_bps,
                "break_ratio": model.get("break_ratio"), "breaks": breaks}
        report = rankmodel.render(results, rankmodel.single_inputs(data, test_days),
                                  rankmodel.look_ahead_example(panels, data, test_days), plan, info)
    print(report)
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "model_test_report.md"
    path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {path}")


def cmd_league(cfg, args):
    """Replay every league bot from stored prices and print the leaderboard."""
    from tradebot.league import run_league

    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    with RunLog(LOGS / "league.log", "league") as say:
        try:
            board = run_league(conn, cfg, say)
        except ValueError as e:
            sys.exit(f"League settings problem: {e}")
        if board.empty:
            say("Nothing to show yet. Run `python main.py fetch` first.")
            return
        pct = lambda v, signed=True: "n/a" if v is None or pd.isna(v) else (f"{v:+.1%}" if signed else f"{v:.0%}")  # noqa: E731
        num = lambda v: "n/a" if v is None or pd.isna(v) else f"{v:.2f}"  # noqa: E731
        say(f"Prices up to {db.last_price_date(conn, cfg['benchmark'])}. Every bot started with "
            f"${(cfg.get('league') or {}).get('start_cash', 100_000):,.0f}.")
        say("LIVE = after the bot's settings were fixed (real evidence). "
            "HISTORY = backtest before that (settings chosen with hindsight).")
        say(f"{'bot':<20}{'symbols':>8}{'invested':>10}{'live days':>11}{'live':>9}{'SPY live':>10}"
            f"{'hist/yr':>9}{'SPY/yr':>8}{'Sharpe':>8}{'worst drop':>12}{'trades':>8}")
        for row in board.itertuples():
            say(f"{row.bot:<20}{row.symbols:>8}{pct(row.invested, False):>10}{row.live_days:>11}"
                f"{pct(row.live_return):>9}{pct(row.spy_live_return):>10}{pct(row.history_cagr):>9}"
                f"{pct(row.spy_history_cagr):>8}{num(row.history_sharpe):>8}"
                f"{pct(row.history_max_drawdown):>12}{row.trades:>8}")


def cmd_pause(cfg, args):
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    pausing = args.cmd == "pause"
    db.set_setting(conn, "paused", "1" if pausing else "0")
    print("Paused: the paper bot will not buy. Sells and stop-loss exits still go through."
          if pausing else "Resumed: the paper bot can buy again on its next run.")


def cmd_alerts(cfg, args):
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    recent = db.alerts_df(conn, limit=args.limit)
    if recent.empty:
        print("No alerts so far.")
        return
    for row in recent.iloc[::-1].itertuples():
        print(f"{row.ts[:16].replace('T', ' ')} UTC  {row.level.upper():<7} {row.title}"
              f"   [sent to: {row.delivered or 'nowhere'}]")
        for line in (row.detail or "").splitlines():
            print(f"    {line}")
        if row.failed:
            print(f"    not sent: {row.failed}")


def cmd_alert_test(cfg, args):
    """Send a test alert through every channel that is switched on, and say what happened."""
    from tradebot.alerts import configured_channels

    load_secrets()                         # reads .env, where the phone channels are set
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    channels = configured_channels(cfg)
    if not channels:
        sys.exit("No alert channel is switched on. See the `alerts` section of README.md.")
    print(f"Sending a test alert to: {', '.join(channels)}")
    alert_id = raise_alert(conn, "info", "Trading assistant: test alert",
                           "If you can read this, alerts work on this channel.", "test",
                           cfg=cfg, log_path=LOGS / "alerts.log")
    delivered, failed = conn.execute("SELECT delivered, failed FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    print(f"Delivered: {delivered or 'none'}")
    if failed:
        sys.exit(f"Not delivered: {failed}")


def cmd_dashboard(cfg, args):
    """Start the web dashboard and open it in the browser (Ctrl+C here stops it)."""
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard.py")], cwd=ROOT))


def cmd_report(cfg, args):
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    t = db.trades_df(conn)
    if t.empty:
        print("No trades logged yet.")
        return
    print(t.groupby(["account", "side"]).agg(trades=("id", "count"), notional=("notional", "sum")).to_string())
    manual = t[t["account"] == "fidelity_manual"]
    if not manual.empty and manual["followed_ai"].notna().any():
        rate = manual["followed_ai"].mean()
        print(f"\nManual trades that followed the AI signal: {rate:.0%}")
    snaps = pd.read_sql_query("SELECT * FROM equity_snapshots ORDER BY ts", conn)
    if not snaps.empty:
        print("\nLatest equity:")
        print(snaps.groupby("account").tail(1).to_string(index=False))


def build_parser():
    p = argparse.ArgumentParser(description="AI trading assistant")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    f = sub.add_parser("fetch"); f.add_argument("symbols", nargs="*")
    f.add_argument("--full", action="store_true", help="re-download full history instead of resuming")
    f.add_argument("--source", choices=["yahoo", "alpaca"], default="yahoo",
                   help="alpaca: for stocks Yahoo no longer serves, such as delisted ones")
    q = sub.add_parser("quote"); q.add_argument("symbols", nargs="*")
    b = sub.add_parser("backtest"); b.add_argument("symbol"); b.add_argument("strategy", choices=list(STRATEGIES))
    b.add_argument("--no-stop", action="store_true", help="leave out the stop-loss")
    sub.add_parser("backtest-all")
    w = sub.add_parser("walk-forward", help="out-of-sample test with per-fold parameter selection")
    w.add_argument("symbols", nargs="*")
    w.add_argument("--strategy", choices=[n for n in STRATEGIES if n != "buy_and_hold"])
    w.add_argument("--verbose", action="store_true", help="print each fold")
    w.add_argument("--no-stop", action="store_true", help="leave out the stop-loss")
    s = sub.add_parser("signals"); s.add_argument("--strategy", choices=list(STRATEGIES))
    r = sub.add_parser("research", help="LLM reads recent news and writes a stance with reasons")
    r.add_argument("symbols", nargs="*")
    r.add_argument("--refresh", action="store_true", help="ask the model again even if nothing changed")
    r.add_argument("--show-prompt", action="store_true",
                   help="print what would be sent to the model, and send nothing")
    sub.add_parser("research-score", help="compare past research notes with what prices did next")
    sub.add_parser("news-archive", help="download historical news for the news test (resumable)")
    sub.add_parser("news-test", help="walk-forward test: does past news predict the next session?")
    pp = sub.add_parser("paper"); pp.add_argument("--execute", action="store_true")
    lt = sub.add_parser("log-trade", help="record a manual (e.g. Fidelity) trade")
    lt.add_argument("side", choices=["buy", "sell"])
    lt.add_argument("symbol"); lt.add_argument("qty", type=float); lt.add_argument("price", type=float)
    lt.add_argument("--account", default="fidelity_manual")
    lt.add_argument("--fees", type=float, default=0.0)
    lt.add_argument("--strategy")
    lt.add_argument("--followed-ai", choices=["yes", "no"])
    lt.add_argument("--notes")
    sub.add_parser("report")
    sub.add_parser("league", help="replay every league bot from stored prices; print the leaderboard")
    mf = sub.add_parser("model-fetch", help="download or refresh the ranking model's prices (Alpaca)")
    mf.add_argument("--full", action="store_true", help="download the whole history again")
    for name, text in (("model-score", "score every stock for the days not scored yet"),
                       ("model-test", "walk-forward test of the ranking model, with a report")):
        mp = sub.add_parser(name, help=text)
        mp.add_argument("--rebuild", action="store_true",
                        help="delete stored scores and saved models first (needed after changing model settings)")
    sub.add_parser("pause", help="stop the paper bot from buying (sells and stop-losses still run)")
    sub.add_parser("resume", help="let the paper bot buy again")
    al = sub.add_parser("alerts", help="show recent alerts"); al.add_argument("--limit", type=int, default=20)
    sub.add_parser("alert-test", help="send a test alert through every channel that is switched on")
    sub.add_parser("dashboard", help="start the web dashboard")
    sub.add_parser("daily", help="fetch, research and paper --execute in one go (for the scheduled task)")
    return p


COMMANDS = {
    "init": cmd_init, "fetch": cmd_fetch, "quote": cmd_quote, "backtest": cmd_backtest,
    "backtest-all": cmd_backtest_all, "walk-forward": cmd_walk_forward,
    "signals": cmd_signals, "research": cmd_research, "research-score": cmd_research_score,
    "news-archive": cmd_news_archive, "news-test": cmd_news_test,
    "paper": cmd_paper, "daily": cmd_daily,
    "log-trade": cmd_log_trade, "report": cmd_report,
    "league": cmd_league, "pause": cmd_pause, "resume": cmd_pause,
    "model-fetch": cmd_model_fetch, "model-score": cmd_model_score, "model-test": cmd_model_test,
    "alerts": cmd_alerts, "alert-test": cmd_alert_test, "dashboard": cmd_dashboard,
}

if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        # News text can hold characters a Windows console can't print; show "?" rather than crash.
        # Under pythonw (the scheduled task, no window) there is no console: stream is None.
        if stream is not None:
            stream.reconfigure(errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    if args.cmd == "daily":
        run_daily_guarded(args)
    else:
        COMMANDS[args.cmd](load_config(), args)
