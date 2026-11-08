"""Command-line entry point.

    python main.py init                      # create database
    python main.py fetch                     # download/refresh prices for watchlist
    python main.py quote [SYMBOLS...]        # real-time bid/ask from Alpaca (free IEX feed)
    python main.py backtest SPY sma_crossover
    python main.py backtest-all              # every watchlist symbol x every strategy
    python main.py walk-forward [SYMBOLS...] # out-of-sample test, parameters re-picked each fold
    python main.py signals                   # today's signals (what the AI suggests)
    python main.py paper                     # dry run: show orders the bot WOULD place
    python main.py paper --execute           # actually send them to Alpaca paper
    python main.py log-trade buy AAPL 2 228.50 --followed-ai yes --notes "SMA signal"
    python main.py report                    # summary of trades logged
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from tradebot import db
from tradebot.backtest import compare, format_metrics, plot_equity, run_backtest, with_stop_loss
from tradebot.config import ROOT, load_config, load_secrets
from tradebot.data import (AlpacaQuotes, fetch_many, fetch_prices, history_was_readjusted,
                           incremental_start, periods_per_year)
from tradebot.rebalance import plan_orders
from tradebot.risk import (RiskViolation, check_daily_loss, check_data_fresh, check_live_allowed,
                           stop_loss_pct, triggered_stops)
from tradebot.strategies import STRATEGIES, get_strategy
from tradebot.walkforward import walk_forward

REPORTS = ROOT / "reports"


def cmd_init(cfg, args):
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    print(f"Database ready at {cfg['db_path']}")


def cmd_fetch(cfg, args):
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    symbols = args.symbols or cfg["watchlist"]
    full_start = cfg["start_date"]
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
        n = db.upsert_prices(conn, result) if not result.empty else 0
        print(f"{sym:<8} {n:>5} rows  (latest: {db.last_price_date(conn, sym)})")
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
        print(f"{q.symbol:<8} {q.bid:>11,.2f} {q.ask:>11,.2f} {q.mid:>11,.2f} {q.spread_bps:>6.1f}bp"
              f"  {q.timestamp:%Y-%m-%d %H:%M:%S}  {q.source}")


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


def _in_cooldown(conn, symbol, risk) -> bool:
    stopped = db.last_stop_date(conn, symbol)
    if stopped is None:
        return False
    bars_since = int((db.load_close(conn, symbol).index > pd.Timestamp(stopped)).sum())
    return bars_since < risk["stop_cooldown_bars"]


def cmd_paper(cfg, args):
    from tradebot.broker import AlpacaBroker

    secrets = load_secrets()
    risk = cfg["risk"]
    check_live_allowed(secrets["alpaca_paper"], risk)
    pb = cfg["paper_bot"]

    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    try:
        broker = AlpacaBroker(secrets["alpaca_key"], secrets["alpaca_secret"], paper=secrets["alpaca_paper"])
    except RuntimeError as e:
        sys.exit(str(e))
    acct = broker.account()
    db.log_equity(conn, broker.account_name, acct["equity"], acct["cash"])
    print(f"[{broker.account_name}] equity ${acct['equity']:,.2f}  cash ${acct['cash']:,.2f}")

    # These guards block new buys but must not block stop-loss exits.
    halted = None
    try:
        check_data_fresh({s: db.last_price_date(conn, s) for s in pb["symbols"]},
                         date.today(), risk)
        check_daily_loss(acct["equity"], acct["last_equity"], risk)
    except RiskViolation as e:
        halted = str(e)

    held = broker.position_details()
    stops = triggered_stops({s: p.unrealized_plpc for s, p in held.items()}, risk)
    for sym, pl in stops.items():
        print(f"  STOP  {sym:<6} {pl:+.1%} (limit {-stop_loss_pct(risk):+.1%})")

    targets = _current_targets(cfg, conn, pb["strategy"], pb["symbols"])
    for sym in targets:
        if sym in stops or _in_cooldown(conn, sym, risk):
            targets[sym] = 0.0
    orders = plan_orders(targets, {s: p.market_value for s, p in held.items()}, acct["equity"], risk,
                         close_unlisted=pb.get("close_unlisted", True))
    if halted:
        print(f"Blocked: {halted}")
        orders = [o for o in orders if o.symbol in stops and o.side == "close"]
        if not orders:
            sys.exit("No stop-loss exits to send, so no orders.")
        print("Sending stop-loss exits only.")
    if not orders:
        print("Portfolio already on target. No orders.")
        return
    for o in orders:
        print(f"  {o.side:<5} {o.symbol:<6} ${o.notional:>9,.2f}   (have ${o.current_value:,.0f} -> want ${o.target_value:,.0f})")
    if not args.execute:
        print("\nDry run. Re-run with --execute to send these orders.")
        return
    if not broker.is_market_open():
        print("Market closed: orders will queue for the next open (time_in_force=DAY).")
    for o in orders:
        if o.side == "close":
            resp = broker.close_position(o.symbol)
            side = "sell"
        else:
            resp = broker.submit_notional(o.symbol, o.side, o.notional)
            side = o.side
        db.log_trade(conn, broker.account_name, o.symbol, side, notional=o.notional,
                     strategy=pb["strategy"], order_id=str(getattr(resp, "id", "")),
                     notes="stop-loss" if o.symbol in stops else None)
        if o.symbol in stops:
            db.log_stop(conn, o.symbol, db.last_price_date(conn, o.symbol) or date.today().isoformat(),
                        stops[o.symbol])
        print(f"  sent {side} {o.symbol}")


def cmd_log_trade(cfg, args):
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    followed = None if args.followed_ai is None else int(args.followed_ai == "yes")
    tid = db.log_trade(conn, args.account, args.symbol, args.side, qty=args.qty,
                       price=args.price, notional=args.qty * args.price, fees=args.fees,
                       strategy=args.strategy, followed_ai=followed, notes=args.notes)
    print(f"Logged trade #{tid}: {args.side} {args.qty} {args.symbol.upper()} @ {args.price} ({args.account})")


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
    return p


COMMANDS = {
    "init": cmd_init, "fetch": cmd_fetch, "quote": cmd_quote, "backtest": cmd_backtest,
    "backtest-all": cmd_backtest_all, "walk-forward": cmd_walk_forward,
    "signals": cmd_signals, "paper": cmd_paper,
    "log-trade": cmd_log_trade, "report": cmd_report,
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    COMMANDS[args.cmd](load_config(), args)
