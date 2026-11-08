"""Command-line entry point.

    python main.py init                      # create database
    python main.py fetch                     # download/refresh prices for watchlist
    python main.py backtest SPY sma_crossover
    python main.py backtest-all              # every watchlist symbol x every strategy
    python main.py signals                   # today's signals (what the AI suggests)
    python main.py paper                     # dry run: show orders the bot WOULD place
    python main.py paper --execute           # actually send them to Alpaca paper
    python main.py log-trade buy AAPL 2 228.50 --followed-ai yes --notes "SMA signal"
    python main.py report                    # summary of trades logged
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tradebot import db
from tradebot.backtest import compare, format_metrics, plot_equity, run_backtest
from tradebot.config import ROOT, load_config, load_secrets
from tradebot.data import fetch_prices, periods_per_year
from tradebot.rebalance import plan_orders
from tradebot.risk import RiskViolation, check_daily_loss, check_live_allowed
from tradebot.strategies import STRATEGIES, get_strategy

REPORTS = ROOT / "reports"


def cmd_init(cfg, args):
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    print(f"Database ready at {cfg['db_path']}")


def cmd_fetch(cfg, args):
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    symbols = args.symbols or cfg["watchlist"]
    for sym in symbols:
        df = fetch_prices(sym, start=cfg["start_date"])
        n = db.upsert_prices(conn, df) if not df.empty else 0
        print(f"{sym:<8} {n:>5} rows  (latest: {db.last_price_date(conn, sym)})")


def _strategy_params(cfg, name):
    return cfg.get("strategies", {}).get(name, {})


def _backtest_one(cfg, conn, symbol, strategy_name, plot=True):
    close = db.load_close(conn, symbol)
    if close.empty:
        sys.exit(f"No prices for {symbol}. Run: python main.py fetch")
    bt = cfg["backtest"]
    ppy = periods_per_year(symbol)
    strat = get_strategy(strategy_name)
    res = run_backtest(close, strat(close, **_strategy_params(cfg, strategy_name)),
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
    res, bench = _backtest_one(cfg, conn, args.symbol, args.strategy)
    print(f"\n{args.symbol}  {res.equity.index[0].date()} -> {res.equity.index[-1].date()}")
    print(format_metrics(compare([res, bench])).to_string())


def cmd_backtest_all(cfg, args):
    conn = db.connect(cfg["db_path"])
    rows = []
    for sym in cfg["watchlist"]:
        for name in STRATEGIES:
            if name == "buy_and_hold":
                continue
            res, bench = _backtest_one(cfg, conn, sym, name, plot=False)
            rows.append({"symbol": sym, "strategy": name,
                         **{f"strat_{k}": v for k, v in res.metrics.items() if k in ("cagr", "sharpe", "max_drawdown")},
                         **{f"bh_{k}": v for k, v in bench.metrics.items() if k in ("cagr", "sharpe", "max_drawdown")}})
    df = pd.DataFrame(rows)
    df["beats_bh_sharpe"] = df["strat_sharpe"] > df["bh_sharpe"]
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "backtest_summary.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(df.to_string(index=False))
    print(f"\nSaved {REPORTS / 'backtest_summary.csv'}")


def _current_targets(cfg, conn, strategy_name, symbols, log=True):
    strat = get_strategy(strategy_name)
    targets = {}
    for sym in symbols:
        close = db.load_close(conn, sym)
        if close.empty:
            continue
        t = float(strat(close, **_strategy_params(cfg, strategy_name)).iloc[-1])
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
    print(f"Signals ({name})  1 = hold/buy, 0 = stay out")
    for sym, t in targets.items():
        print(f"  {sym:<8} {t:.0f}")


def cmd_paper(cfg, args):
    from tradebot.broker import AlpacaBroker

    secrets = load_secrets()
    risk = cfg["risk"]
    check_live_allowed(secrets["alpaca_paper"], risk)
    broker = AlpacaBroker(secrets["alpaca_key"], secrets["alpaca_secret"], paper=secrets["alpaca_paper"])

    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    acct = broker.account()
    db.log_equity(conn, broker.account_name, acct["equity"], acct["cash"])
    print(f"[{broker.account_name}] equity ${acct['equity']:,.2f}  cash ${acct['cash']:,.2f}")

    try:
        check_daily_loss(acct["equity"], acct["last_equity"], risk)
    except RiskViolation as e:
        sys.exit(str(e))

    pb = cfg["paper_bot"]
    targets = _current_targets(cfg, conn, pb["strategy"], pb["symbols"])
    orders = plan_orders(targets, broker.positions(), acct["equity"], risk)
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
                     strategy=pb["strategy"], order_id=str(getattr(resp, "id", "")))
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
    b = sub.add_parser("backtest"); b.add_argument("symbol"); b.add_argument("strategy", choices=list(STRATEGIES))
    sub.add_parser("backtest-all")
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
    "init": cmd_init, "fetch": cmd_fetch, "backtest": cmd_backtest,
    "backtest-all": cmd_backtest_all, "signals": cmd_signals, "paper": cmd_paper,
    "log-trade": cmd_log_trade, "report": cmd_report,
}

if __name__ == "__main__":
    args = build_parser().parse_args()
    COMMANDS[args.cmd](load_config(), args)
