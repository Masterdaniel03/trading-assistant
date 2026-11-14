"""The bot league: several simulated bots, each with its own strategy and symbols.

Every bot follows the same rules as the real paper bot: decide after the close,
trade at the next open, equal weight capped by max_position_pct, the same
order planner (rebalance.plan_orders), the same stop-loss and cooldown, and a
cost on every trade. The fills are simulated from stored prices, so any number
of bots can run side by side without a broker account each.

Each run replays every bot from `history_start` to the latest stored price.
That makes the result independent of whether the run on the day before happened.

Two periods are kept apart, because only one of them is evidence:

  history  before the bot's settings were first recorded. This is a backtest:
           the strategy and settings were chosen knowing these prices.
  live     after that day (`frozen_on`). Nothing about the bot changed once
           these prices were known. Changing a bot's settings restarts its
           live record.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradebot import db
from tradebot.data import is_crypto
from tradebot.rebalance import plan_orders
from tradebot.risk import stop_loss_pct
from tradebot.strategies import get_strategy

TOLERANCE = 0.10          # the real bot's rebalance band (plan_orders' default)


@dataclass(frozen=True)
class BotSpec:
    name: str
    strategy: str
    universe: str
    symbols: tuple[str, ...]
    params: dict
    risk: dict
    stop_loss: bool = True

    def fingerprint(self, start_cash: float, cost_bps: float, history_start: str) -> str:
        """Everything that decides this bot's trades. If it changes, the live record restarts."""
        keys = ("max_position_pct", "max_total_exposure_pct", "max_risk_per_position_pct",
                "stop_cooldown_bars", "min_order_notional")
        return json.dumps({"strategy": self.strategy, "symbols": list(self.symbols), "params": self.params,
                           "risk": {k: self.risk[k] for k in keys}, "stop_loss": self.stop_loss,
                           "start_cash": start_cash, "cost_bps": cost_bps, "history_start": history_start},
                          sort_keys=True)


@dataclass
class SimResult:
    equity: pd.Series                       # value at each close
    cash: pd.Series
    trades: list[tuple] = field(default_factory=list)   # (date, symbol, side, qty, price, notional, reason)
    positions: dict = field(default_factory=dict)       # symbol -> {qty, entry_price, last_price, value}


def load_bots(cfg: dict) -> list[BotSpec]:
    """Read league.bots from the config, with every mistake reported in plain words."""
    league = cfg.get("league") or {}
    universes = cfg.get("universes") or {}
    bots, seen = [], set()
    for entry in league.get("bots", []):
        name = entry["name"]
        if name in seen:
            raise ValueError(f"Two league bots are both called {name!r}.")
        seen.add(name)
        get_strategy(entry["strategy"])                       # raises on an unknown name
        if "symbols" in entry:
            symbols, universe = entry["symbols"], entry.get("universe", "custom")
        elif entry.get("universe") in universes:
            symbols, universe = universes[entry["universe"]], entry["universe"]
        else:
            raise ValueError(f"League bot {name!r}: universe {entry.get('universe')!r} is not listed "
                             f"under `universes` (options: {sorted(universes)}).")
        crypto = [s for s in symbols if is_crypto(s)]
        if crypto:
            raise ValueError(f"League bot {name!r}: crypto is not supported in the league ({', '.join(crypto)}).")
        params = {**cfg.get("strategies", {}).get(entry["strategy"], {}), **entry.get("params", {})}
        bots.append(BotSpec(name, entry["strategy"], universe, tuple(symbols), params,
                            {**cfg["risk"], **entry.get("risk", {})}, bool(entry.get("stop_loss", True))))
    return bots


def league_symbols(cfg: dict) -> list[str]:
    """Every symbol some league bot trades, in a stable order, without repeats."""
    out: dict[str, None] = {}
    for bot in load_bots(cfg):
        out.update(dict.fromkeys(bot.symbols))
    return list(out)


def price_panels(conn, symbols) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(open, close) tables: one row per day, one column per symbol, NaN where there is no bar."""
    bars = db.load_ohlcv(conn, list(symbols))
    if bars.empty:
        empty = pd.DataFrame(columns=list(symbols), dtype=float)
        return empty, empty.copy()
    opens = bars.pivot(index="date", columns="symbol", values="open").reindex(columns=list(symbols))
    closes = bars.pivot(index="date", columns="symbol", values="close").reindex(columns=list(symbols))
    return opens, closes


def target_table(closes: pd.DataFrame, spec: BotSpec) -> pd.DataFrame:
    """The strategy's wish for each symbol after each close (0..1). NaN before a symbol has prices."""
    strategy = get_strategy(spec.strategy)
    out = {}
    for sym in closes.columns:
        own = closes[sym].dropna()
        if own.empty:
            out[sym] = pd.Series(np.nan, index=closes.index)
            continue
        wish = strategy(own, **{**spec.params, "periods_per_year": 252}).clip(0.0, 1.0)
        # A day with no bar for this symbol keeps the previous wish; days before its first bar stay NaN.
        out[sym] = wish.reindex(closes.index).ffill()
    return pd.DataFrame(out, index=closes.index)


def simulate(opens: pd.DataFrame, closes: pd.DataFrame, targets: pd.DataFrame, spec: BotSpec,
             start_cash: float, cost_bps: float, start: str | None = None) -> SimResult:
    """Replay one bot day by day.

    At each close it values the account, applies the stop-loss, and plans orders
    exactly as the real bot does. Those orders fill at the next day's open. An
    order for a symbol with no bar that day lapses, like an unfilled day order.
    """
    symbols = list(closes.columns)
    column = {sym: j for j, sym in enumerate(symbols)}
    days = closes.index if start is None else closes.index[closes.index >= pd.Timestamp(start)]
    opens_, closes_ = opens.reindex(days).to_numpy(dtype=float), closes.reindex(days).to_numpy(dtype=float)
    wishes = targets.reindex(days).to_numpy(dtype=float)
    n = len(symbols)
    cost = cost_bps / 1e4
    stop = stop_loss_pct(spec.risk) if spec.stop_loss else None
    cooldown = int(spec.risk["stop_cooldown_bars"])

    shares, entry, last = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
    blocked_until = np.zeros(n, dtype=int)
    cash = float(start_cash)
    pending: list = []
    stopped: set[str] = set()
    equity_out, cash_out, trades = [], [], []

    for i, day in enumerate(days):
        date = day.strftime("%Y-%m-%d")
        # 1. the orders planned at the last close fill at this open: sells first, so buys never lean on margin
        for order in sorted(pending, key=lambda o: o.side == "buy"):
            j = column[order.symbol]
            price = opens_[i, j]
            if np.isnan(price) or price <= 0:
                continue
            if order.side == "buy":
                spend = min(order.notional, cash / (1 + cost))
                if spend < spec.risk["min_order_notional"]:
                    continue
                qty = spend / price
                held_cost = 0.0 if shares[j] == 0 else shares[j] * entry[j]
                shares[j] += qty
                entry[j] = (held_cost + spend) / shares[j]
                cash -= spend * (1 + cost)
                trades.append((date, order.symbol, "buy", qty, price, spend, None))
            else:
                qty = shares[j] if order.side == "close" else min(shares[j], order.notional / price)
                if qty <= 0:
                    continue
                shares[j] -= qty
                cash += qty * price * (1 - cost)
                if shares[j] < 1e-9:
                    shares[j], entry[j] = 0.0, np.nan
                trades.append((date, order.symbol, "sell", qty, price, qty * price,
                               "stop-loss" if order.symbol in stopped else None))
        # 2. value the account at this close (a symbol with no bar keeps its last price)
        last = np.where(np.isnan(closes_[i]), last, closes_[i])
        values = np.where(shares > 0, shares * last, 0.0)
        equity = cash + float(values.sum())
        equity_out.append(equity)
        cash_out.append(cash)
        # 3. decide what to hold from here, for the next open
        wish = wishes[i].copy()
        stopped = set()
        if stop is not None:
            with np.errstate(invalid="ignore"):
                hit = (shares > 0) & (last / entry - 1 <= -stop)
            for j in np.flatnonzero(hit):
                blocked_until[j] = i + cooldown
                stopped.add(symbols[j])
            wish[hit | (blocked_until > i)] = 0.0
        want = {symbols[j]: float(wish[j]) for j in range(n) if not np.isnan(wish[j])}
        have = {symbols[j]: float(values[j]) for j in range(n) if shares[j] > 0}
        pending = plan_orders(want, have, equity, spec.risk, tolerance=TOLERANCE) if want else []

    positions = {symbols[j]: {"qty": float(shares[j]), "entry_price": float(entry[j]),
                              "last_price": float(last[j]), "value": float(shares[j] * last[j])}
                 for j in range(n) if shares[j] > 0}
    return SimResult(pd.Series(equity_out, index=days, name=spec.name),
                     pd.Series(cash_out, index=days, name=spec.name), trades, positions)


def stats(equity: pd.Series) -> dict:
    """Return, yearly rate (a year or more of history only), worst drop and Sharpe ratio."""
    equity = equity.dropna()
    if len(equity) < 2:
        return {"days": max(len(equity) - 1, 0), "total_return": None, "cagr": None,
                "max_drawdown": None, "sharpe": None}
    returns = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    years = len(returns) / 252
    enough = len(returns) >= 20 and returns.std() > 0
    return {"days": len(returns), "total_return": total,
            "cagr": float((1 + total) ** (1 / years) - 1) if years >= 1 else None,
            "max_drawdown": float((equity / equity.cummax() - 1).min()),
            "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)) if enough else None}


def run_league(conn, cfg: dict, say=print) -> pd.DataFrame:
    """Replay every bot, store the results, and return the leaderboard."""
    league = cfg.get("league") or {}
    bots = load_bots(cfg)
    if not bots:
        say("No league bots are listed in config.yaml (league.bots).")
        return pd.DataFrame()
    start_cash = float(league.get("start_cash", 100_000))
    cost_bps = float(cfg["backtest"]["cost_bps"])
    history_start = str(league.get("history_start", cfg["start_date"]))

    known = {name: (spec, frozen) for name, spec, frozen in
             conn.execute("SELECT name, spec, frozen_on FROM league_bots")}
    for gone in sorted(set(known) - {b.name for b in bots}):
        for table, column in (("league_bots", "name"), ("league_equity", "bot"),
                              ("league_trades", "bot"), ("league_positions", "bot")):
            conn.execute(f"DELETE FROM {table} WHERE {column} = ?", (gone,))
        say(f"{gone}: no longer in the config, removed from the league.")

    for bot in bots:
        opens, closes = price_panels(conn, bot.symbols)
        missing = [s for s in bot.symbols if closes.empty or closes[s].dropna().empty]
        if missing:
            say(f"{bot.name}: no prices for {', '.join(missing)}. Run `python main.py fetch`.")
        if closes.dropna(how="all").empty:
            continue
        latest = closes.index.max().strftime("%Y-%m-%d")
        mark = bot.fingerprint(start_cash, cost_bps, history_start)
        if bot.name not in known:
            frozen = latest
            say(f"{bot.name}: new bot. Its live record starts after {frozen}.")
        elif known[bot.name][0] != mark:
            frozen = latest
            say(f"{bot.name}: settings changed, so its live record restarts after {frozen}.")
        else:
            frozen = known[bot.name][1]
        conn.execute("INSERT OR REPLACE INTO league_bots VALUES (?,?,?,?)", (bot.name, mark, frozen, db.now_utc()))

        result = simulate(opens, closes, target_table(closes, bot), bot, start_cash, cost_bps, history_start)
        conn.execute("DELETE FROM league_equity WHERE bot = ?", (bot.name,))
        conn.executemany("INSERT INTO league_equity VALUES (?,?,?,?)",
                         [(bot.name, d.strftime("%Y-%m-%d"), float(e), float(c))
                          for d, e, c in zip(result.equity.index, result.equity, result.cash)])
        conn.execute("DELETE FROM league_trades WHERE bot = ?", (bot.name,))
        conn.executemany("INSERT INTO league_trades VALUES (?,?,?,?,?,?,?,?)",
                         [(bot.name, *trade) for trade in result.trades])
        conn.execute("DELETE FROM league_positions WHERE bot = ?", (bot.name,))
        conn.executemany("INSERT INTO league_positions VALUES (?,?,?,?,?,?)",
                         [(bot.name, sym, p["qty"], p["entry_price"], p["last_price"], p["value"])
                          for sym, p in result.positions.items()])
        conn.commit()
    conn.commit()
    return leaderboard(conn, cfg)


def equity_table(conn) -> pd.DataFrame:
    """One column per bot, one row per day."""
    rows = pd.read_sql_query("SELECT bot, date, equity FROM league_equity ORDER BY date", conn,
                             parse_dates=["date"])
    return rows.pivot(index="date", columns="bot", values="equity") if len(rows) else pd.DataFrame()


def leaderboard(conn, cfg: dict) -> pd.DataFrame:
    """One row per bot: the live record first, then the backtest history, each next to SPY."""
    equity = equity_table(conn)
    if equity.empty:
        return pd.DataFrame()
    info = {name: (json.loads(spec), frozen) for name, spec, frozen in
            conn.execute("SELECT name, spec, frozen_on FROM league_bots")}
    benchmark = db.load_close(conn, cfg["benchmark"])
    invested = dict(conn.execute("SELECT bot, SUM(value) FROM league_positions GROUP BY bot"))
    order = [b["name"] for b in (cfg.get("league") or {}).get("bots", [])]
    rows = []
    for name in [n for n in order if n in equity.columns] or list(equity.columns):
        series = equity[name].dropna()
        spec, frozen = info.get(name, ({}, None))
        cut = pd.Timestamp(frozen) if frozen else series.index[-1]
        live, history = stats(series[series.index >= cut]), stats(series[series.index <= cut])
        spy = benchmark.reindex(series.index).ffill()
        spy_live, spy_history = stats(spy[spy.index >= cut]), stats(spy[spy.index <= cut])
        rows.append({
            "bot": name, "strategy": spec.get("strategy"), "symbols": len(spec.get("symbols", [])),
            "value": float(series.iloc[-1]),
            "invested": float(invested.get(name, 0.0) / series.iloc[-1]),
            "live_since": frozen, "live_days": live["days"],
            "live_return": live["total_return"], "spy_live_return": spy_live["total_return"],
            "history_cagr": history["cagr"], "spy_history_cagr": spy_history["cagr"],
            "history_sharpe": history["sharpe"], "history_max_drawdown": history["max_drawdown"],
            "spy_history_max_drawdown": spy_history["max_drawdown"],
            "trades": conn.execute("SELECT COUNT(*) FROM league_trades WHERE bot = ?", (name,)).fetchone()[0],
        })
    return pd.DataFrame(rows)
