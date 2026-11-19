"""How much of each rank bot's league result do the stop-loss and the trading costs take?

    python scripts/league_attribution.py [--end 2026-11-13]

Research only. It replays the league's rank bots with the league's own simulator
(tradebot/league.py) three ways: as configured, without the stop-loss, and without
the stop-loss or trading costs. Each is compared with the league's baseline on the
same stocks (`hold-sp500`), including how much of any difference only comes from
swinging more with the market (beta). The model's price database is opened
read-only: nothing is stored, and no bot's live record changes.

--end stops the replay on that day. The leaderboard's history column ends on the
day a bot's settings were first recorded, so the same day here gives numbers that
match it.

What this found on 2026-11-18 is in the README (Bot league).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tradebot.config import load_config  # noqa: E402
from tradebot.league import against, load_bots, simulate, stats, target_table, variants  # noqa: E402
from tradebot.modeldata import load_clean_panels  # noqa: E402
from tradebot.rankmodel import load_scores  # noqa: E402


def read_only(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)


def replay(cfg: dict, conn, end: str | None = None) -> tuple[pd.DataFrame, dict]:
    """One row per rank bot and variant, and the baseline's own numbers."""
    model = cfg["model"]
    bots = load_bots(cfg)
    rank_bots = [b for b in bots if b.model]
    baseline = next((b for b in bots if not b.model and b.universe == model["universe"]), None)
    if not rank_bots or baseline is None:
        raise SystemExit("The league needs rank bots and a buy-and-hold baseline on the model's universe "
                         "(league.bots in config.yaml).")
    panels, _ = load_clean_panels(conn, model.get("break_ratio"))
    if not panels:
        raise SystemExit("No prices for the model's universe. Run `python main.py model-fetch`.")
    opens, closes = panels["open"].loc[:end], panels["close"].loc[:end]
    league = cfg.get("league") or {}
    cash = float(league.get("start_cash", 100_000))
    cost = float(cfg["backtest"]["cost_bps"])
    history_start = str(league.get("history_start", cfg.get("start_date")))

    def run(spec, targets, cost_bps):
        columns = list(spec.symbols)
        return simulate(opens.reindex(columns=columns), closes.reindex(columns=columns), targets, spec,
                        cash, cost_bps, spec.start or history_start)

    base = run(baseline, target_table(closes.reindex(columns=list(baseline.symbols)), baseline), cost)
    base_returns = base.equity.pct_change().dropna()
    scores, rows = {}, []
    for bot in rank_bots:
        if bot.model not in scores:
            scores[bot.model] = load_scores(conn, bot.model).loc[:end]
        for variant, (spec, cost_bps) in variants(bot, cost).items():
            result = run(spec, scores[bot.model], cost_bps)
            numbers = stats(result.equity)
            rows.append({"bot": bot.name, "variant": variant, "total_return": numbers["total_return"],
                         "cagr": numbers["cagr"], "sharpe": numbers["sharpe"], "worst_drop": numbers["max_drawdown"],
                         "stop_outs": sum(1 for trade in result.trades if trade[6] == "stop-loss"),
                         **against(result.equity.pct_change().dropna(), base_returns)})
    return pd.DataFrame(rows), {"name": baseline.name, **stats(base.equity),
                                "first": base.equity.index[0], "last": base.equity.index[-1]}


def show(table: pd.DataFrame, base: dict) -> None:
    pct = lambda v: "n/a" if v is None or pd.isna(v) else f"{v:+.1%}"  # noqa: E731
    num = lambda v: "n/a" if v is None or pd.isna(v) else f"{v:.2f}"  # noqa: E731
    print(f"Rank bots against {base['name']}, {base['first']:%Y-%m-%d} to {base['last']:%Y-%m-%d}\n")
    print(f"{'bot':<23}{'replayed':<18}{'per year':>9}{'Sharpe':>8}{'worst drop':>12}{'stop-outs':>11}"
          f"{'beta':>7}{'after beta':>12}{'t':>7}")
    for row in table.itertuples():
        print(f"{row.bot:<23}{row.variant:<18}{pct(row.cagr):>9}{num(row.sharpe):>8}{pct(row.worst_drop):>12}"
              f"{row.stop_outs:>11}{num(row.beta):>7}{pct(row.alpha):>12}{row.t_stat:>+7.2f}")
    print(f"{base['name']:<23}{'as set':<18}{pct(base['cagr']):>9}{num(base['sharpe']):>8}"
          f"{pct(base['max_drawdown']):>12}")
    print(f"\nworst drop: the largest fall from a peak. beta: how much the bot moved when {base['name']} moved 1%. "
          f"after beta: the yearly return left once beta times {base['name']}'s return is taken out. "
          "A t below 2 means that part could be luck.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--end", help="last day of the replay (YYYY-MM-DD); default: the latest stored price")
    args = parser.parse_args(argv)
    cfg = load_config()
    table, base = replay(cfg, read_only(cfg["model"]["prices_path"]), args.end)
    show(table, base)


if __name__ == "__main__":
    main()
