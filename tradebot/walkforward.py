"""Walk-forward (out-of-sample) testing.

For each fold: pick the parameters with the best Sharpe on a training window,
then trade them on the following test window. Only the test windows are
stitched into the reported result, so every number comes from data the
parameters were not chosen on.

A strategy only ever receives prices up to the end of the window it is being
evaluated on, so a strategy that peeks ahead can't see past the fold.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .strategies import buy_and_hold


@dataclass(frozen=True)
class Fold:
    train_start: int
    test_start: int
    test_end: int  # exclusive


@dataclass
class WalkForwardResult:
    oos: BacktestResult
    benchmark: BacktestResult
    folds: list[dict] = field(default_factory=list)

    @property
    def mean_train_sharpe(self) -> float:
        return float(pd.Series([f["train_sharpe"] for f in self.folds]).mean())

    @property
    def folds_beating_benchmark(self) -> int:
        return sum(f["test_return"] > f["benchmark_return"] for f in self.folds)


def make_folds(n: int, train_bars: int, test_bars: int) -> list[Fold]:
    """Back-to-back test windows anchored to the end of the data, so the most
    recent bars are always tested. Any remainder is dropped from the start."""
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError("train_bars and test_bars must be positive")
    count = (n - train_bars) // test_bars
    if count < 1:
        raise ValueError(f"Need at least {train_bars + test_bars} bars, have {n}")
    first = n - train_bars - count * test_bars
    return [Fold(first + i * test_bars, first + i * test_bars + train_bars,
                 first + (i + 1) * test_bars + train_bars) for i in range(count)]


def expand_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def walk_forward(close: pd.Series, strategy: Callable, grid: dict[str, list], *,
                 train_bars: int, test_bars: int, name: str = "walk_forward",
                 initial_capital: float = 4000.0, cost_bps: float = 5.0,
                 periods_per_year: int = 252) -> WalkForwardResult:
    close = close.dropna().astype(float)
    combos = expand_grid(grid)
    costs = {"initial_capital": initial_capital, "cost_bps": cost_bps,
             "periods_per_year": periods_per_year}
    folds = make_folds(len(close), train_bars, test_bars)

    pieces, records = [], []
    for fold in folds:
        history = close.iloc[:fold.test_start]
        best, best_sharpe = combos[0], float("-inf")
        for params in combos:
            target = strategy(history, **params, periods_per_year=periods_per_year)
            sharpe = run_backtest(history.iloc[fold.train_start:], target, **costs).metrics["sharpe"]
            if pd.notna(sharpe) and sharpe > best_sharpe:
                best, best_sharpe = params, sharpe

        visible = close.iloc[:fold.test_end]
        target = strategy(visible, **best, periods_per_year=periods_per_year)
        pieces.append(target.iloc[fold.test_start:])
        records.append({"train_start": close.index[fold.train_start],
                        "test_start": close.index[fold.test_start],
                        "test_end": close.index[fold.test_end - 1],
                        "params": best, "train_sharpe": best_sharpe})

    oos_close = close.iloc[folds[0].test_start:]
    oos = run_backtest(oos_close, pd.concat(pieces), name=name, **costs)
    benchmark = run_backtest(oos_close, buy_and_hold(oos_close), name="buy_and_hold", **costs)

    for rec in records:
        span = rec["test_start"], rec["test_end"]
        rec["test_return"] = float((1 + oos.returns.loc[span[0]:span[1]]).prod() - 1)
        rec["benchmark_return"] = float((1 + benchmark.returns.loc[span[0]:span[1]]).prod() - 1)
    return WalkForwardResult(oos, benchmark, records)
