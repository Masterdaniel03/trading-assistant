"""Strategies turn a price series into a *target position* series.

Convention: target[t] in [0, 1] is the position you want to hold AFTER
seeing the close on day t. The backtester applies it starting on t+1,
so a strategy can never trade on information it didn't have yet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(close: pd.Series, **_) -> pd.Series:
    return pd.Series(1.0, index=close.index)


def sma_crossover(close: pd.Series, fast: int = 50, slow: int = 200, **_) -> pd.Series:
    """Long when the fast moving average is above the slow one."""
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    target = (fast_ma > slow_ma).astype(float)
    target[slow_ma.isna()] = 0.0
    return target


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def rsi_reversion(close: pd.Series, period: int = 14, buy_below: float = 30,
                  sell_above: float = 55, **_) -> pd.Series:
    """Buy oversold dips, exit once RSI recovers (stateful)."""
    r = rsi(close, period).to_numpy()
    target = np.zeros(len(r))
    holding = 0.0
    for i, v in enumerate(r):
        if holding == 0 and v < buy_below:
            holding = 1.0
        elif holding == 1 and v > sell_above:
            holding = 0.0
        target[i] = holding
    return pd.Series(target, index=close.index)


def momentum(close: pd.Series, lookback: int = 252, skip: int = 21, **_) -> pd.Series:
    """Time-series momentum: long when the return from `lookback` bars ago to
    `skip` bars ago is positive. Skipping the latest month avoids short-term reversal."""
    return (close.shift(skip) > close.shift(lookback)).astype(float)


def vol_target(close: pd.Series, target_vol: float = 0.15, window: int = 20,
               periods_per_year: int = 252, **_) -> pd.Series:
    """Hold more when markets are calm: position = target_vol / realized vol,
    capped at 1 (no leverage). Flat until `window` returns are available."""
    realized = close.pct_change(fill_method=None).rolling(window).std() * np.sqrt(periods_per_year)
    return (target_vol / realized).clip(upper=1.0).fillna(0.0)


STRATEGIES = {
    "buy_and_hold": buy_and_hold,
    "sma_crossover": sma_crossover,
    "rsi_reversion": rsi_reversion,
    "momentum": momentum,
    "vol_target": vol_target,
}


def get_strategy(name: str):
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}. Options: {list(STRATEGIES)}")
    return STRATEGIES[name]
