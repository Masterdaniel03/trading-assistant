"""Vectorized daily backtester with transaction costs and no look-ahead."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    metrics: dict = field(default_factory=dict)


def apply_stop_loss(close: pd.Series, target: pd.Series, stop_pct: float,
                    cooldown_bars: int = 20) -> pd.Series:
    """Exit when the close falls stop_pct below the entry close, then stay out.

    Checked on closes only, like the bot, so a loss can exceed stop_pct on a gap.
    Each decision uses prices up to that bar, so this adds no look-ahead.
    """
    prices = close.to_numpy(dtype=float)
    out = target.reindex(close.index).fillna(0.0).to_numpy(dtype=float).copy()
    entry, blocked_until = np.nan, 0
    for i, price in enumerate(prices):
        if i < blocked_until:
            out[i], entry = 0.0, np.nan
        elif out[i] <= 0:
            entry = np.nan
        elif np.isnan(entry):
            entry = price
        elif price <= entry * (1 - stop_pct):
            out[i], entry, blocked_until = 0.0, np.nan, i + cooldown_bars
    return pd.Series(out, index=close.index)


def with_stop_loss(strategy, stop_pct: float, cooldown_bars: int = 20):
    """Wrap a strategy function so its targets include the stop-loss."""
    def wrapped(close: pd.Series, **params) -> pd.Series:
        return apply_stop_loss(close, strategy(close, **params), stop_pct, cooldown_bars)
    return wrapped


def run_backtest(close: pd.Series, target: pd.Series, *, name: str = "strategy",
                 initial_capital: float = 4000.0, cost_bps: float = 5.0,
                 periods_per_year: int = 252) -> BacktestResult:
    close = close.dropna().astype(float)
    target = target.reindex(close.index).fillna(0.0).clip(0.0, 1.0)

    # Decide at close t, hold from t -> t+1. This shift is what prevents look-ahead.
    position = target.shift(1).fillna(0.0)
    asset_ret = close.pct_change().fillna(0.0)

    turnover = position.diff().abs()
    turnover.iloc[0] = position.iloc[0]
    costs = turnover * cost_bps / 1e4

    returns = position * asset_ret - costs
    equity = initial_capital * (1.0 + returns).cumprod()

    metrics = compute_metrics(returns, equity, position, turnover, periods_per_year)
    return BacktestResult(name, equity, returns, position, metrics)


def compute_metrics(returns: pd.Series, equity: pd.Series, position: pd.Series,
                    turnover: pd.Series, periods_per_year: int) -> dict:
    n = len(returns)
    years = n / periods_per_year if n else np.nan
    start_capital = equity.iloc[0] / (1 + returns.iloc[0])
    total = equity.iloc[-1] / start_capital - 1
    cagr = (1 + total) ** (1 / years) - 1 if years and years > 0 else np.nan
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year) / vol if vol > 0 else np.nan  # rf = 0
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "trades": int((turnover > 0).sum()),
        "exposure": position.mean(),
        "final_equity": equity.iloc[-1],
    }


def compare(results: list[BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame({r.name: r.metrics for r in results}).T


def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["total_return", "cagr", "ann_vol", "max_drawdown", "exposure"]:
        out[c] = out[c].map(lambda x: f"{x:+.1%}" if c != "ann_vol" and c != "exposure" else f"{x:.1%}")
    out["sharpe"] = out["sharpe"].map(lambda x: f"{x:.2f}")
    out["trades"] = out["trades"].map(lambda x: f"{x:.0f}")
    out["final_equity"] = out["final_equity"].map(lambda x: f"${x:,.0f}")
    return out


def plot_equity(results: list[BacktestResult], path: str, title: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for r in results:
        ax1.plot(r.equity.index, r.equity.values, label=r.name, linewidth=1.4)
        dd = r.equity / r.equity.cummax() - 1
        ax2.plot(dd.index, dd.values * 100, linewidth=1)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(title)
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.2)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.2)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
