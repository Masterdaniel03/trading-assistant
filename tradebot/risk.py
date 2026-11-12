"""Risk limits applied before any order reaches a broker."""
from __future__ import annotations



class RiskViolation(Exception):
    pass


def target_notional(equity: float, target_fraction: float, n_symbols: int, risk: dict) -> float:
    """Dollar target for one symbol: equal-weight, capped by max_position_pct."""
    per_symbol_cap = equity * risk["max_position_pct"]
    equal_weight = equity * risk["max_total_exposure_pct"] / max(n_symbols, 1)
    return max(0.0, min(per_symbol_cap, equal_weight) * target_fraction)


def check_daily_loss(equity: float, last_equity: float, risk: dict) -> None:
    """last_equity = equity at previous close (Alpaca: account.last_equity)."""
    if last_equity and (equity / last_equity - 1) <= -risk["daily_loss_limit_pct"]:
        raise RiskViolation(
            f"Daily loss limit hit ({equity / last_equity - 1:.2%}); no new buys today."
        )


def stop_loss_pct(risk: dict) -> float:
    """Loss from entry that uses up the per-position risk budget at full size."""
    return risk["max_risk_per_position_pct"] / risk["max_position_pct"]


def triggered_stops(returns: dict[str, float], risk: dict) -> dict[str, float]:
    """Symbols whose return from entry has hit the stop. returns: symbol -> e.g. -0.08."""
    limit = -stop_loss_pct(risk)
    return {sym: r for sym, r in returns.items() if r <= limit}


def check_data_fresh(last_dates: dict[str, str | None], expected: str | None) -> None:
    """Refuse to trade unless every symbol has the latest finished session's close.

    last_dates: symbol -> latest stored price date (YYYY-MM-DD) or None.
    expected:   the exchange's most recent session whose daily bar is final,
                taken from the broker's calendar (so weekends, holidays and the
                machine's own timezone play no part).

    Without this, a failed or lagging `fetch` would silently trade on an older
    signal, and a symbol with no data at all would be treated as 'target 0'
    and sold.
    """
    if expected is None:
        raise RiskViolation("Could not tell which session's prices should be available.")
    problems = []
    for sym, d in last_dates.items():
        if d is None:
            problems.append(f"{sym} (no data)")
        elif d < expected:
            problems.append(f"{sym} (has {d}, needs {expected})")
    if problems:
        raise RiskViolation(
            f"Stale or missing prices: {', '.join(problems)}. Run `python main.py fetch` first."
        )


def check_live_allowed(paper: bool, risk: dict) -> None:
    if not paper and not risk.get("allow_live", False):
        raise RiskViolation("Live trading is disabled (risk.allow_live = false).")
