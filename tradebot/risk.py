"""Risk limits applied before any order reaches a broker."""
from __future__ import annotations

from datetime import date

DEFAULT_MAX_DATA_AGE_DAYS = 4  # Fri close -> Tue morning after a Monday holiday


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
            f"Daily loss limit hit ({equity / last_equity - 1:.2%}); no new orders today."
        )


def stop_loss_pct(risk: dict) -> float:
    """Loss from entry that uses up the per-position risk budget at full size."""
    return risk["max_risk_per_position_pct"] / risk["max_position_pct"]


def triggered_stops(returns: dict[str, float], risk: dict) -> dict[str, float]:
    """Symbols whose return from entry has hit the stop. returns: symbol -> e.g. -0.08."""
    limit = -stop_loss_pct(risk)
    return {sym: r for sym, r in returns.items() if r <= limit}


def check_data_fresh(last_dates: dict[str, str | None], today: date, risk: dict) -> None:
    """Refuse to trade on missing or stale prices.

    last_dates: symbol -> latest stored price date (YYYY-MM-DD) or None.
    Without this, a failed `fetch` would silently trade on old signals, and a
    symbol with no data at all would be treated as 'target 0' and sold.
    """
    max_age = risk.get("max_data_age_days", DEFAULT_MAX_DATA_AGE_DAYS)
    problems = []
    for sym, d in last_dates.items():
        if d is None:
            problems.append(f"{sym} (no data)")
            continue
        age = (today - date.fromisoformat(d)).days
        if age > max_age:
            problems.append(f"{sym} (last close {d}, {age} days old)")
    if problems:
        raise RiskViolation(
            f"Stale or missing prices: {', '.join(problems)}. Run `python main.py fetch` first."
        )


def check_live_allowed(paper: bool, risk: dict) -> None:
    if not paper and not risk.get("allow_live", False):
        raise RiskViolation("Live trading is disabled (risk.allow_live = false).")
