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
            f"Daily loss limit hit ({equity / last_equity - 1:.2%}); no new orders today."
        )


def check_live_allowed(paper: bool, risk: dict) -> None:
    if not paper and not risk.get("allow_live", False):
        raise RiskViolation("Live trading is disabled (risk.allow_live = false).")
