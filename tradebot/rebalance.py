"""Turn target positions into a list of orders (pure logic, easy to test)."""
from __future__ import annotations

from dataclasses import dataclass

from .risk import target_notional


@dataclass
class PlannedOrder:
    symbol: str
    side: str          # buy | sell | close
    notional: float
    target_value: float
    current_value: float


def plan_orders(targets: dict[str, float], current: dict[str, float], equity: float,
                risk: dict, tolerance: float = 0.10,
                close_unlisted: bool = False) -> list[PlannedOrder]:
    """targets: symbol -> fraction 0..1; current: symbol -> $ value held.

    Only trades when the gap exceeds `tolerance` of the target (avoids churn).
    With close_unlisted, positions in `current` that aren't in `targets` are closed.
    """
    orders: list[PlannedOrder] = []
    n = len(targets)
    for sym, frac in targets.items():
        want = target_notional(equity, frac, n, risk)
        have = current.get(sym, 0.0)
        gap = want - have
        if want == 0 and have > 0:
            orders.append(PlannedOrder(sym, "close", have, want, have))
        elif abs(gap) >= max(risk["min_order_notional"], tolerance * max(want, 1e-9)):
            orders.append(PlannedOrder(sym, "buy" if gap > 0 else "sell", abs(gap), want, have))
    if close_unlisted:
        # Held but no longer managed (e.g. removed from paper_bot.symbols).
        for sym, have in current.items():
            if sym not in targets and have > 0:
                orders.append(PlannedOrder(sym, "close", have, 0.0, have))
    return orders
