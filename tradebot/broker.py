"""Thin wrapper around Alpaca's trading API (paper by default).

All calls are retried with backoff (see tradebot.retry). Order placement is
made idempotent with a client_order_id: if a submit fails ambiguously (timeout,
5xx) we look the order up before sending it again, so a flaky network can't
double-buy.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from tradebot.retry import call_with_retry, is_rate_limit, is_transient

log = logging.getLogger(__name__)


def disable_builtin_retry(client) -> None:
    """alpaca-py retries 429 *and 504* itself with a fixed wait, which also
    blindly re-POSTs orders on 504. Turn it off so tradebot.retry is the only
    retry layer (exponential backoff, Retry-After, idempotent orders)."""
    if hasattr(client, "_retry"):
        client._retry = 0


def _status(exc) -> int | None:
    return getattr(exc, "status_code", None)


@dataclass(frozen=True)
class PositionInfo:
    market_value: float
    unrealized_plpc: float  # return from average entry, e.g. -0.08


@dataclass(frozen=True)
class OrderInfo:
    status: str                      # new, accepted, filled, canceled, ...
    filled_qty: float
    filled_avg_price: float | None


def enum_text(value) -> str | None:
    """'filled' from OrderStatus.FILLED; plain strings pass through."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


class AlpacaBroker:
    def __init__(self, key: str, secret: str, paper: bool = True):
        if not key or not secret:
            raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")
        from alpaca.trading.client import TradingClient
        self.paper = paper
        self.client = TradingClient(key, secret, paper=paper)
        disable_builtin_retry(self.client)

    @property
    def account_name(self) -> str:
        return "alpaca_paper" if self.paper else "alpaca_live"

    # ---------------------------------------------------------------- reads

    def account(self) -> dict:
        a = call_with_retry(self.client.get_account, what="alpaca account")
        return {"equity": float(a.equity), "last_equity": float(a.last_equity),
                "cash": float(a.cash), "status": str(a.status)}

    def positions(self) -> dict[str, float]:
        """symbol -> current market value in $."""
        return {s: p.market_value for s, p in self.position_details().items()}

    def position_details(self) -> dict[str, PositionInfo]:
        ps = call_with_retry(self.client.get_all_positions, what="alpaca positions")
        return {p.symbol: PositionInfo(float(p.market_value), float(p.unrealized_plpc or 0))
                for p in ps}

    def is_market_open(self) -> bool:
        return bool(call_with_retry(self.client.get_clock, what="alpaca clock").is_open)

    def open_order_symbols(self) -> set[str]:
        """Symbols with an order that is still waiting to fill.

        Orders sent while the market is closed sit here until the open, and the
        position still reads $0 meanwhile, so planning from positions alone
        would send the same order again."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = call_with_retry(self.client.get_orders,
                                 GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500),
                                 what="alpaca open orders")
        return {o.symbol for o in orders}

    def order_info(self, order_id: str) -> OrderInfo | None:
        """Current state of one order, or None if Alpaca no longer knows it."""
        def lookup():
            try:
                return self.client.get_order_by_id(order_id)
            except Exception as exc:  # noqa: BLE001
                if _status(exc) == 404:
                    return None
                raise

        o = call_with_retry(lookup, what="alpaca order status")
        if o is None:
            return None
        price = float(o.filled_avg_price) if o.filled_avg_price else None
        return OrderInfo(enum_text(o.status) or "unknown", float(o.filled_qty or 0), price)

    def _order_by_client_id(self, client_order_id: str):
        try:
            return self.client.get_order_by_client_id(client_order_id)
        except Exception as exc:  # noqa: BLE001
            if _status(exc) == 404:
                return None
            raise

    # ---------------------------------------------------------------- writes

    def submit_notional(self, symbol: str, side: str, notional: float,
                        client_order_id: str | None = None):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        coid = client_order_id or f"tb-{uuid.uuid4().hex[:24]}"
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(notional, 2),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=coid,
        )

        def attempt():
            try:
                return self.client.submit_order(req)
            except Exception as exc:  # noqa: BLE001
                # 429 = rejected before processing, safe to resend as-is.
                # Anything else that might have reached the server: check first.
                if not is_rate_limit(exc) and (is_transient(exc) or _status(exc) == 422):
                    existing = call_with_retry(self._order_by_client_id, coid,
                                               what="alpaca order lookup")
                    if existing is not None:
                        log.info("order %s already accepted (%s); not resending", coid, existing.id)
                        return existing
                raise

        return call_with_retry(attempt, what=f"alpaca submit {side} {symbol}")

    def close_position(self, symbol: str):
        def attempt():
            try:
                return self.client.close_position(symbol)
            except Exception as exc:  # noqa: BLE001
                if not is_rate_limit(exc) and is_transient(exc):
                    # The close may have gone through; don't send a second one.
                    pending = call_with_retry(self._close_in_flight, symbol,
                                              what="alpaca close check")
                    if pending:
                        log.info("close for %s already accepted; not resending", symbol)
                        return pending
                raise

        return call_with_retry(attempt, what=f"alpaca close {symbol}")

    def _close_in_flight(self, symbol: str):
        """Return an open *sell* order for `symbol` (or True if the position
        is already gone), else None. A pending buy must not count as a close."""
        from alpaca.trading.enums import OrderSide, QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        if symbol not in {p.symbol for p in self.client.get_all_positions()}:
            return True
        orders = self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN,
                                                         symbols=[symbol]))
        return next((o for o in orders if o.side == OrderSide.SELL), None)
