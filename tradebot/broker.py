"""Thin wrapper around Alpaca's trading API (paper by default)."""
from __future__ import annotations


class AlpacaBroker:
    def __init__(self, key: str, secret: str, paper: bool = True):
        if not key or not secret:
            raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")
        from alpaca.trading.client import TradingClient
        self.paper = paper
        self.client = TradingClient(key, secret, paper=paper)

    @property
    def account_name(self) -> str:
        return "alpaca_paper" if self.paper else "alpaca_live"

    def account(self) -> dict:
        a = self.client.get_account()
        return {"equity": float(a.equity), "last_equity": float(a.last_equity),
                "cash": float(a.cash), "status": str(a.status)}

    def positions(self) -> dict[str, float]:
        """symbol -> current market value in $."""
        return {p.symbol: float(p.market_value) for p in self.client.get_all_positions()}

    def is_market_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def submit_notional(self, symbol: str, side: str, notional: float):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(notional, 2),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(req)

    def close_position(self, symbol: str):
        return self.client.close_position(symbol)
