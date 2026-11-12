"""Copy what the dashboard needs from the broker into the database.

The dashboard reads only the database, so it opens instantly, works offline
and never handles API keys on its own. This is the one place that refreshes
the broker-side numbers. It runs on every `paper` run and from the
dashboard's "Refresh from Alpaca" button.
"""
from __future__ import annotations

from tradebot import db


def sync_account(conn, broker, positions: dict | None = None) -> dict:
    """Store the daily account value and the current holdings. Returns a short summary.

    `positions` can be passed in when the caller has just fetched them.
    """
    account = broker.account_name
    history = broker.equity_history()
    db.save_equity_daily(conn, account, history)
    positions = broker.position_details() if positions is None else positions
    db.save_positions(conn, account, positions)
    return {"account": account, "days": len(history), "positions": len(positions)}


def refresh_from_broker(conn, broker) -> dict:
    """Everything the dashboard's refresh button does: a balance snapshot, the
    outcome of orders sent earlier, then the sync. Sends no orders."""
    acct = broker.account()
    db.log_equity(conn, broker.account_name, acct["equity"], acct["cash"])
    updated = 0
    for trade_id, _ts, _symbol, _side, order_id in db.open_trades(conn, broker.account_name):
        info = broker.order_info(order_id)
        if info is not None:
            db.update_trade(conn, trade_id, info.status, info.filled_qty, info.filled_avg_price)
            updated += 1
    return {**sync_account(conn, broker), "equity": acct["equity"], "orders_checked": updated}
