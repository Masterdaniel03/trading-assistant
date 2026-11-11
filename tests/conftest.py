"""Shared test helpers: a fake broker and a ready-to-run paper bot."""
import argparse
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

import main
from tradebot import db

RISK = {"max_position_pct": 0.20, "max_total_exposure_pct": 1.0, "max_risk_per_position_pct": 0.015,
        "stop_cooldown_bars": 2, "daily_loss_limit_pct": 0.03, "min_order_notional": 1.0,
        "max_data_age_days": 4, "allow_live": False}


@pytest.fixture
def risk():
    return dict(RISK)


class FakeBroker:
    """Stands in for AlpacaBroker. Class attributes are the 'account state' a test sets up."""
    account_name = "alpaca_paper"
    equity = 100_000.0
    held: dict = {}
    open_symbols: set = set()     # symbols that still have an unfilled order
    order_updates: dict = {}      # order_id -> what the broker now says about it
    fail_on: set = set()          # symbols whose orders the broker rejects
    market_open = True
    last = None

    def __init__(self, *args, **kwargs):
        self.closed, self.bought, self.sold, self.sequence = [], [], [], []
        FakeBroker.last = self

    def account(self):
        return {"equity": FakeBroker.equity, "last_equity": 100_000.0, "cash": 0.0, "status": "ACTIVE"}

    def position_details(self):
        return FakeBroker.held

    def open_order_symbols(self):
        return set(FakeBroker.open_symbols)

    def order_info(self, order_id):
        return FakeBroker.order_updates.get(order_id)

    def is_market_open(self):
        return FakeBroker.market_open

    def _send(self, kind, symbol):
        if symbol in FakeBroker.fail_on:
            raise RuntimeError(f"{symbol} rejected")
        {"close": self.closed, "buy": self.bought, "sell": self.sold}[kind].append(symbol)
        self.sequence.append((kind, symbol))
        return SimpleNamespace(id=f"{kind}-{symbol}", status="accepted")

    def close_position(self, symbol):
        return self._send("close", symbol)

    def submit_notional(self, symbol, side, notional):
        return self._send(side, symbol)


@pytest.fixture
def bot(tmp_path, monkeypatch):
    """A paper bot wired to FakeBroker, a temp database and a temp log folder."""
    import tradebot.broker
    monkeypatch.setattr(tradebot.broker, "AlpacaBroker", FakeBroker)
    monkeypatch.setattr(main, "load_secrets",
                        lambda: {"alpaca_key": "k", "alpaca_secret": "s", "alpaca_paper": True})
    monkeypatch.setattr(main, "LOGS", tmp_path / "logs", raising=False)
    cfg = {"db_path": str(tmp_path / "t.db"), "risk": dict(RISK), "strategies": {},
           "paper_bot": {"strategy": "buy_and_hold", "symbols": ["AAA", "BBB"]}}
    conn = db.connect(cfg["db_path"])
    db.init_db(conn)
    FakeBroker.equity = 100_000.0
    FakeBroker.held, FakeBroker.open_symbols = {}, set()
    FakeBroker.order_updates, FakeBroker.fail_on = {}, set()
    FakeBroker.market_open = True

    def store(symbol, days_old=0, n=5):
        last = date.today() - timedelta(days=days_old)
        dates = [(last - timedelta(days=i)).isoformat() for i in range(n)]
        db.upsert_prices(conn, pd.DataFrame({"symbol": symbol, "date": dates, "open": 1.0, "high": 1.0,
                                             "low": 1.0, "close": 1.0, "volume": 1.0}))

    def run(execute=True):
        main.cmd_paper(cfg, argparse.Namespace(execute=execute))
        return FakeBroker.last

    def log_text():
        path = tmp_path / "logs" / "paper_trading.log"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    return SimpleNamespace(cfg=cfg, conn=conn, store=store, run=run, log_text=log_text,
                           broker=FakeBroker)
