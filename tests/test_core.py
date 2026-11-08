import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradebot import db
from tradebot.backtest import run_backtest
from tradebot.rebalance import plan_orders
from tradebot.risk import RiskViolation, check_daily_loss, check_live_allowed
from tradebot.strategies import buy_and_hold, rsi_reversion, sma_crossover

RISK = {"max_position_pct": 0.2, "max_total_exposure_pct": 1.0,
        "daily_loss_limit_pct": 0.03, "min_order_notional": 1.0, "allow_live": False}


@pytest.fixture
def prices():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=600)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx)))), index=idx)


def test_buy_and_hold_matches_price_return(prices):
    r = run_backtest(prices, buy_and_hold(prices), cost_bps=0)
    # day 0 is the entry day; held from day 1 on
    assert r.metrics["total_return"] == pytest.approx(prices.iloc[-1] / prices.iloc[0] - 1, rel=1e-9)


def test_no_lookahead(prices):
    # Signal built from TODAY's close ("today was an up day"). If the engine
    # wrongly credited it with today's return it would capture every up day.
    up_today = (prices.pct_change() > 0).astype(float)
    perfect = (prices.pct_change().clip(lower=0) + 1).prod() - 1
    r = run_backtest(prices, up_today, cost_bps=0)
    assert r.metrics["total_return"] < perfect * 0.5


def test_costs_reduce_returns(prices):
    target = sma_crossover(prices, fast=10, slow=30)
    free = run_backtest(prices, target, cost_bps=0)
    costly = run_backtest(prices, target, cost_bps=50)
    assert costly.metrics["final_equity"] < free.metrics["final_equity"]


def test_strategies_are_bounded(prices):
    for t in (sma_crossover(prices, 10, 30), rsi_reversion(prices)):
        assert set(t.unique()) <= {0.0, 1.0}
        assert len(t) == len(prices)


def test_plan_orders_caps_position_size():
    orders = plan_orders({"AAPL": 1.0, "MSFT": 0.0}, {"MSFT": 500.0}, equity=4000, risk=RISK)
    by = {o.symbol: o for o in orders}
    assert by["AAPL"].side == "buy" and by["AAPL"].notional == pytest.approx(800)  # 20% cap
    assert by["MSFT"].side == "close"


def test_plan_orders_skips_small_gaps():
    assert plan_orders({"AAPL": 1.0}, {"AAPL": 790.0}, equity=4000, risk=RISK) == []


def test_risk_guards():
    with pytest.raises(RiskViolation):
        check_daily_loss(equity=960, last_equity=1000, risk=RISK)
    check_daily_loss(equity=990, last_equity=1000, risk=RISK)
    with pytest.raises(RiskViolation):
        check_live_allowed(paper=False, risk=RISK)


def test_db_roundtrip(tmp_path, prices):
    conn = db.connect(str(tmp_path / "t.db"))
    db.init_db(conn)
    df = pd.DataFrame({"symbol": "TEST", "date": prices.index.strftime("%Y-%m-%d"),
                       "open": prices.values, "high": prices.values, "low": prices.values,
                       "close": prices.values, "volume": 1.0})
    db.upsert_prices(conn, df)
    assert len(db.load_close(conn, "TEST")) == len(prices)
    db.log_trade(conn, "fidelity_manual", "aapl", "buy", qty=1, price=200, followed_ai=1)
    assert db.trades_df(conn).iloc[0]["symbol"] == "AAPL"
