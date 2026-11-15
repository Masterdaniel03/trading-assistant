"""Features: price and volume history turned into numbers a model can rank stocks by.

Every feature for day t uses bars up to and including day t and nothing later
(only backward-looking shifts and rolling windows). tests/test_features.py
checks this by cutting the data off after a day and confirming that day's
features do not change.

Input and output are "panels": tables with one row per day and one column
per symbol, NaN where a symbol has no bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# name -> (group, what it measures). The order here is the column order everywhere.
FEATURES = {
    # trend: where the price is, compared with its own past
    "ret_1d": ("trend", "return over the last day"),
    "ret_5d": ("trend", "return over the last week"),
    "ret_21d": ("trend", "return over the last month"),
    "ret_63d": ("trend", "return over the last three months"),
    "mom_12_1": ("trend", "return over the last year, leaving out the latest month"),
    "dist_high_252": ("trend", "how far below its one-year high"),
    "ma_gap_50": ("trend", "price against its 50-day average"),
    "ma_gap_200": ("trend", "price against its 200-day average"),
    "rsi_14": ("trend", "RSI over 14 days"),
    # risk: how much it moves
    "vol_21d": ("risk", "day-to-day swings over the last month"),
    "vol_63d": ("risk", "day-to-day swings over the last three months"),
    "max_ret_21d": ("risk", "biggest one-day gain in the last month"),
    "range_21d": ("risk", "average daily high-to-low range over the last month"),
    # candle shape: what the daily candlesticks looked like
    "close_loc_5d": ("candle shape", "where the close sat between the day's low and high, last week"),
    "body_5d": ("candle shape", "candle body (close minus open) as a share of the day's range, last week"),
    "upper_shadow_5d": ("candle shape", "upper shadow as a share of the day's range, last week"),
    "lower_shadow_5d": ("candle shape", "lower shadow as a share of the day's range, last week"),
    "overnight_21d": ("candle shape", "return earned overnight (close to next open), last month"),
    "intraday_21d": ("candle shape", "return earned during trading hours (open to close), last month"),
    # volume and money flow
    "volume_surge": ("volume and money flow", "last week's volume against the three-month average"),
    "dollar_volume": ("volume and money flow", "dollars traded per day (how big and liquid the stock is)"),
    "money_flow_14": ("volume and money flow", "money flow index: share of dollar volume on rising days"),
    "chaikin_flow_21": ("volume and money flow", "Chaikin money flow: volume weighted by where each day closed"),
    "obv_21": ("volume and money flow", "on-balance volume: up-day volume minus down-day volume, last month"),
    "illiquidity_21": ("volume and money flow", "price move per dollar traded (Amihud)"),
}
NAMES = list(FEATURES)


def compute(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """All features, each as a (days x symbols) table of raw values."""
    o, h, l, c, v = (panels[k].astype(float) for k in ("open", "high", "low", "close", "volume"))  # noqa: E741
    ret = c / c.shift(1) - 1
    spread = (h - l).where(h > l)                     # a day with no range has no candle shape
    dollars = c * v
    f: dict[str, pd.DataFrame] = {}

    f["ret_1d"] = ret
    f["ret_5d"] = c / c.shift(5) - 1
    f["ret_21d"] = c / c.shift(21) - 1
    f["ret_63d"] = c / c.shift(63) - 1
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    f["dist_high_252"] = c / h.rolling(252, min_periods=126).max() - 1
    f["ma_gap_50"] = c / c.rolling(50).mean() - 1
    f["ma_gap_200"] = c / c.rolling(200).mean() - 1
    change = c.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss.where(loss > 0))

    f["vol_21d"] = ret.rolling(21).std()
    f["vol_63d"] = ret.rolling(63).std()
    f["max_ret_21d"] = ret.rolling(21).max()
    f["range_21d"] = ((h - l) / c).rolling(21).mean()

    f["close_loc_5d"] = ((c - l) / spread).rolling(5).mean()
    f["body_5d"] = ((c - o) / spread).rolling(5).mean()
    f["upper_shadow_5d"] = ((h - np.maximum(o, c)) / spread).rolling(5).mean()
    f["lower_shadow_5d"] = ((np.minimum(o, c) - l) / spread).rolling(5).mean()
    f["overnight_21d"] = (o / c.shift(1) - 1).rolling(21).sum()
    f["intraday_21d"] = (c / o - 1).rolling(21).sum()

    f["volume_surge"] = v.rolling(5).mean() / v.rolling(63).mean().where(lambda x: x > 0)
    f["dollar_volume"] = np.log(dollars.rolling(21).mean().where(lambda x: x > 0))
    typical = (h + l + c) / 3
    known = typical.notna() & typical.shift(1).notna()
    flow = typical * v
    rising = flow.where(typical > typical.shift(1), 0.0).where(known).rolling(14).sum()
    falling = flow.where(typical < typical.shift(1), 0.0).where(known).rolling(14).sum()
    f["money_flow_14"] = rising / (rising + falling).where(lambda x: x > 0)
    volume_21 = v.rolling(21).sum().where(lambda x: x > 0)
    f["chaikin_flow_21"] = ((((c - l) - (h - c)) / spread) * v).rolling(21).sum() / volume_21
    f["obv_21"] = (np.sign(ret) * v).rolling(21).sum() / volume_21
    f["illiquidity_21"] = np.log((ret.abs() / dollars.where(dollars > 0)).rolling(21).mean().where(lambda x: x > 0))

    traded = c.notna()
    return {name: f[name].where(traded) for name in NAMES}


def rank_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Each day's values replaced by their rank among that day's stocks, from -0.5 to +0.5.

    Ranks ignore how extreme a value is, so one wild number cannot dominate,
    and they make every feature comparable across days and with each other.
    """
    return frame.rank(axis=1, pct=True) - 0.5


def forward_return(opens: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """What a decision made after day t's close would earn: buy at the next
    open, sell at the open `horizon` trading days after that."""
    return opens.shift(-(1 + horizon)) / opens.shift(-1) - 1
