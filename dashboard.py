"""The dashboard: paper bot vs SPY, holdings, orders, signals, research notes, your own trades.

    streamlit run dashboard.py

It reads the local database only and never places an order. The numbers that
come from the broker are refreshed by every `python main.py paper` run, or by
the "Refresh from Alpaca" button here (which also sends no orders).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from tradebot import dashboard_data as dd
from tradebot import db
from tradebot.config import ROOT, load_config, load_secrets
from tradebot.dashboard_charts import SERIES, comparison_chart

def money(value, cents: bool = True) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${value:,.2f}" if cents else f"${value:,.0f}"


def percent(value, signed: bool = True) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def plain(text: str) -> str:
    """Streamlit reads $...$ as maths. Escape dollar signs in text we did not write."""
    return text.replace("$", "\\$")


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # noqa: BLE001 - older Streamlit, or no browser attached
        return "light"


# ---------------------------------------------------------------- page

st.set_page_config(page_title="Trading assistant", layout="wide")
cfg = load_config()
secrets = load_secrets()
account = "alpaca_paper" if secrets["alpaca_paper"] else "alpaca_live"
conn = db.connect(os.environ.get("TRADEBOT_DB") or cfg["db_path"])
db.init_db(conn)
risk, bot = cfg["risk"], cfg["paper_bot"]
mode = theme_mode()

with st.sidebar:
    st.subheader("Data")
    if st.button("Refresh from Alpaca", help="Updates the balance, holdings and order results. Sends no orders."):
        try:
            from tradebot.account_sync import refresh_from_broker
            from tradebot.broker import AlpacaBroker
            done = refresh_from_broker(conn, AlpacaBroker(secrets["alpaca_key"], secrets["alpaca_secret"],
                                                         paper=secrets["alpaca_paper"]))
            st.success(f"Updated: {done['positions']} holdings, {done['orders_checked']} orders checked.")
        except Exception as error:  # noqa: BLE001 - shown to the reader, the page carries on
            st.error(f"Could not refresh: {type(error).__name__}: {error}")
    state = dd.health(conn, account, bot["symbols"])
    updated = state["last_account_update"]
    st.caption("Account last updated: " + (updated.strftime("%b %d, %H:%M New York") if updated is not None else "never"))
    st.caption(f"Prices stored up to: {state['oldest_price_date'] or 'none'}")
    if state["symbols_without_prices"]:
        st.warning("No prices for: " + ", ".join(state["symbols_without_prices"]))
    st.caption("Prices and signals update when you run `python main.py fetch` and `python main.py paper`.")

st.title("Trading assistant")
st.caption(f"{'Paper' if secrets['alpaca_paper'] else 'LIVE'} account, strategy `{bot['strategy']}`. "
           "This page only reads data. It never places an order.")

snapshot = dd.latest_snapshot(conn, account)
equity = dd.equity_series(conn, account)
start = dd.first_trade_date(conn, account)
compared = dd.compare_with_benchmark(equity, db.load_close(conn, cfg["benchmark"]), start)
bot_stats = dd.performance(compared["bot"]) if len(compared) else dd.performance(pd.Series(dtype=float))
spy_stats = dd.performance(compared["benchmark"]) if len(compared) else dd.performance(pd.Series(dtype=float))

# ---- headline numbers
tiles = st.columns(5)
day_change = float(equity.iloc[-1] - equity.iloc[-2]) if len(equity) >= 2 else None
tiles[0].metric("Account value", money(snapshot["equity"]) if snapshot else "n/a",
                delta=f"{day_change:+,.2f} since the previous day" if day_change else None)
gap = (bot_stats["total_return"] - spy_stats["total_return"]) if bot_stats["total_return"] is not None else None
tiles[1].metric("Return since the first trade", percent(bot_stats["total_return"]),
                delta=f"{gap * 100:+.2f} points vs SPY" if gap is not None else None)
tiles[2].metric("SPY over the same days", percent(spy_stats["total_return"]))
tiles[3].metric("Worst drop from a peak", percent(bot_stats["max_drawdown"]),
                help="The largest fall from a high point to a later low.")
tiles[4].metric("Sharpe ratio", f"{bot_stats['sharpe']:.2f}" if bot_stats["sharpe"] is not None else "n/a",
                help=f"Return per unit of risk. Shown after {dd.MIN_DAYS_FOR_SHARPE} trading days; "
                     f"{bot_stats['days']} so far.")

# ---- bot vs SPY
st.subheader("Account value vs SPY buy and hold")
if len(compared):
    st.caption(f"Both start from {plain(money(compared['bot'].iloc[0], cents=False))} on "
               f"{compared['date'].iloc[0]:%b %d, %Y}, the day of the first order. "
               "SPY is bought at that day's close.")
    st.altair_chart(comparison_chart(compared, mode), use_container_width=True)
    with st.expander("Show these numbers as a table"):
        st.dataframe(compared.rename(columns={"date": "Day", "bot": SERIES[0], "benchmark": SERIES[1]}),
                     hide_index=True, use_container_width=True,
                     column_config={"Day": st.column_config.DateColumn(format="MMM D, YYYY"),
                                    SERIES[0]: st.column_config.NumberColumn(format="$%.2f"),
                                    SERIES[1]: st.column_config.NumberColumn(format="$%.2f")})
elif start is None:
    st.info("No orders have been sent yet. Run `python main.py paper --execute` to start the paper bot.")
else:
    st.info(f"The first order went out on {start:%b %d, %Y}. The comparison appears once there are two "
            "days of account history and SPY prices from that day on. Run `python main.py fetch`, then "
            "use Refresh in the sidebar.")

holdings_tab, signals_tab, manual_tab, tests_tab = st.tabs(
    ["Holdings and orders", "Signals and research", "Your own trades", "Backtests"])

# ---------------------------------------------------------------- holdings and orders
with holdings_tab:
    held = dd.positions_table(conn, account, snapshot["equity"] if snapshot else None, risk)
    orders = dd.orders_table(conn, account)
    waiting = dd.waiting_orders(orders)
    stop = risk["max_risk_per_position_pct"] / risk["max_position_pct"]

    st.markdown("**Holdings**")
    if held.empty:
        st.info("Nothing is held right now." + (f" {len(waiting)} order(s) are waiting to fill." if len(waiting) else ""))
    else:
        st.caption(f"Each holding is capped at {risk['max_position_pct']:.0%} of the account and is sold if it "
                   f"falls {stop:.1%} below its entry price.")
        st.dataframe(
            held[["symbol", "market_value", "share_of_account", "avg_entry_price", "unrealized_plpc",
                  "room_before_stop", "status"]],
            hide_index=True, use_container_width=True,
            column_config={
                "symbol": "Symbol",
                "market_value": st.column_config.NumberColumn("Value", format="$%.2f"),
                "share_of_account": st.column_config.ProgressColumn(
                    f"Share of account (cap {risk['max_position_pct']:.0%})", format="percent",
                    min_value=0, max_value=risk["max_position_pct"]),
                "avg_entry_price": st.column_config.NumberColumn("Entry price", format="$%.2f"),
                "unrealized_plpc": st.column_config.NumberColumn("Return since entry", format="percent"),
                "room_before_stop": st.column_config.NumberColumn("Room before the stop", format="percent"),
                "status": "Stop-loss status"})

    st.markdown("**Orders sent by the bot**")
    if orders.empty:
        st.info("No orders yet.")
    else:
        if len(waiting):
            st.caption(f"{len(waiting)} waiting to fill. Fill prices appear after the next refresh.")
        st.dataframe(orders, hide_index=True, use_container_width=True, column_config={
            "ts": st.column_config.DatetimeColumn("Sent (UTC)", format="MMM D, HH:mm"),
            "symbol": "Symbol", "side": "Side",
            "notional": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "qty": st.column_config.NumberColumn("Shares", format="%.4f"),
            "price": st.column_config.NumberColumn("Fill price", format="$%.2f"),
            "status": "Status", "notes": "Notes"})

# ---------------------------------------------------------------- signals and research
with signals_tab:
    board = dd.signal_board(conn, bot["strategy"], cfg["watchlist"])
    st.markdown(f"**Latest view per symbol** (bot strategy: `{bot['strategy']}`)")
    st.caption("The research notes are advisory. The bot follows its strategy and does not trade on them.")
    st.dataframe(board, hide_index=True, use_container_width=True, column_config={
        "symbol": "Symbol",
        "bot_target": st.column_config.NumberColumn("Bot target (0 to 1)", format="%.2f"),
        "bot_view": "Bot", "research_stance": "Research note",
        "confidence": st.column_config.NumberColumn("How sure", format="percent"),
        "agreement": "Do they agree?",
        "summary": st.column_config.TextColumn("What the note says", width="large"),
        "prices_to": "Prices up to"})

    st.markdown("**How past research notes turned out**")
    from tradebot import research
    scores = research.score(conn)
    if scores["notes"].sum() == 0:
        st.info("No research notes kept yet. Run `python main.py research`.")
    elif scores[[c for c in scores if c.startswith("scored_")]].to_numpy().sum() == 0:
        st.info(f"{int(scores['notes'].sum())} notes kept so far. None are old enough to score yet: "
                "a note is scored against the closes after it was written.")
    else:
        st.caption("Average return after each kind of note, over the next 1, 5 and 20 trading days.")
        st.dataframe(scores, hide_index=True, use_container_width=True, column_config={
            c: st.column_config.NumberColumn(format="percent") for c in scores if c.startswith("avg_return_")})

    with st.expander("Every research attempt, including rejected ones"):
        st.dataframe(dd.research_log(conn), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- your own trades
with manual_tab:
    table, summary = dd.manual_review(conn, benchmark=cfg["benchmark"])
    if table.empty:
        st.info("No manual trades logged yet. Record one with, for example: "
                "`python main.py log-trade buy AAPL 2 350.00 --followed-ai yes`")
    else:
        a, b, c = st.columns(3)
        a.metric("Profit on your trades", money(summary["profit"]),
                 help="Valued at the latest stored close, after fees.")
        b.metric("Same dollars in SPY", money(summary["benchmark_profit"]),
                 help="What each trade's money would have made in SPY, bought the same day.")
        c.metric("Trades that followed the AI",
                 percent(summary["follow_rate"], signed=False) if summary["follow_rate"] is not None else "n/a")
        if summary["unpriced"]:
            st.warning("No stored prices for: " + ", ".join(summary["unpriced"]) +
                       ". Run `python main.py fetch " + " ".join(summary["unpriced"]) + "` to value them.")
        if len(summary["by_choice"]):
            st.dataframe(summary["by_choice"], hide_index=True, use_container_width=True, column_config={
                "choice": "Decision", "trades": "Trades",
                "profit": st.column_config.NumberColumn("Profit", format="$%.2f"),
                "benchmark_profit": st.column_config.NumberColumn("Same dollars in SPY", format="$%.2f")})
        st.dataframe(table, hide_index=True, use_container_width=True, column_config={
            "date": "Day", "symbol": "Symbol", "side": "Side", "qty": "Shares",
            "price": st.column_config.NumberColumn("Your price", format="$%.2f"),
            "last_close": st.column_config.NumberColumn("Latest close", format="$%.2f"),
            "profit": st.column_config.NumberColumn("Profit", format="$%.2f"),
            "benchmark_profit": st.column_config.NumberColumn("Same dollars in SPY", format="$%.2f"),
            "followed_ai": st.column_config.CheckboxColumn("Followed the AI"), "notes": "Notes"})

# ---------------------------------------------------------------- backtests
with tests_tab:
    reports = Path(os.environ.get("TRADEBOT_REPORTS") or ROOT / "reports")
    walk = reports / "walk_forward_summary.csv"
    st.markdown("**Walk-forward test of the strategies** (settings picked on past years, scored on the next)")
    if walk.exists():
        st.dataframe(pd.read_csv(walk), hide_index=True, use_container_width=True)
    else:
        st.info("Not run yet. Run `python main.py walk-forward`.")
    news = reports / "news_test_report.md"
    st.markdown("**News test** (do headlines predict the next session?)")
    if news.exists():
        with st.expander("Show the full report"):
            st.markdown(plain(news.read_text(encoding="utf-8")))
    else:
        st.info("Not run yet. Run `python main.py news-archive`, then `python main.py news-test`.")

conn.close()
