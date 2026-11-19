"""The dashboard: paper bot vs SPY, the bot league, holdings, orders, signals,
research notes, your own trades, run history and alerts.

    python main.py dashboard        (or: streamlit run dashboard.py)

Everything shown comes from the local database. The sidebar holds the controls:
pause or resume buying, fetch prices, update the league, preview the bot's
orders, and (after ticking a box) send them or start the full daily run.
Every button runs the same `python main.py ...` command you could type, so the
bot's guards and logs apply exactly as they do in a scheduled run.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from tradebot import dashboard_data as dd
from tradebot import db
from tradebot.config import ROOT, load_config, load_secrets
from tradebot.daily import Step, run_step
from tradebot.dashboard_charts import SERIES, comparison_chart, league_chart

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


def run_command(label: str, args: list[str]) -> None:
    """Run `python main.py <args>`, wait for it, and keep its output to show on the page."""
    with st.spinner(f"{label}..."):
        code, output = run_step(ROOT, Step(label, args), timeout_s=900)
    st.session_state["last_action"] = {"label": label, "code": code, "output": output}


def start_daily_run() -> None:
    """Start the full daily run in the background; it records itself under Runs and alerts."""
    subprocess.Popen([sys.executable, str(ROOT / "main.py"), "daily"], cwd=ROOT,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


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
    st.subheader("Paper bot")
    paused = db.is_paused(conn)
    if paused:
        st.warning("Buying is paused. Sells and stop-loss exits still go through.")
        if st.button("Resume buying"):
            db.set_setting(conn, "paused", "0")
            st.rerun()
    else:
        st.caption("Buying is on. The bot trades on its next run.")
        if st.button("Pause buying", help="The bot stops buying until you resume. "
                                          "Sells and stop-loss exits still go through."):
            db.set_setting(conn, "paused", "1")
            st.rerun()

    st.subheader("Run something now")
    if st.button("Fetch prices", help="Downloads the latest daily prices for every symbol. About a minute."):
        run_command("Fetch prices", ["fetch"])
    if st.button("Update the league", help="Replays every league bot on the stored prices. Under a minute."):
        run_command("Update the league", ["league"])
    if st.button("Preview the bot's orders", help="Shows what the paper bot would do. Sends nothing."):
        run_command("Preview the bot's orders", ["paper"])
    with st.expander("Send orders"):
        st.caption("These send orders to the Alpaca paper account. The bot's limits and checks still apply.")
        sure = st.checkbox("Yes, send paper orders")
        if st.button("Send the bot's orders now", disabled=not sure):
            run_command("Send the bot's orders", ["paper", "--execute"])
        busy = dd.run_in_progress(conn)
        if st.button("Start the full daily run", disabled=not sure or busy,
                     help="Fetch, research notes, paper orders and the league, as the scheduled task does. "
                          "Runs in the background for a few minutes; see Runs and alerts."):
            start_daily_run()
            st.session_state["last_action"] = {
                "label": "Start the full daily run", "code": 0,
                "output": "Started in the background. It appears under Runs and alerts when it finishes."}
        if busy:
            st.caption("A daily run is in progress.")

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
    st.caption("The scheduled run updates everything on weekdays at 2:30 pm. "
               "The buttons above do the same by hand.")

st.title("Trading assistant")
st.caption(f"{'Paper' if secrets['alpaca_paper'] else 'LIVE'} account, strategy `{bot['strategy']}`. "
           "Nothing is sent to the broker unless you use Send orders in the sidebar.")

# ---- what the last button did
action = st.session_state.get("last_action")
if action:
    with st.expander(f"{action['label']}: {'done' if action['code'] == 0 else 'FAILED'}", expanded=True):
        st.code(action["output"] or "(no output)", language=None)
        if st.button("Close"):
            del st.session_state["last_action"]
            st.rerun()

# ---- alerts nobody has read yet
unread = db.alerts_df(conn, unread_only=True)
problems = unread[unread["level"].isin(["error", "warning"])]
if len(problems):
    for row in problems.head(5).itertuples():
        show = st.error if row.level == "error" else st.warning
        if row.detail:
            show(plain(row.detail), title=plain(row.title))
        else:
            show(plain(row.title))
    if len(problems) > 5:
        st.caption(f"and {len(problems) - 5} more under Runs and alerts.")
    if st.button("Mark alerts as read"):
        db.mark_alerts_read(conn)
        st.rerun()

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
    st.altair_chart(comparison_chart(compared, mode), width="stretch",
                    alt="Line chart of the paper account's value against SPY buy and hold")
    with st.expander("Show these numbers as a table"):
        st.dataframe(compared.rename(columns={"date": "Day", "bot": SERIES[0], "benchmark": SERIES[1]}),
                     hide_index=True, width="stretch",
                     column_config={"Day": st.column_config.DateColumn(format="MMM D, YYYY"),
                                    SERIES[0]: st.column_config.NumberColumn(format="$%.2f"),
                                    SERIES[1]: st.column_config.NumberColumn(format="$%.2f")})
elif start is None:
    st.info("No orders have been sent yet. Run `python main.py paper --execute` to start the paper bot.")
else:
    st.info(f"The first order went out on {start:%b %d, %Y}. The comparison appears once there are two "
            "days of account history and SPY prices from that day on. Run `python main.py fetch`, then "
            "use Refresh in the sidebar.")

holdings_tab, league_tab, signals_tab, manual_tab, tests_tab, runs_tab = st.tabs(
    ["Holdings and orders", "Bot league", "Signals and research", "Your own trades", "Backtests",
     "Runs and alerts"])

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
            hide_index=True, width="stretch",
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
        st.dataframe(orders, hide_index=True, width="stretch", column_config={
            "ts": st.column_config.DatetimeColumn("Sent (UTC)", format="MMM D, HH:mm"),
            "symbol": "Symbol", "side": "Side",
            "notional": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "qty": st.column_config.NumberColumn("Shares", format="%.4f"),
            "price": st.column_config.NumberColumn("Fill price", format="$%.2f"),
            "status": "Status", "notes": "Notes"})

# ---------------------------------------------------------------- the bot league
with league_tab:
    from tradebot import league as league_mod
    board = league_mod.leaderboard(conn, cfg)
    if board.empty:
        st.info("The league has not been run yet. Press Update the league in the sidebar "
                "(or run `python main.py league`).")
    else:
        st.caption("Simulated bots that follow the paper bot's rules, each with its own strategy and list of "
                   "symbols. Each started with the same cash. **Live** counts only the days after a bot's "
                   "settings were fixed, so it is real evidence. **History** is a backtest: the strategies "
                   "and the symbols were chosen knowing those prices, so it flatters every bot. The `rank-` bots "
                   "hold the 25 stocks the ranking model scores highest; `hold-sp500` holds all of that list "
                   "and is the baseline they have to beat.")
        st.dataframe(
            board[["bot", "strategy", "symbols", "since", "value", "invested", "live_days", "live_return",
                   "spy_live_return", "history_cagr", "spy_history_cagr", "history_sharpe",
                   "history_max_drawdown", "trades"]],
            hide_index=True, width="stretch",
            column_config={
                "bot": "Bot", "strategy": "Strategy", "symbols": "Symbols",
                "since": st.column_config.DateColumn("Replayed from", format="MMM YYYY"),
                "value": st.column_config.NumberColumn("Value now", format="$%.0f"),
                "invested": st.column_config.ProgressColumn("Invested", format="percent", min_value=0, max_value=1),
                "live_days": st.column_config.NumberColumn("Live days", help="Trading days since the settings were fixed."),
                "live_return": st.column_config.NumberColumn("Live return", format="percent"),
                "spy_live_return": st.column_config.NumberColumn("SPY, same days", format="percent"),
                "history_cagr": st.column_config.NumberColumn("History, per year", format="percent"),
                "spy_history_cagr": st.column_config.NumberColumn("SPY, per year", format="percent"),
                "history_sharpe": st.column_config.NumberColumn("History Sharpe", format="%.2f"),
                "history_max_drawdown": st.column_config.NumberColumn("Worst drop", format="percent"),
                "trades": "Trades"})

        pick, span = st.columns([1, 2])
        chosen = pick.selectbox("Bot to look at", list(board["bot"]), key="league_bot")
        period = span.segmented_control("Period", list(dd.PERIODS), default="All history", required=True,
                                        key="league_period")
        row = board.set_index("bot").loc[chosen]
        live_from = pd.Timestamp(row["live_since"])
        curves = dd.league_curves(conn, cfg["benchmark"], period, row["live_since"], chosen)
        if curves.empty:
            st.info(f"`{chosen}` has no live days yet. Its live record starts with the first trading day "
                    f"after {live_from:%b %d, %Y}. Choose another period to see the backtest history.")
        else:
            st.caption(f"Every line starts at {plain(money(curves[chosen].iloc[0], cents=False))} on "
                       f"{curves['date'].iloc[0]:%b %d, %Y}. `{chosen}` is in blue, SPY in orange, the other "
                       "bots in grey. A bot that had not started by that day is not drawn.")
            st.altair_chart(league_chart(curves, chosen, mode, dd.BENCHMARK_LABEL, live_from), width="stretch",
                            alt=f"Line chart of the value of {chosen} against SPY and the other league bots")
            with st.expander("Show these numbers as a table"):
                st.dataframe(curves, hide_index=True, width="stretch", column_config={
                    "date": st.column_config.DateColumn("Day", format="MMM D, YYYY"),
                    **{c: st.column_config.NumberColumn(format="$%.0f") for c in curves.columns if c != "date"}})

        left, right = st.columns(2)
        left.markdown(f"**What `{chosen}` holds**")
        holdings = dd.league_holdings(conn, chosen)
        if holdings.empty:
            left.info("Nothing. It is all in cash.")
        else:
            left.dataframe(holdings[["symbol", "value", "share_of_bot", "entry_price", "last_price",
                                     "return_since_entry"]],
                           hide_index=True, width="stretch", column_config={
                "symbol": "Symbol", "value": st.column_config.NumberColumn("Value", format="$%.0f"),
                "share_of_bot": st.column_config.NumberColumn("Share of bot", format="percent"),
                "entry_price": st.column_config.NumberColumn("Entry price", format="$%.2f"),
                "last_price": st.column_config.NumberColumn("Latest close", format="$%.2f"),
                "return_since_entry": st.column_config.NumberColumn("Return since entry", format="percent")})
        right.markdown(f"**Latest trades by `{chosen}`**")
        recent = dd.league_trades(conn, chosen)
        if recent.empty:
            right.info("No trades yet.")
        else:
            right.dataframe(recent, hide_index=True, width="stretch", column_config={
                "date": st.column_config.DateColumn("Day", format="MMM D, YYYY"),
                "symbol": "Symbol", "side": "Side",
                "qty": st.column_config.NumberColumn("Shares", format="%.2f"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "notional": st.column_config.NumberColumn("Amount", format="$%.0f"), "reason": "Why"})

# ---------------------------------------------------------------- signals and research
with signals_tab:
    board = dd.signal_board(conn, bot["strategy"], cfg["watchlist"])
    st.markdown(f"**Latest view per symbol** (bot strategy: `{bot['strategy']}`)")
    st.caption("The research notes are advisory. The bot follows its strategy and does not trade on them.")
    st.dataframe(board, hide_index=True, width="stretch", column_config={
        "symbol": "Symbol",
        "bot_target": st.column_config.NumberColumn("Bot target (0 to 1)", format="%.2f"),
        "bot_view": "Bot", "research_stance": "Research note",
        "confidence": st.column_config.NumberColumn("How sure", format="percent"),
        "agreement": "Do they agree?",
        "summary": st.column_config.TextColumn("What the note says", width="large"),
        "prices_to": "Prices up to"})

    st.markdown("**How past research notes turned out**")
    from tradebot import research
    bench = cfg["benchmark"]
    scores = research.score(conn, benchmark=bench)
    if scores["notes"].sum() == 0:
        st.info("No research notes kept yet. Run `python main.py research`.")
    elif scores[[c for c in scores if c.startswith("scored_")]].to_numpy().sum() == 0:
        st.info(f"{int(scores['notes'].sum())} notes kept so far. None are old enough to score yet: "
                "a note is scored against the closes after it was written.")
    else:
        st.caption(f"Average return after each kind of note, over the next 1, 5 and 20 trading days. "
                   f"`vs_{bench}` is that return minus {bench}'s over the same days: these stocks move "
                   f"with the market, so judge the notes by `vs_{bench}`, not the plain return.")
        st.dataframe(scores, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="percent") for c in scores
            if c.startswith(("avg_return_", "vs_"))})

    with st.expander("Every research attempt, including rejected ones"):
        st.dataframe(dd.research_log(conn), hide_index=True, width="stretch")

# ---------------------------------------------------------------- your own trades
with manual_tab:
    with st.expander("Log a trade you made yourself"):
        with st.form("log_trade", clear_on_submit=True):
            first, second, third, fourth = st.columns(4)
            side = first.selectbox("Side", ["buy", "sell"])
            symbol = second.text_input("Symbol", placeholder="AAPL")
            qty = third.number_input("Shares", min_value=0.0, step=1.0, format="%.4f")
            price = fourth.number_input("Price per share", min_value=0.0, step=0.01, format="%.2f")
            fifth, sixth = st.columns([1, 3])
            fees = fifth.number_input("Fees", min_value=0.0, step=0.01, format="%.2f")
            followed = sixth.segmented_control("Did you follow the AI signal?", ["yes", "no", "not sure"],
                                               default="yes", required=True)
            notes = st.text_input("Notes (optional)")
            if st.form_submit_button("Save trade"):
                if not symbol.strip() or qty <= 0 or price <= 0:
                    st.error("Fill in the symbol, the number of shares and the price.")
                else:
                    db.log_trade(conn, "fidelity_manual", symbol.strip(), side, qty=qty, price=price,
                                 notional=qty * price, fees=fees, notes=notes or None,
                                 followed_ai={"yes": 1, "no": 0}.get(followed))
                    st.success(f"Saved: {side} {qty:g} {symbol.strip().upper()} at {plain(money(price))}.")
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
            st.dataframe(summary["by_choice"], hide_index=True, width="stretch", column_config={
                "choice": "Decision", "trades": "Trades",
                "profit": st.column_config.NumberColumn("Profit", format="$%.2f"),
                "benchmark_profit": st.column_config.NumberColumn("Same dollars in SPY", format="$%.2f")})
        st.dataframe(table, hide_index=True, width="stretch", column_config={
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
        st.dataframe(pd.read_csv(walk), hide_index=True, width="stretch")
    else:
        st.info("Not run yet. Run `python main.py walk-forward`.")
    ranking = reports / "model_test_report.md"
    st.markdown("**Ranking model test** (can price and volume history rank next week's winners?)")
    if ranking.exists():
        with st.expander("Show the full report"):
            st.markdown(plain(ranking.read_text(encoding="utf-8")))
    else:
        st.info("Not run yet. Run `python main.py model-fetch`, then `python main.py model-test`.")
    news = reports / "news_test_report.md"
    st.markdown("**News test** (do headlines predict the next session?)")
    if news.exists():
        with st.expander("Show the full report"):
            st.markdown(plain(news.read_text(encoding="utf-8")))
    else:
        st.info("Not run yet. Run `python main.py news-archive`, then `python main.py news-test`.")

# ---------------------------------------------------------------- runs and alerts
with runs_tab:
    st.markdown("**Daily runs**")
    runs = dd.run_history(conn)
    if not runs:
        st.info("No daily run has been recorded yet. The scheduled task runs on weekdays at 2:30 pm; "
                "each run then appears here with its steps.")
    for run in runs:
        when = f"{run['started']:%a %b %d, %H:%M}" if run["started"] is not None else "?"
        took = f", {run['minutes']:.0f} min" if run["minutes"] is not None else ""
        outcome = {"ok": "all steps ok", "running": "did not finish (or still running)"}.get(
            run["status"], "PROBLEM in " + ", ".join(run["failed_steps"]))
        with st.expander(f"{when}{took}: {outcome}", expanded=run["status"] == "failed" and run is runs[0]):
            if not run["steps"]:
                st.caption("No step results were stored for this run.")
            for step in run["steps"]:
                result = "ok" if step["code"] == 0 else f"FAILED (exit code {step['code']})"
                st.markdown(f"**{step['name']}**: {result}, {step['seconds']:.0f} s")
                st.code(step["output"] or "(no output)", language=None)

    st.markdown("**Alerts**")
    alerts = dd.alerts_table(conn)
    if alerts.empty:
        st.info("No alerts so far. A failed run, a stop-loss sale or a day the bot refused to trade "
                "would be listed here.")
    else:
        st.dataframe(alerts, hide_index=True, width="stretch", column_config={
            "when": st.column_config.DatetimeColumn("When (this PC's time)", format="MMM D, HH:mm"),
            "level": "Level", "title": "What happened",
            "detail": st.column_config.TextColumn("Details", width="large"),
            "delivered": "Sent to", "failed": "Not sent",
            "new": st.column_config.CheckboxColumn("Unread")})
    st.caption("Alerts also go to this PC's notifications and logs/alerts.log. For phone alerts, set "
               "NTFY_TOPIC in .env, then test with `python main.py alert-test`.")

conn.close()
