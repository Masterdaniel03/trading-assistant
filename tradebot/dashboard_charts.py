"""The dashboard's chart, kept apart from the page so it can be tested and rendered on its own."""
from __future__ import annotations

import altair as alt
import pandas as pd

alt.data_transformers.disable_max_rows()   # several years of daily values for every bot

# Chart colours: slots 1 and 2 of a palette checked for colour-blind safety and
# contrast on both backgrounds. A colour belongs to a series, never to its rank.
SERIES = ["Paper bot", "SPY buy and hold"]
COLOURS = {"light": ["#2a78d6", "#eb6834"], "dark": ["#3987e5", "#d95926"]}
OTHERS = "Other bots"
INK = {"light": {"text": "#0b0b0b", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"},
       "dark": {"text": "#f0efec", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "surface": "#1a1a19"}}



def comparison_chart(frame: pd.DataFrame, mode: str) -> alt.Chart:
    """Two lines on one dollar axis, both starting from the same amount."""
    ink = INK[mode]
    wide = frame.rename(columns={"bot": SERIES[0], "benchmark": SERIES[1]})
    long = wide.melt("date", var_name="series", value_name="value")
    colour = alt.Color("series:N", title=None, scale=alt.Scale(domain=SERIES, range=COLOURS[mode]),
                       legend=alt.Legend(orient="top", symbolType="stroke", symbolStrokeWidth=2,
                                         labelColor=ink["text"]))
    # Ticks fall on whole days (about eight across), so a short history does not
    # repeat the same date several times.
    span_days = max((frame["date"].max() - frame["date"].min()).days, 1)
    ticks = {"interval": "day", "step": max(1, -(-span_days // 8))}
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%b %d", grid=False, labelColor=ink["muted"],
                                                   domainColor=ink["axis"], tickColor=ink["axis"],
                                                   tickCount=ticks, labelOverlap=True))
    y = alt.Y("value:Q", title=None, scale=alt.Scale(zero=False, nice=True),
              axis=alt.Axis(format="$,.0f", gridColor=ink["grid"], gridDash=[], domain=False,
                            ticks=False, labelColor=ink["muted"], tickCount=5))

    lines = alt.Chart(long).mark_line(strokeWidth=2, strokeJoin="round", strokeCap="round").encode(
        x=x, y=y, color=colour)
    last = long[long["date"] == long["date"].max()]
    ends = alt.Chart(last).mark_point(size=64, filled=True, opacity=1, stroke=ink["surface"],
                                      strokeWidth=2).encode(x=x, y=y, color=colour)
    layers = [lines, ends]

    # Label the two end values only when they sit far enough apart to read.
    spread = long["value"].max() - long["value"].min()
    if spread > 0 and abs(last["value"].iloc[0] - last["value"].iloc[-1]) / spread > 0.08:
        layers.append(alt.Chart(last).mark_text(align="left", dx=8, color=ink["text"], fontSize=12).encode(
            x=x, y=y, text=alt.Text("value:Q", format="$,.0f")))

    # Hover: a vertical hairline snaps to the nearest day and reads out both series.
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False)
    hairline = alt.Chart(wide).mark_rule(color=ink["axis"], strokeWidth=1).encode(
        x="date:T", opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", title="Day", format="%a %b %d, %Y"),
                 alt.Tooltip(f"{SERIES[0]}:Q", format="$,.2f"),
                 alt.Tooltip(f"{SERIES[1]}:Q", format="$,.2f")]).add_params(nearest)
    layers.append(hairline)
    return alt.layer(*layers).properties(height=320).configure_view(strokeWidth=0)


def league_chart(frame: pd.DataFrame, selected: str, mode: str, benchmark: str = SERIES[1],
                 live_from: pd.Timestamp | None = None) -> alt.Chart:
    """Every bot on one dollar axis, all starting from the same amount.

    The chosen bot and the benchmark carry the two series colours; the other
    bots are thin grey lines for context. A dashed line marks where the live
    record begins, when that day is inside the period shown.
    """
    ink = INK[mode]
    names = [c for c in frame.columns if c != "date"]
    # Long periods are drawn from weekly points: the shape is the same and the page stays quick.
    shown = frame if len(frame) <= 400 else pd.concat(
        [frame.set_index("date").resample("W").last().dropna(how="all").reset_index(), frame.tail(1)]
    ).drop_duplicates("date", keep="last")
    long = shown.melt("date", var_name="bot", value_name="value").dropna(subset=["value"])
    long["role"] = long["bot"].map(lambda b: selected if b == selected else benchmark if b == benchmark else OTHERS)
    roles = [r for r in (selected, benchmark, OTHERS) if r in set(long["role"])]
    palette = {selected: COLOURS[mode][0], benchmark: COLOURS[mode][1], OTHERS: ink["muted"]}

    span_days = max((shown["date"].max() - shown["date"].min()).days, 1)
    axis_format, ticks = ("%b %d", {"interval": "day", "step": max(1, -(-span_days // 8))}) if span_days <= 120 \
        else ("%b %Y", 8) if span_days <= 800 else ("%Y", 8)
    x = alt.X("date:T", title=None, axis=alt.Axis(format=axis_format, grid=False, labelColor=ink["muted"],
                                                   domainColor=ink["axis"], tickColor=ink["axis"],
                                                   tickCount=ticks, labelOverlap=True))
    y = alt.Y("value:Q", title=None, scale=alt.Scale(zero=False, nice=True),
              axis=alt.Axis(format="$,.0f", gridColor=ink["grid"], gridDash=[], domain=False,
                            ticks=False, labelColor=ink["muted"], tickCount=5))
    colour = alt.Color("role:N", title=None, scale=alt.Scale(domain=roles, range=[palette[r] for r in roles]),
                       legend=alt.Legend(orient="top", symbolType="stroke", symbolStrokeWidth=2,
                                         symbolOpacity=1, labelColor=ink["text"]))

    context = alt.Chart(long[long["role"] == OTHERS]).mark_line(strokeWidth=1, opacity=0.45).encode(
        x=x, y=y, color=colour, detail="bot:N")
    main = alt.Chart(long[long["role"] != OTHERS]).mark_line(
        strokeWidth=2, strokeJoin="round", strokeCap="round").encode(x=x, y=y, color=colour)
    last = long[(long["date"] == long["date"].max()) & (long["role"] != OTHERS)]
    ends = alt.Chart(last).mark_point(size=64, filled=True, opacity=1, stroke=ink["surface"],
                                      strokeWidth=2).encode(x=x, y=y, color=colour)
    layers = [context, main, ends]

    spread = long["value"].max() - long["value"].min()
    if len(last) == 1 or (spread > 0 and abs(last["value"].iloc[0] - last["value"].iloc[-1]) / spread > 0.06):
        layers.append(alt.Chart(last).mark_text(align="left", dx=8, color=ink["text"], fontSize=12).encode(
            x=x, y=y, text=alt.Text("value:Q", format="$,.0f")))

    if live_from is not None and shown["date"].min() < live_from <= shown["date"].max():
        marker = pd.DataFrame({"date": [live_from], "label": ["live record starts"]})
        layers.append(alt.Chart(marker).mark_rule(color=ink["muted"], strokeDash=[4, 4], strokeWidth=1).encode(x="date:T"))
        # The label sits low, where growing lines rarely are, on a halo of the surface colour.
        for style in ({"stroke": ink["surface"], "strokeWidth": 4, "color": ink["surface"]}, {"color": ink["muted"]}):
            layers.append(alt.Chart(marker).mark_text(align="right", dx=-6, dy=150, fontSize=11, **style).encode(
                x="date:T", text="label:N"))

    # Hover: a hairline snaps to the nearest day and reads out every bot, the chosen one first.
    order = [selected] + ([benchmark] if benchmark in names else []) + [n for n in names if n not in (selected, benchmark)]
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False)
    hairline = alt.Chart(shown).mark_rule(color=ink["axis"], strokeWidth=1).encode(
        x="date:T", opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", title="Day", format="%a %b %d, %Y")] +
                [alt.Tooltip(field=name, type="quantitative", title=name, format="$,.0f")
                 for name in order if name in shown.columns]).add_params(nearest)
    layers.append(hairline)
    return alt.layer(*layers).properties(height=360).configure_view(strokeWidth=0)

