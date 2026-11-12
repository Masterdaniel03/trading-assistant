"""The dashboard's chart, kept apart from the page so it can be tested and rendered on its own."""
from __future__ import annotations

import altair as alt
import pandas as pd

# Chart colours: slots 1 and 2 of a palette checked for colour-blind safety and
# contrast on both backgrounds. A colour belongs to a series, never to its rank.
SERIES = ["Paper bot", "SPY buy and hold"]
COLOURS = {"light": ["#2a78d6", "#eb6834"], "dark": ["#3987e5", "#d95926"]}
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
