"""
charts.py — Plotly figures shared by the Streamlit dashboard (app.py) and the static
HTML report (scripts/build_report.py). Pure Plotly: no Streamlit imports.

The dashboard charts (make_trend_chart ... make_risk_gauge) use the dark theme of
.streamlit/config.toml; the evaluation charts below them take results tables from
reports/results/ and are used by both the dashboard's Backtest tab and the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from etl import OBLAST_EN_MAP

# ---------------------------------------------------------------------------
# Colour palette constants
# ---------------------------------------------------------------------------
COLOUR_PRIMARY = "#FF4B4B"
COLOUR_SECONDARY = "#FF8C00"
COLOUR_ACCENT = "#FFD700"
COLOUR_BG = "#0E1117"
COLOUR_CARD = "#1A1F2E"
COLOUR_SUCCESS = "#00C864"
COLOUR_CI_FILL = "rgba(255, 75, 75, 0.15)"

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_BG = "rgba(0,0,0,0)"
PLOTLY_PAPER_BG = "rgba(0,0,0,0)"

# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------

def make_trend_chart(
    daily_df: pd.DataFrame,
    selected_oblasts: list[str],
) -> go.Figure:
    """
    Build a bar + moving-average line combination chart for daily alert counts.

    Args:
        daily_df:         Daily aggregated counts DataFrame.
        selected_oblasts: List of selected oblast English names (for title).

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # Bar: daily raw counts
    fig.add_trace(go.Bar(
        x=daily_df["date"],
        y=daily_df["count"],
        name="Daily Alerts",
        marker_color=COLOUR_PRIMARY,
        marker_opacity=0.75,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Alerts: %{y}<extra></extra>",
    ))

    # Line: 7-day rolling average
    if "rolling_7d_avg" in daily_df.columns:
        fig.add_trace(go.Scatter(
            x=daily_df["date"],
            y=daily_df["rolling_7d_avg"],
            name="7-Day Avg",
            line=dict(color=COLOUR_ACCENT, width=2.5, dash="solid"),
            mode="lines",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>7d Avg: %{y:.1f}<extra></extra>",
        ))

    # Trendline via OLS (linear)
    if len(daily_df) > 3:
        x_numeric = np.arange(len(daily_df))
        y_vals = daily_df["count"].fillna(0).values
        coeffs = np.polyfit(x_numeric, y_vals, deg=1)
        trendline = np.polyval(coeffs, x_numeric)
        fig.add_trace(go.Scatter(
            x=daily_df["date"],
            y=trendline,
            name="OLS Trend",
            line=dict(color=COLOUR_SECONDARY, width=1.5, dash="dot"),
            mode="lines",
            hoverinfo="skip",
        ))

    region_label = (
        "All Oblasts" if len(selected_oblasts) == len(OBLAST_EN_MAP)
        else ", ".join(selected_oblasts[:3]) + ("…" if len(selected_oblasts) > 3 else "")
    )

    fig.update_layout(
        title=dict(text=f"Daily Alert Count — {region_label}", font=dict(size=16, color="#FAFAFA")),
        xaxis_title="Date",
        yaxis_title="Number of Alerts",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        bargap=0.15,
    )
    return fig


def make_heatmap_chart(pivot_df: pd.DataFrame) -> go.Figure:
    """
    Build a diurnal heatmap (Hour of Day × Day of Week).

    Args:
        pivot_df: Output of etl.compute_heatmap_data().

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(go.Heatmap(
        z=pivot_df.values,
        x=[f"{h:02d}:00" for h in pivot_df.columns],
        y=pivot_df.index.tolist(),
        colorscale=[
            [0.0, "#0E1117"],
            [0.3, "#3d0a0a"],
            [0.6, "#8B0000"],
            [0.85, "#CC2200"],
            [1.0, "#FF4B4B"],
        ],
        hovertemplate="<b>%{y}, %{x}</b><br>Alerts: %{z}<extra></extra>",
        colorbar=dict(
            title="Alert Count",
            titlefont=dict(color="#FAFAFA"),
            tickfont=dict(color="#FAFAFA"),
            bgcolor="rgba(26,31,46,0.8)",
        ),
    ))

    fig.update_layout(
        title=dict(text="Diurnal Alert Pattern — Hour × Day of Week", font=dict(size=16, color="#FAFAFA")),
        xaxis_title="Hour of Day (Kyiv time)",
        yaxis_title="Day of Week",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG,
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(side="bottom"),
    )
    return fig


def make_alert_type_chart(df: pd.DataFrame) -> go.Figure:
    """
    Build a horizontal bar chart of alert types by count and avg duration.

    Args:
        df: Processed alerts DataFrame.

    Returns:
        Plotly Figure.
    """
    type_stats = (
        df[df["location_type"] == "oblast"]
        .groupby("alert_type_en")
        .agg(
            count=("id", "count"),
            avg_duration=("duration_minutes", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=True)
    )

    if type_stats.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available", template=PLOTLY_TEMPLATE,
                          paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG)
        return fig

    colours = [COLOUR_PRIMARY, COLOUR_SECONDARY, COLOUR_ACCENT, "#E879F9", "#34D399", "#60A5FA"]
    bar_colours = [colours[i % len(colours)] for i in range(len(type_stats))]

    fig = go.Figure(go.Bar(
        x=type_stats["count"],
        y=type_stats["alert_type_en"],
        orientation="h",
        marker_color=bar_colours,
        marker_opacity=0.85,
        text=type_stats["count"],
        textposition="outside",
        customdata=type_stats["avg_duration"].round(1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Count: %{x}<br>"
            "Avg Duration: %{customdata} min<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text="Alert Type Distribution", font=dict(size=16, color="#FAFAFA")),
        xaxis_title="Number of Alerts",
        yaxis_title="",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG,
        margin=dict(l=0, r=40, t=50, b=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        showlegend=False,
    )
    return fig


def make_vulnerability_chart(vuln_df: pd.DataFrame) -> go.Figure:
    """
    Build a horizontal sorted bar chart of oblast vulnerability scores.

    Args:
        vuln_df: Output of etl.compute_vulnerability_score().

    Returns:
        Plotly Figure.
    """
    if vuln_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available", template=PLOTLY_TEMPLATE,
                          paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG)
        return fig

    df_sorted = vuln_df.sort_values("vulnerability_score", ascending=True)

    # Gradient colour by score
    scores_norm = df_sorted["vulnerability_score"].values
    scores_norm = (scores_norm - scores_norm.min()) / (scores_norm.max() - scores_norm.min() + 1e-9)
    colours = [
        f"rgba({int(255 * s)}, {int(75 * (1 - s))}, {int(75 * (1 - s))}, 0.85)"
        for s in scores_norm
    ]

    fig = go.Figure(go.Bar(
        x=df_sorted["vulnerability_score"],
        y=df_sorted["location_oblast_en"],
        orientation="h",
        marker_color=colours,
        text=df_sorted["vulnerability_score"].round(3),
        textposition="outside",
        customdata=np.column_stack([
            df_sorted["alert_count"],
            df_sorted["avg_duration_minutes"].round(1),
        ]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Vulnerability Score: %{x:.3f}<br>"
            "Total Alerts: %{customdata[0]}<br>"
            "Avg Duration: %{customdata[1]} min<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text="Oblast Vulnerability Ranking", font=dict(size=16, color="#FAFAFA")),
        xaxis_title="Vulnerability Score (0–1)",
        yaxis_title="",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG,
        height=max(400, len(df_sorted) * 26),
        margin=dict(l=0, r=60, t=50, b=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 1.15]),
        showlegend=False,
    )
    return fig


def make_forecast_chart(
    daily_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    oblast_label: str,
) -> go.Figure:
    """
    Build a forecast ribbon chart with historical context and 95% CI band.

    Args:
        daily_df:     Historical daily counts DataFrame.
        forecast_df:  Output of forecasting.train_and_forecast().
        oblast_label: Display name for the chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # Historical bars (last 30 days for context)
    hist = daily_df.tail(30).copy()
    hist["date"] = pd.to_datetime(hist["date"])
    fig.add_trace(go.Bar(
        x=hist["date"],
        y=hist["count"],
        name="Historical",
        marker_color="rgba(150,150,200,0.4)",
        hovertemplate="<b>%{x|%d %b}</b><br>Actual: %{y}<extra></extra>",
    ))

    if not forecast_df.empty:
        # 95% CI fill band
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"].iloc[::-1]]),
            y=pd.concat([forecast_df["ci_upper"], forecast_df["ci_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor=COLOUR_CI_FILL,
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
            hoverinfo="skip",
            showlegend=True,
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast"],
            name=f"Forecast ({forecast_df['model'].iloc[0]})",
            line=dict(color=COLOUR_PRIMARY, width=2.5),
            mode="lines+markers",
            marker=dict(size=7, symbol="diamond", color=COLOUR_PRIMARY),
            hovertemplate=(
                "<b>%{x|%d %b %Y}</b><br>"
                "Forecast: %{y:.1f}<br>"
                "<extra></extra>"
            ),
        ))

    # Vertical divider between historical and forecast
    # NOTE: fig.add_vline() is intentionally NOT used here. With a pandas
    # Timestamp/datetime x-axis, Plotly's internal annotation auto-placement
    # (axis_spanning_shape_annotation -> annotation_params_for_line -> _mean(X))
    # tries to do float(sum(x)) / len(x) on Timestamp objects, which raises a
    # TypeError on current pandas/plotly combinations. Adding the shape and
    # annotation manually avoids that buggy code path entirely.
    if not forecast_df.empty and not daily_df.empty:
        boundary = forecast_df["date"].iloc[0]

        fig.add_shape(
            type="line",
            x0=boundary, x1=boundary,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
        )
        fig.add_annotation(
            x=boundary,
            y=1,
            xref="x",
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            font=dict(color="rgba(255,255,255,0.5)"),
            xanchor="left",
            yanchor="bottom",
        )

    fig.update_layout(
        title=dict(
            text=f"7-Day Alert Count Forecast — {oblast_label}",
            font=dict(size=16, color="#FAFAFA"),
        ),
        xaxis_title="Date",
        yaxis_title="Predicted Alert Count",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=60, b=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", rangemode="tozero"),
        bargap=0.2,
    )
    return fig


def make_risk_gauge(score: float, oblast_label: str) -> go.Figure:
    """
    Build a gauge chart visualizing the predicted risk score.

    Args:
        score:        Vulnerability score in [0, 1].
        oblast_label: Display label for the title.

    Returns:
        Plotly Figure.
    """
    if score < 0.33:
        colour = COLOUR_SUCCESS
        level = "Low Risk"
    elif score < 0.66:
        colour = COLOUR_SECONDARY
        level = "Moderate Risk"
    else:
        colour = COLOUR_PRIMARY
        level = "High Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score * 100, 1),
        number=dict(suffix="%", font=dict(color=colour, size=40)),
        title=dict(text=f"{oblast_label}<br><span style='font-size:0.8em;color:gray'>{level}</span>"),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#FAFAFA", tickfont=dict(color="#FAFAFA")),
            bar=dict(color=colour, thickness=0.25),
            bgcolor="rgba(26,31,46,0.5)",
            bordercolor="rgba(255,255,255,0.1)",
            steps=[
                dict(range=[0, 33], color="rgba(0,200,100,0.1)"),
                dict(range=[33, 66], color="rgba(255,140,0,0.1)"),
                dict(range=[66, 100], color="rgba(255,75,75,0.1)"),
            ],
            threshold=dict(
                line=dict(color="white", width=2),
                thickness=0.75,
                value=score * 100,
            ),
        ),
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG,
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Evaluation charts (reports/results/*): dashboard Backtest tab + HTML report
# ---------------------------------------------------------------------------
# `theme` is "dark" (dashboard, matches .streamlit/config.toml) or "light" (the
# report's default; the report switches templates in the browser for dark mode).
# No text colours are hard-coded here, so the template decides them.

MODEL_LABELS = {
    "lgbm": "LightGBM",
    "logit": "Logistic regression",
    "lgbm_text": "LightGBM + text",
    "logit_text": "Logistic + text",
    "unit_rate_28d": "Raion rate, 28 d",
    "hour_of_day_rate_28d": "Same hour rate, 28 d",
    "recent_rate_24h": "Last 24 h rate",
    "seasonal_naive": "Same hour/day last week",
    "poisson_glm": "Poisson GLM",
    "holt_winters": "Holt-Winters (dashboard)",
    "mean_28d": "Mean of 28 d",
    "mean_7d": "Mean of 7 d",
    "last_day": "Last day",
    "lgbm_quantile": "LightGBM quantiles",
    "lgbm_conformal": "LightGBM quantiles + conformal",
}
MODEL_COLOURS = {
    "lgbm": COLOUR_PRIMARY,
    "logit": COLOUR_SECONDARY,
    "lgbm_text": "#F9A8A8",
    "logit_text": "#FCC98A",
    "poisson_glm": COLOUR_SECONDARY,
    "holt_winters": "#A78BFA",
    "lgbm_quantile": "#F9A8A8",
    "lgbm_conformal": COLOUR_PRIMARY,
}
BASELINE_COLOUR = "#60A5FA"
NEUTRAL_COLOUR = "#94A3B8"
COUNTRY = "Україна"


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def model_colour(name: str) -> str:
    return MODEL_COLOURS.get(name, BASELINE_COLOUR if name in MODEL_LABELS else NEUTRAL_COLOUR)


def oblast_label(name: str) -> str:
    return "Ukraine (all raions)" if name == COUNTRY else OBLAST_EN_MAP.get(name, name)


def _style(fig: go.Figure, theme: str, title: str, height: int = 420, **layout) -> go.Figure:
    grid = "rgba(255,255,255,0.08)" if theme == "dark" else "rgba(0,0,0,0.08)"
    fig.update_layout({
        "title": dict(text=title, font=dict(size=15)),
        "template": "plotly_dark" if theme == "dark" else "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "height": height,
        "margin": dict(l=10, r=10, t=60, b=10),
        "legend": dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        "font": dict(family="Inter, system-ui, sans-serif", size=12),
    } | layout)
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid)
    return fig


def make_skill_chart(bootstrap: pd.DataFrame, vs: str = "unit_rate_28d", theme: str = "dark") -> go.Figure:
    """Target A: Brier skill vs the reference per model and horizon, with 95% day-block bootstrap CIs."""
    b = bootstrap[(bootstrap["target"] == "A") & (bootstrap["vs"] == vs) & (bootstrap["block_days"] == 1)]
    # "Same hour last week" (BSS ~ -1) is left out: it would flatten the axis; it is in the tables
    order = ["lgbm", "logit", "lgbm_text", "logit_text", "hour_of_day_rate_28d", "recent_rate_24h"]
    fig = go.Figure()
    for m in [m for m in order if m in set(b["model"])]:
        g = b[b["model"] == m].sort_values("h")
        fig.add_trace(go.Scatter(
            x=[f"h = {int(h)} h" for h in g["h"]], y=g["skill"], name=model_label(m), mode="markers",
            marker=dict(size=10, color=model_colour(m)),
            error_y=dict(type="data", symmetric=False, array=g["ci_high"] - g["skill"],
                         arrayminus=g["skill"] - g["ci_low"], thickness=1.5, width=5),
            hovertemplate="%{x}: BSS %{y:.3f}<extra>" + model_label(m) + "</extra>",
        ))
    fig.add_hline(y=0, line=dict(color=NEUTRAL_COLOUR, dash="dash", width=1))
    return _style(fig, theme, f"Brier skill vs {model_label(vs)} (95% CI, bootstrap over days)",
                  scattermode="group", yaxis_title="Brier skill score (higher is better)")


def make_reliability_chart(reliability: pd.DataFrame, h: int, theme: str = "dark",
                           models: tuple[str, ...] = ("lgbm", "logit", "unit_rate_28d")) -> go.Figure:
    """Reliability diagram: mean forecast vs observed frequency per probability bin."""
    r = reliability[reliability["h"] == h]
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                               line=dict(color=NEUTRAL_COLOUR, dash="dash", width=1), hoverinfo="skip"))
    for m in models:
        g = r[r["model"] == m].sort_values("mean_forecast")
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=g["mean_forecast"], y=g["observed"], name=model_label(m), mode="lines+markers",
            line=dict(color=model_colour(m)),
            marker=dict(size=np.clip(np.sqrt(g["n"]) / 8, 5, 18), color=model_colour(m)),
            customdata=g["n"], hovertemplate="forecast %{x:.2f} → observed %{y:.2f} (n=%{customdata})<extra></extra>",
        ))
    return _style(fig, theme, f"Calibration, alert start within {h} h", height=440,
                  legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
                  margin=dict(l=10, r=10, t=50, b=10),
                  xaxis=dict(title="Mean forecast probability", range=[0, 1]),
                  yaxis=dict(title="Observed frequency", range=[0, 1]))


def make_hourly_fact_chart(series: pd.DataFrame, h: int, theme: str = "dark",
                           ref: str = "unit_rate_28d", initial: str = COUNTRY) -> go.Figure:
    """
    Target A on the test period: share of the oblast's raions where an alert started
    within h hours vs the mean forecast probability; a dropdown picks the oblast.
    """
    s = series[series["h"] == h].sort_values("origin")
    oblasts = [COUNTRY] + sorted((o for o in s["oblast"].unique() if o != COUNTRY), key=oblast_label)
    fig = go.Figure()
    traces = (("y", "Observed share of raions", NEUTRAL_COLOUR, 1.0, "tozeroy"),
              (ref, model_label(ref), BASELINE_COLOUR, 1.6, None),
              ("lgbm", model_label("lgbm"), COLOUR_PRIMARY, 1.8, None))
    for o in oblasts:
        g = s[s["oblast"] == o]
        for col, name, colour, width, fill in traces:
            fig.add_trace(go.Scatter(
                x=g["origin"], y=g[col], name=name, mode="lines", visible=o == initial, fill=fill,
                line=dict(color=colour, width=width, shape="hv" if col == "y" else "linear"),
                opacity=0.45 if col == "y" else 1.0, legendgroup=col,
                hovertemplate="%{y:.2f}<extra>" + name + "</extra>",
            ))
    n = len(traces)
    buttons = [dict(label=oblast_label(o), method="update",
                    args=[{"visible": [j // n == i for j in range(n * len(oblasts))]}])
               for i, o in enumerate(oblasts)]
    return _style(fig, theme, f"Forecast vs actual, test period: alert start within {h} h", height=460,
                  hovermode="x unified", yaxis=dict(title="Probability / share", range=[0, 1]),
                  updatemenus=[dict(buttons=buttons, direction="down", x=1, xanchor="right", y=1.18,
                                    yanchor="top", showactive=True)])


def make_daily_fact_chart(series: pd.DataFrame, target: str = "count", k: int = 1, theme: str = "dark",
                          initial: str | None = None) -> go.Figure:
    """Target B on the test period per raion: actual vs LightGBM with its conformal 80/95% band."""
    s = series[(series["target"] == target) & (series["k"] == k)].sort_values("day")
    units = sorted(s["unit"].unique(), key=lambda u: (oblast_label(s.loc[s["unit"] == u, "oblast"].iloc[0]), u))
    initial = initial if initial in units else s.groupby("unit")["y"].mean().idxmax()
    fig = go.Figure()
    per_unit = 5
    for u in units:
        g = s[s["unit"] == u]
        vis = u == initial
        x = pd.concat([g["day"], g["day"][::-1]])
        for lo, hi, alpha, name in (("lo95", "hi95", 0.12, "95% interval"), ("lo80", "hi80", 0.22, "80% interval")):
            fig.add_trace(go.Scatter(x=x, y=pd.concat([g[hi], g[lo][::-1]]), fill="toself", mode="lines",
                                     line=dict(width=0), fillcolor=f"rgba(255,75,75,{alpha})", name=name,
                                     hoverinfo="skip", visible=vis, legendgroup=name))
        fig.add_trace(go.Bar(x=g["day"], y=g["y"], name="Actual", marker_color=NEUTRAL_COLOUR, opacity=0.6,
                             visible=vis, legendgroup="y", hovertemplate="%{y:.1f}<extra>Actual</extra>"))
        fig.add_trace(go.Scatter(x=g["day"], y=g["mean_28d"], name=model_label("mean_28d"), mode="lines",
                                 line=dict(color=BASELINE_COLOUR, dash="dot"), visible=vis, legendgroup="m28",
                                 hovertemplate="%{y:.2f}<extra>" + model_label("mean_28d") + "</extra>"))
        fig.add_trace(go.Scatter(x=g["day"], y=g["lgbm"], name=model_label("lgbm"), mode="lines+markers",
                                 line=dict(color=COLOUR_PRIMARY, width=2), visible=vis, legendgroup="lgbm",
                                 hovertemplate="%{y:.2f}<extra>LightGBM</extra>"))
    buttons = []
    for i, u in enumerate(units):
        ob = oblast_label(s.loc[s["unit"] == u, "oblast"].iloc[0])
        buttons.append(dict(label=f"{ob} · {u}", method="update",
                            args=[{"visible": [j // per_unit == i for j in range(per_unit * len(units))]}]))
    what = "alerts per day" if target == "count" else "hours under alert per day"
    return _style(fig, theme, f"Raion {what}, {k} day(s) ahead: test period", height=460, hovermode="x unified",
                  yaxis=dict(title=what, rangemode="tozero"), bargap=0.25,
                  updatemenus=[dict(buttons=buttons, direction="down", x=1, xanchor="right", y=1.18,
                                    yanchor="top", showactive=True)])


def make_oblast_skill_chart(errors: pd.DataFrame, h: int, model: str = "lgbm", theme: str = "dark",
                            floor: float = -0.5) -> go.Figure:
    """
    Brier skill of `model` vs the raion rate per oblast (bar colour: base rate). Where alerts
    are almost absent the reference Brier is ~0 and BSS explodes; bars are cut at `floor`
    and labelled with the true value.
    """
    e = errors[(errors["h"] == h) & (errors["model"] == model) & (errors["by"] == "oblast")].copy()
    e["label"] = e["group"].map(oblast_label)
    e = e.sort_values("bss")
    clipped = e["bss"] < floor
    fig = go.Figure(go.Bar(
        x=e["bss"].clip(lower=floor), y=e["label"], orientation="h",
        text=[f"{v:.1f} ◀" if c else "" for v, c in zip(e["bss"], clipped)], textposition="inside",
        marker=dict(color=e["base_rate"], colorscale="Reds", cmin=0, cmax=max(0.05, e["base_rate"].max()),
                    colorbar=dict(title="Base<br>rate", thickness=12)),
        customdata=np.column_stack([e["base_rate"], e["n"], e["bss"]]),
        hovertemplate="<b>%{y}</b><br>BSS %{customdata[2]:.3f}<br>base rate %{customdata[0]:.3f}<br>"
                      "n %{customdata[1]}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=NEUTRAL_COLOUR, width=1))
    return _style(fig, theme, f"{model_label(model)}: Brier skill vs raion rate by oblast (h = {h} h)",
                  height=max(420, 22 * len(e) + 80), xaxis_title="Brier skill score", showlegend=False)


def make_hour_skill_chart(errors: pd.DataFrame, model: str = "lgbm", theme: str = "dark") -> go.Figure:
    """Brier skill vs the raion rate by Kyiv hour of the forecast origin, one line per horizon."""
    e = errors[(errors["model"] == model) & (errors["by"] == "kyiv_hour")]
    fig = go.Figure()
    for i, (h, g) in enumerate(e.groupby("h")):
        g = g.assign(group=g["group"].astype(int)).sort_values("group")
        fig.add_trace(go.Scatter(x=g["group"], y=g["bss"], name=f"h = {h} h", mode="lines+markers",
                                 line=dict(color=[COLOUR_PRIMARY, COLOUR_SECONDARY, BASELINE_COLOUR][i % 3])))
    fig.add_hline(y=0, line=dict(color=NEUTRAL_COLOUR, dash="dash", width=1))
    return _style(fig, theme, f"{model_label(model)}: Brier skill by hour of day (Kyiv)", height=380,
                  xaxis=dict(title="Origin hour (Kyiv)", dtick=3), yaxis_title="Brier skill score")


def make_mase_chart(daily_scores: pd.DataFrame, target: str = "count", metric: str = "rel_mae",
                    cells: str = "common", theme: str = "dark") -> go.Figure:
    """
    Target B: a point-forecast metric by horizon (days) per model. `rel_mae` is MAE relative
    to the row's `ref` forecast: MASE on the "common" cells, vs the 28-day mean on "all_days".
    """
    d = daily_scores[(daily_scores["target"] == target) & (daily_scores["cells"] == cells)]
    fig = go.Figure()
    for m in ["lgbm", "poisson_glm", "holt_winters", "mean_28d", "mean_7d", "last_day", "seasonal_naive"]:
        g = d[d["model"] == m].sort_values("k")
        if g.empty:
            continue
        fig.add_trace(go.Scatter(x=g["k"], y=g[metric], name=model_label(m), mode="lines+markers",
                                 line=dict(color=model_colour(m), dash="solid" if m in MODEL_COLOURS else "dot")))
    what = "alert count" if target == "count" else "hours under alert"
    rel = "MASE (vs same day last week)" if cells == "common" else "MAE relative to the 28-day mean"
    label = {"rel_mae": rel, "poisson_dev": "Poisson deviance", "mae": "MAE"}[metric]
    return _style(fig, theme, f"Daily {what} per raion: {label}", height=400,
                  xaxis=dict(title="Horizon, days", dtick=1), yaxis_title=label + " (lower is better)")


def make_coverage_chart(intervals: pd.DataFrame, target: str = "count", theme: str = "dark") -> go.Figure:
    """Empirical coverage of the 80/95% intervals by horizon, against the nominal level."""
    iv = intervals[intervals["target"] == target]
    fig = go.Figure()
    for level, dash in ((80, "dot"), (95, "solid")):
        fig.add_hline(y=level / 100, line=dict(color=NEUTRAL_COLOUR, dash="dash", width=1),
                      annotation_text=f"nominal {level}%", annotation_position="right")
        for m in ("lgbm_quantile", "lgbm_conformal", "holt_winters"):
            g = iv[(iv["model"] == m) & (iv["level"] == level)].sort_values("k")
            if g.empty:
                continue
            fig.add_trace(go.Scatter(x=g["k"], y=g["coverage"], name=f"{model_label(m)}, {level}%",
                                     mode="lines+markers", line=dict(color=model_colour(m), dash=dash),
                                     customdata=g["width"], hovertemplate="coverage %{y:.3f}, mean width "
                                     "%{customdata:.2f}<extra></extra>"))
    what = "alert count" if target == "count" else "hours under alert"
    return _style(fig, theme, f"Interval coverage, daily {what}", height=400,
                  xaxis=dict(title="Horizon, days", dtick=1), yaxis=dict(title="Empirical coverage"))


def make_history_chart(overview: pd.DataFrame, theme: str = "dark") -> go.Figure:
    """Daily mean hours under alert per raion (Kyiv days) with a 28-day average; gaps stay empty."""
    o = overview.sort_values("day")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=o["day"], y=o["hours_per_raion"], name="Hours under alert per raion, daily",
                         marker_color=NEUTRAL_COLOUR, opacity=0.5,
                         hovertemplate="%{x|%d %b %Y}: %{y:.1f} h<extra></extra>"))
    fig.add_trace(go.Scatter(x=o["day"], y=o["hours_per_raion"].rolling(28, min_periods=14).mean(),
                             name="28-day average", mode="lines", line=dict(color=COLOUR_PRIMARY, width=2),
                             hovertemplate="%{x|%d %b %Y}: %{y:.1f} h<extra>28-day avg</extra>"))
    return _style(fig, theme, "How much of the day a raion spends under alert, 2022–2026", height=400,
                  yaxis_title="Hours per raion per day", bargap=0, hovermode="x unified")


def make_hour_dow_heatmap(pivot: pd.DataFrame, theme: str = "dark", title: str = "") -> go.Figure:
    """Alert starts by Kyiv hour of day × day of week (pivot: rows = weekdays, columns = hours)."""
    fig = go.Figure(go.Heatmap(z=pivot.to_numpy(), x=[f"{h:02d}" for h in pivot.columns], y=list(pivot.index),
                               colorscale="Reds", colorbar=dict(title="Starts", thickness=12),
                               hovertemplate="%{y} %{x}:00 · %{z}<extra></extra>"))
    return _style(fig, theme, title or "Alert starts by hour (Kyiv) and weekday", height=340,
                  xaxis_title="Hour of day (Kyiv)", yaxis=dict(autorange="reversed"))


def make_oblast_share_chart(shares: pd.DataFrame, theme: str = "dark", title: str = "") -> go.Figure:
    """Share of time under alert per oblast (mean over its raions)."""
    s = shares.assign(label=shares["oblast"].map(oblast_label)).sort_values("share")
    fig = go.Figure(go.Bar(x=s["share"], y=s["label"], orientation="h", marker_color=COLOUR_PRIMARY,
                           hovertemplate="<b>%{y}</b>: %{x:.1%}<extra></extra>"))
    return _style(fig, theme, title or "Share of time under alert by oblast", height=max(420, 22 * len(s) + 80),
                  xaxis=dict(title="Share of hours under alert", tickformat=".0%"), showlegend=False)


def make_split_chart(windows: list[dict], theme: str = "dark") -> go.Figure:
    """Evaluation layout on a time axis: rows of (label, start, end, kind)."""
    colours = {"train": NEUTRAL_COLOUR, "fold": BASELINE_COLOUR, "validation": COLOUR_SECONDARY,
               "gap": "rgba(148,163,184,0.25)", "test": COLOUR_PRIMARY}
    fig = go.Figure()
    for w in windows:
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        fig.add_trace(go.Bar(
            x=[(end - start).total_seconds() * 1000], base=[start], y=[w["row"]], orientation="h",
            marker_color=colours[w["kind"]], name=w["label"], showlegend=False,
            text=w["label"], textposition="inside", insidetextanchor="middle",
            hovertemplate=f"<b>{w['label']}</b><br>{start:%d %b %Y} – {end:%d %b %Y}<extra></extra>",
        ))
    return _style(fig, theme, "Chronological splits (no random shuffling)", height=260, barmode="overlay",
                  xaxis=dict(type="date"), yaxis=dict(autorange="reversed"), bargap=0.35)
