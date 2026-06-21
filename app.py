"""
app.py — Ukraine Air Raid Alarms Time-Series Dashboard
=======================================================
Streamlit application providing live + historical analysis and
7-day forecasting for Ukrainian air-raid alert data.

Run locally:
    streamlit run app.py

Secrets required (in .streamlit/secrets.toml or Streamlit Cloud):
    ALERTS_API_TOKEN = "your_token_here"

Token request form: https://alerts.in.ua/api-request
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🇺🇦 Air Raid Alarms Analytics",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://devs.alerts.in.ua",
        "Report a bug": None,
        "About": "Air Raid Alarms Time-Series Dashboard · Data: alerts.in.ua",
    },
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local module imports (after page config)
# ---------------------------------------------------------------------------
from etl import (
    OBLAST_EN_MAP,
    ALERT_TYPE_EN_MAP,
    compute_daily_counts,
    compute_heatmap_data,
    compute_vulnerability_score,
    fetch_and_process_data,
)
from forecasting import (
    prepare_series_for_region,
    train_and_forecast,
    compute_national_forecast,
    compute_risk_forecast_score,
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-mode styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient header */
    .dashboard-header {
        background: linear-gradient(135deg, #1a0a0a 0%, #2d0f0f 40%, #1a1a2e 100%);
        border: 1px solid rgba(255, 75, 75, 0.3);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #FF4B4B, #FF8C00, #FFD700);
    }
    .dashboard-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #FAFAFA;
        margin: 0 0 4px 0;
        letter-spacing: -0.5px;
    }
    .dashboard-header p {
        color: rgba(250,250,250,0.6);
        font-size: 0.9rem;
        margin: 0;
    }

    /* KPI metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1A1F2E 0%, #141820 100%);
        border: 1px solid rgba(255, 75, 75, 0.2);
        border-radius: 10px;
        padding: 16px 20px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(255, 75, 75, 0.5);
        transform: translateY(-2px);
    }

    /* Status badges */
    .status-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 200, 100, 0.15);
        border: 1px solid rgba(0, 200, 100, 0.4);
        color: #00C864;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-offline {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 75, 75, 0.15);
        border: 1px solid rgba(255, 75, 75, 0.4);
        color: #FF4B4B;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1A1F2E;
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(255,75,75,0.2);
        color: rgba(250,250,250,0.7);
        font-weight: 500;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2d0f0f, #1a1a2e);
        border-color: rgba(255,75,75,0.6);
        color: #FF4B4B !important;
    }

    /* Alert banner */
    .active-alert-banner {
        background: linear-gradient(90deg, rgba(255,75,75,0.15), rgba(255,140,0,0.1));
        border-left: 4px solid #FF4B4B;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        animation: pulse-border 2s ease-in-out infinite;
    }
    @keyframes pulse-border {
        0%, 100% { border-left-color: #FF4B4B; }
        50%       { border-left-color: #FF8C00; }
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #141820 100%);
        border-right: 1px solid rgba(255,75,75,0.15);
    }

    /* Streamlit metric delta colour overrides */
    [data-testid="metric-container"] {
        background: #1A1F2E;
        border: 1px solid rgba(255,75,75,0.15);
        border-radius: 10px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
# Helper: resolve API token (secrets > env var)
# ---------------------------------------------------------------------------

def get_api_token() -> Optional[str]:
    """
    Retrieve the API token from Streamlit secrets or environment variable.

    Priority: st.secrets["ALERTS_API_TOKEN"] → os.getenv("ALERTS_API_TOKEN")

    Returns:
        Token string or None if not configured.
    """
    try:
        token = st.secrets.get("ALERTS_API_TOKEN")
        if token:
            return str(token).strip()
    except Exception:
        pass
    return os.getenv("ALERTS_API_TOKEN", "").strip() or None


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_data(api_token: str, fetch_history: bool) -> tuple[pd.DataFrame, dict]:
    """
    Cache-wrapped ETL entry point.
    TTL = 300s (5 minutes) to respect API rate limits.

    Args:
        api_token:     Bearer token for alerts.in.ua.
        fetch_history: Whether to pull 30-day per-oblast history.

    Returns:
        Tuple of (processed DataFrame, status dict).
    """
    return fetch_and_process_data(api_token, fetch_history=fetch_history)


@st.cache_resource(show_spinner=False)
def get_forecast(
    series_hash: int,
    series_values: tuple,
    series_dates: tuple,
    horizon: int,
) -> pd.DataFrame:
    """
    Cache-wrapped forecasting (cached by series hash + horizon).

    Args:
        series_hash:   Hash of the series for cache keying.
        series_values: Tuple of series float values (hashable).
        series_dates:  Tuple of series date strings (hashable).
        horizon:       Forecast horizon in days.

    Returns:
        Forecast DataFrame.
    """
    # Reconstruct the pd.Series from hashable tuple
    idx = pd.to_datetime(list(series_dates), utc=True)
    series = pd.Series(list(series_values), index=idx, name="count", dtype=float)
    return train_and_forecast(series, horizon)


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
        xaxis_title="Hour of Day (UTC)",
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
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(
    api_ok: bool,
    last_updated: Optional[datetime],
    active_count: int,
    history_failures: Optional[dict] = None,
) -> dict:
    """
    Render the sidebar controls and return the user's filter selections.

    Args:
        api_ok:           Whether the API connection was successful.
        last_updated:      Timestamp of the last successful data fetch.
        active_count:      Number of currently active alerts from the API.
        history_failures:  Dict of {oblast_en: reason} for oblasts whose
                            history fetch failed or returned nothing, as
                            reported by etl.fetch_and_process_data().

    Returns:
        Dict with keys: selected_oblasts (list[str]), days_range (int),
        fetch_history (bool), forecast_horizon (int).
    """
    history_failures = history_failures or {}

    with st.sidebar:
        st.markdown("## 🚨 Dashboard Controls")

        # API Status badge
        st.markdown("### API Status")
        if api_ok:
            st.markdown(
                '<span class="status-online">● API Connected</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-offline">● API Offline</span>',
                unsafe_allow_html=True,
            )

        if active_count > 0:
            st.markdown(
                f'<div class="active-alert-banner">⚠️ <b>{active_count}</b> active alerts right now</div>',
                unsafe_allow_html=True,
            )

        if last_updated:
            st.caption(f"Last updated: {last_updated.strftime('%H:%M:%S UTC')}")

        # Surface per-oblast history fetch issues instead of letting them
        # fail silently into logs (which Streamlit Cloud users never see).
        if history_failures:
            with st.expander(f"⚠️ {len(history_failures)} region(s) had history-fetch issues", expanded=False):
                for oblast_en, reason in sorted(history_failures.items()):
                    st.caption(f"**{oblast_en}** — {reason}")

        st.divider()

        # Oblast multiselect
        st.markdown("### 🗺️ Region Filter")
        all_oblasts = sorted(OBLAST_EN_MAP.values())
        selected_oblasts = st.multiselect(
            "Select Oblasts",
            options=all_oblasts,
            default=all_oblasts,
            help="Filter data to specific Ukrainian oblasts. Default = all.",
            key="oblast_filter",
        )
        if not selected_oblasts:
            st.warning("Select at least one oblast.")
            selected_oblasts = all_oblasts

        st.divider()

        # Date range slider
        st.markdown("### 📅 Time Window")
        days_options = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30}
        days_label = st.select_slider(
            "Analysis Window",
            options=list(days_options.keys()),
            value="Last 30 days",
            help="Filter historical charts to this time window.",
            key="date_range",
        )
        days_range = days_options[days_label]

        st.divider()

        # History fetch toggle
        st.markdown("### ⚙️ Data Options")
        fetch_history = st.toggle(
            "Fetch 30-day history",
            value=True,
            help=(
                "Loads per-oblast historical data via the history endpoint. "
                "Rate-limited to 2 req/min — initial load takes ~12 min for all oblasts. "
                "Disable for instant load (active alerts only)."
            ),
            key="fetch_history",
        )

        # Forecast horizon
        forecast_horizon = st.slider(
            "Forecast Horizon (days)",
            min_value=3,
            max_value=14,
            value=7,
            step=1,
            help="Number of days to forecast ahead.",
            key="forecast_horizon",
        )

        st.divider()

        # Manual refresh
        if st.button("🔄 Refresh Data", use_container_width=True, key="refresh_btn"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.caption(
            "Data source: [alerts.in.ua](https://alerts.in.ua) · "
            "API docs: [devs.alerts.in.ua](https://devs.alerts.in.ua)"
        )

    return {
        "selected_oblasts": selected_oblasts,
        "days_range": days_range,
        "fetch_history": fetch_history,
        "forecast_horizon": forecast_horizon,
    }


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

def render_kpi_row(
    df: pd.DataFrame,
    daily_df: pd.DataFrame,
    days_range: int,
    active_count: int,
) -> None:
    """
    Render the top KPI metrics row.

    Computes current-window KPIs and delta vs previous equivalent window.

    Args:
        df:           Full processed alerts DataFrame.
        daily_df:     Daily aggregated counts.
        days_range:   Number of days in the current analysis window.
        active_count: Live active alert count from API.
    """
    now = datetime.now(timezone.utc)
    cutoff_current = pd.Timestamp(now - timedelta(days=days_range))
    cutoff_previous = pd.Timestamp(now - timedelta(days=days_range * 2))

    oblast_df = df[df["location_type"] == "oblast"].copy()
    current = oblast_df[oblast_df["started_at"] >= cutoff_current]
    previous = oblast_df[
        (oblast_df["started_at"] >= cutoff_previous)
        & (oblast_df["started_at"] < cutoff_current)
    ]

    def safe_delta(curr_val: float, prev_val: float) -> Optional[float]:
        if prev_val == 0:
            return None
        return round(curr_val - prev_val, 1)

    total_alerts = len(current)
    prev_total = len(previous)
    avg_duration = current["duration_minutes"].mean() if not current.empty else 0.0
    prev_avg_dur = previous["duration_minutes"].mean() if not previous.empty else 0.0
    max_duration = current["duration_minutes"].max() if not current.empty else 0.0
    unique_oblasts = current["location_oblast_en"].nunique()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="🚨 Total Alerts",
            value=f"{total_alerts:,}",
            delta=safe_delta(total_alerts, prev_total),
            delta_color="inverse",
            help=f"Total alerts in the last {days_range} days",
        )
    with col2:
        st.metric(
            label="⏱️ Avg Duration",
            value=f"{avg_duration:.0f} min" if avg_duration else "N/A",
            delta=safe_delta(avg_duration, prev_avg_dur),
            delta_color="inverse",
            help="Average alert duration in minutes",
        )
    with col3:
        max_dur_hours = max_duration / 60 if max_duration else 0
        st.metric(
            label="🔥 Max Duration",
            value=f"{max_dur_hours:.1f} hr" if max_duration else "N/A",
            help="Longest single alert duration in the window",
        )
    with col4:
        st.metric(
            label="🗺️ Affected Oblasts",
            value=f"{unique_oblasts}",
            help=f"Unique oblasts with alerts in the last {days_range} days",
        )
    with col5:
        st.metric(
            label="⚡ Active Now",
            value=f"{active_count}",
            delta=None,
            help="Currently active alerts (live from API)",
        )


# ---------------------------------------------------------------------------
# Tab 1: Historical Analytics
# ---------------------------------------------------------------------------

def render_history_tab(
    df: pd.DataFrame,
    selected_oblasts: list[str],
    days_range: int,
) -> None:
    """
    Render the historical analytics tab with trend, heatmap, and type breakdown.

    Args:
        df:               Full processed alerts DataFrame.
        selected_oblasts: User-selected oblast filter.
        days_range:       Analysis window in days.
    """
    # Apply time filter
    now = datetime.now(timezone.utc)
    cutoff = pd.Timestamp(now - timedelta(days=days_range))
    df_filtered = df[df["started_at"] >= cutoff].copy()

    # Apply oblast filter
    if len(selected_oblasts) < len(OBLAST_EN_MAP):
        df_filtered = df_filtered[df_filtered["location_oblast_en"].isin(selected_oblasts)]

    if df_filtered.empty:
        st.warning("No alert data available for the selected filters and time window.")
        return

    # Daily counts for the filtered window
    daily = compute_daily_counts(
        df_filtered,
        oblast_filter=selected_oblasts if len(selected_oblasts) < len(OBLAST_EN_MAP) else None,
    )

    # ── Row 1: Trend chart ────────────────────────────────────────────────
    st.plotly_chart(
        make_trend_chart(daily, selected_oblasts),
        use_container_width=True,
        key="trend_chart",
    )

    # ── Row 2: Heatmap + Alert type breakdown ────────────────────────────
    col_heat, col_types = st.columns([3, 2], gap="medium")

    with col_heat:
        pivot = compute_heatmap_data(
            df_filtered,
            oblast_filter=selected_oblasts if len(selected_oblasts) < len(OBLAST_EN_MAP) else None,
        )
        st.plotly_chart(
            make_heatmap_chart(pivot),
            use_container_width=True,
            key="heatmap_chart",
        )

    with col_types:
        st.plotly_chart(
            make_alert_type_chart(df_filtered),
            use_container_width=True,
            key="type_chart",
        )

    # ── Row 3: Oblast vulnerability ranking ──────────────────────────────
    st.markdown("### Oblast Vulnerability Ranking")
    vuln = compute_vulnerability_score(
        df_filtered,
        oblast_filter=selected_oblasts if len(selected_oblasts) < len(OBLAST_EN_MAP) else None,
    )
    st.plotly_chart(
        make_vulnerability_chart(vuln),
        use_container_width=True,
        key="vuln_chart",
    )

    # ── Raw data expander ────────────────────────────────────────────────
    with st.expander("📋 Raw Data Table", expanded=False):
        display_cols = [
            "started_at", "location_oblast_en", "alert_type_en",
            "duration_minutes", "is_active", "location_raion",
        ]
        available_cols = [c for c in display_cols if c in df_filtered.columns]
        show_df = df_filtered[available_cols].copy()
        show_df["started_at"] = show_df["started_at"].dt.strftime("%Y-%m-%d %H:%M UTC")
        if "duration_minutes" in show_df.columns:
            show_df["duration_minutes"] = show_df["duration_minutes"].round(1)
        st.dataframe(
            show_df.rename(columns={
                "started_at": "Started At",
                "location_oblast_en": "Oblast",
                "alert_type_en": "Alert Type",
                "duration_minutes": "Duration (min)",
                "is_active": "Active",
                "location_raion": "Raion",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Tab 2: Forecast
# ---------------------------------------------------------------------------

def render_forecast_tab(
    df: pd.DataFrame,
    selected_oblasts: list[str],
    forecast_horizon: int,
) -> None:
    """
    Render the predictive forecasting tab with regional and national forecasts.

    Args:
        df:               Full processed alerts DataFrame.
        selected_oblasts: User-selected oblast filter.
        forecast_horizon: Number of days to forecast.
    """
    # Oblast selector for forecast focus
    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        forecast_oblast = st.selectbox(
            "Select Oblast for Detailed Forecast",
            options=["🇺🇦 All Ukraine (National)"] + sorted(selected_oblasts),
            index=0,
            key="forecast_oblast_select",
            help="Choose a specific region or view the national aggregate forecast.",
        )

    is_national = forecast_oblast.startswith("🇺🇦")
    oblast_label = "Ukraine (National)" if is_national else forecast_oblast

    # Compute daily series
    if is_national:
        daily_df = compute_daily_counts(df)
    else:
        daily_df = compute_daily_counts(df, oblast_filter=[forecast_oblast])

    series = prepare_series_for_region(daily_df, region_name=oblast_label)

    if series.empty or len(series) < 3:
        if series.empty:
            reason = (
                "No date range could be established at all for this region — "
                "this usually means the overall dataset itself is empty."
            )
        else:
            reason = (
                f"Only {len(series)} day(s) of history are available for "
                f"**{oblast_label}** so far (need at least 3)."
            )
        st.warning(
            f"Insufficient historical data for **{oblast_label}** to generate a forecast. "
            f"{reason}\n\n"
            "Try enabling **'Fetch 30-day history'** in the sidebar, waiting for more data to "
            "accumulate, or selecting a different region. If you've already enabled history "
            "fetch, check the **⚠️ region(s) had history-fetch issues** expander in the "
            "sidebar — that region's fetch may have failed silently (e.g. rate limit or "
            "transient API error)."
        )
        return

    # Get or compute forecast (cached)
    series_hash = hash((tuple(series.values.tolist()), forecast_horizon))
    try:
        with st.spinner(f"Training forecast model for {oblast_label}…"):
            forecast_df = get_forecast(
                series_hash=series_hash,
                series_values=tuple(series.values.tolist()),
                series_dates=tuple(series.index.astype(str).tolist()),
                horizon=forecast_horizon,
            )
    except Exception as exc:
        st.error(f"Forecast computation failed: {exc}")
        return

    risk_score = compute_risk_forecast_score(forecast_df)

    # ── Forecast summary row ──────────────────────────────────────────────
    with col_info:
        model_used = forecast_df["model"].iloc[0] if not forecast_df.empty else "N/A"
        fc_mean = forecast_df["forecast"].mean() if not forecast_df.empty else 0.0
        fc_max = forecast_df["forecast"].max() if not forecast_df.empty else 0.0
        st.markdown(
            f"""
            <div style='
                background: linear-gradient(135deg, #1A1F2E, #141820);
                border: 1px solid rgba(255,75,75,0.25);
                border-radius: 10px;
                padding: 16px 20px;
                margin-top: 4px;
            '>
                <b style='color:#FAFAFA'>Model:</b>
                <span style='color:#FF8C00'> {model_used}</span> &nbsp;|&nbsp;
                <b style='color:#FAFAFA'>Horizon:</b>
                <span style='color:#FFD700'> {forecast_horizon} days</span> &nbsp;|&nbsp;
                <b style='color:#FAFAFA'>Predicted Mean:</b>
                <span style='color:#FF4B4B'> {fc_mean:.1f} alerts/day</span> &nbsp;|&nbsp;
                <b style='color:#FAFAFA'>Predicted Peak:</b>
                <span style='color:#FF4B4B'> {fc_max:.1f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Gauge + Forecast ribbon chart ────────────────────────────────────
    col_gauge, col_fc = st.columns([1, 3], gap="medium")
    with col_gauge:
        st.plotly_chart(
            make_risk_gauge(risk_score, oblast_label),
            use_container_width=True,
            key="risk_gauge",
        )
    with col_fc:
        st.plotly_chart(
            make_forecast_chart(daily_df, forecast_df, oblast_label),
            use_container_width=True,
            key="forecast_ribbon",
        )

    # ── Forecast data table ───────────────────────────────────────────────
    if not forecast_df.empty:
        with st.expander("📊 Forecast Values Table", expanded=False):
            display_fc = forecast_df.copy()
            display_fc["date"] = display_fc["date"].dt.strftime("%Y-%m-%d")
            display_fc["forecast"] = display_fc["forecast"].round(1)
            display_fc["ci_lower"] = display_fc["ci_lower"].round(1)
            display_fc["ci_upper"] = display_fc["ci_upper"].round(1)
            st.dataframe(
                display_fc.rename(columns={
                    "date": "Date",
                    "forecast": "Predicted Alerts",
                    "ci_lower": "CI Lower (95%)",
                    "ci_upper": "CI Upper (95%)",
                    "model": "Model",
                }),
                use_container_width=True,
                hide_index=True,
            )

    # ── Multi-oblast risk summary ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗺️ Multi-Oblast Risk Overview (Next 7 Days)")
    st.caption(
        "Predicted risk scores for all available oblasts based on their individual forecast models. "
        "Score = mean forecast / (mean forecast + 1), normalised to [0, 1]."
    )

    risk_data: list[dict] = []
    for ob in selected_oblasts:
        ob_daily = compute_daily_counts(df, oblast_filter=[ob])
        ob_series = prepare_series_for_region(ob_daily, region_name=ob)
        if len(ob_series) < 3:
            continue
        try:
            ob_hash = hash((tuple(ob_series.values.tolist()), 7))
            ob_fc = get_forecast(
                series_hash=ob_hash,
                series_values=tuple(ob_series.values.tolist()),
                series_dates=tuple(ob_series.index.astype(str).tolist()),
                horizon=7,
            )
            ob_score = compute_risk_forecast_score(ob_fc)
            risk_data.append({
                "Oblast": ob,
                "Risk Score": round(ob_score, 3),
                "Avg Forecast (alerts/day)": round(ob_fc["forecast"].mean(), 1),
                "Peak Forecast": round(ob_fc["forecast"].max(), 1),
                "Model": ob_fc["model"].iloc[0],
            })
        except Exception:
            continue

    if risk_data:
        risk_summary = pd.DataFrame(risk_data).sort_values("Risk Score", ascending=False)

        # Horizontal bar chart of risk scores
        fig_risk = px.bar(
            risk_summary,
            x="Risk Score",
            y="Oblast",
            orientation="h",
            color="Risk Score",
            color_continuous_scale=[[0, COLOUR_SUCCESS], [0.5, COLOUR_SECONDARY], [1, COLOUR_PRIMARY]],
            title="Predicted 7-Day Risk Score by Oblast",
            hover_data=["Avg Forecast (alerts/day)", "Peak Forecast", "Model"],
            template=PLOTLY_TEMPLATE,
        )
        fig_risk.update_layout(
            paper_bgcolor=PLOTLY_PAPER_BG,
            plot_bgcolor=PLOTLY_BG,
            height=max(400, len(risk_summary) * 26),
            margin=dict(l=0, r=0, t=50, b=0),
            xaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.05)"),
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            showlegend=False,
        )
        st.plotly_chart(fig_risk, use_container_width=True, key="risk_multi_chart")

        st.dataframe(risk_summary, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main app entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main application orchestrator — renders the full dashboard.
    """
    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>🇺🇦 Air Raid Alarms Analytics</h1>
            <p>Time-Series Analysis &amp; Forecasting Dashboard · Powered by alerts.in.ua API</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Token resolution ─────────────────────────────────────────────────
    api_token = get_api_token()

    if not api_token:
        st.error(
            "**API Token Missing.**\n\n"
            "This dashboard requires a valid `ALERTS_API_TOKEN` to function.\n\n"
            "**Local development:** Add your token to `.streamlit/secrets.toml`:\n"
            "```toml\nALERTS_API_TOKEN = \"your_token_here\"\n```\n\n"
            "**Streamlit Cloud:** Set the token in your app's Secrets settings.\n\n"
            "Request a free token at [alerts.in.ua/api-request](https://alerts.in.ua/api-request)."
        )
        st.stop()

    # ── Data loading ─────────────────────────────────────────────────────
    # Read sidebar state for fetch_history before rendering sidebar
    # (sidebar renders after token check)
    sidebar_fetch_history = st.session_state.get("fetch_history", True)

    loading_placeholder = st.empty()
    with loading_placeholder.container():
        if sidebar_fetch_history:
            st.info(
                "⏳ **Loading data…** Fetching historical alerts for all oblasts "
                "(rate-limited: ~2 requests/min). This may take a few minutes on first load. "
                "Subsequent loads use a 5-minute cache."
            )

    try:
        df, api_status = load_data(api_token, fetch_history=sidebar_fetch_history)
    except RuntimeError as exc:
        loading_placeholder.empty()
        st.error(f"**API Error:** {exc}")
        st.stop()

    loading_placeholder.empty()

    # ── Sidebar ───────────────────────────────────────────────────────────
    filters = render_sidebar(
        api_ok=api_status["api_ok"],
        last_updated=api_status.get("last_updated"),
        active_count=api_status.get("active_count", 0),
        history_failures=api_status.get("history_failures", {}),
    )

    if df.empty:
        st.warning(
            "No alert data returned. This may mean:\n"
            "- No active alerts currently, and history fetch was disabled\n"
            "- The API returned an empty response\n\n"
            "Try enabling **'Fetch 30-day history'** in the sidebar."
        )
        st.stop()

    # ── KPI row ───────────────────────────────────────────────────────────
    daily_all = compute_daily_counts(df)
    render_kpi_row(df, daily_all, filters["days_range"], api_status["active_count"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_hist, tab_forecast = st.tabs([
        "📊 Historical Analytics",
        "🔮 Forecast & Risk Assessment",
    ])

    with tab_hist:
        render_history_tab(df, filters["selected_oblasts"], filters["days_range"])

    with tab_forecast:
        render_forecast_tab(df, filters["selected_oblasts"], filters["forecast_horizon"])


if __name__ == "__main__":
    main()
