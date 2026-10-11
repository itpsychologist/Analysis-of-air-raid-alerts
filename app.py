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
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import plotly.express as px
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
    compute_daily_counts,
    mean_finished_duration,
    compute_heatmap_data,
    compute_vulnerability_score,
    fetch_and_process_data,
)
from forecasting import (
    prepare_series_for_region,
    train_and_forecast,
    compute_risk_forecast_score,
)
from src.viz.charts import (
    COLOUR_PRIMARY,
    COLOUR_SECONDARY,
    COLOUR_SUCCESS,
    PLOTLY_BG,
    PLOTLY_PAPER_BG,
    PLOTLY_TEMPLATE,
    make_alert_type_chart,
    make_forecast_chart,
    make_heatmap_chart,
    make_risk_gauge,
    make_trend_chart,
    make_vulnerability_chart,
)
from src.viz import charts as eval_charts
from src.viz.results import load_results

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
    # Active alerts are censored at fetch time, so averages use finished alerts only
    avg_duration = mean_finished_duration(current) if not current.empty else 0.0
    prev_avg_dur = mean_finished_duration(previous) if not previous.empty else 0.0
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
# Backtest tab: the committed final-test results (no API, no model fitting)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_results():
    """reports/results/ of the one-time final-test run (None before it exists)."""
    return load_results()


def render_backtest_tab() -> None:
    """Final-test metrics, calibration and forecast vs actual (scripts/run_final_test.py)."""
    res = get_results()
    if res is None:
        st.info("No final-test results yet: run `scripts/run_final_test.py` and `scripts/build_report.py`.")
        return
    m = res.marker
    t0, t1 = (pd.Timestamp(x) for x in m["test_window_utc"])
    st.markdown(
        f"Evaluation on the **frozen final test, {t0:%d %b} – {t1:%d %b %Y}** (scored once, "
        f"{pd.Timestamp(m['scored_at']):%d %b %Y}); data cutoff {pd.Timestamp(m['data_cutoff']):%d %b %Y %H:%M} UTC. "
        "Models were selected on earlier data only. Forecasts are analytical and **do not replace official "
        "alert signals**. The static report `reports/index.html` explains the method."
    )

    st.markdown("#### Will an alert start in the raion within *h* hours?")
    st.plotly_chart(eval_charts.make_skill_chart(res.bootstrap), use_container_width=True)
    h = st.radio("Horizon", (1, 3, 6), index=0, horizontal=True, format_func=lambda x: f"{x} h")
    cols = ["model", "n", "brier", "bss", "roc_auc", "roc_auc_within", "pr_auc", "log_loss"]
    table = res.scores[res.scores["h"] == h][cols].copy()
    table["model"] = table["model"].map(eval_charts.model_label)
    st.dataframe(
        table.rename(columns={"bss": "BSS vs raion rate", "roc_auc_within": "AUC within raion"}).style.format(
            {c: "{:.3f}" for c in cols[2:]} | {"brier": "{:.4f}", "n": "{:,}"}),
        hide_index=True, use_container_width=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(eval_charts.make_reliability_chart(res.reliability, h), use_container_width=True)
    with c2:
        st.plotly_chart(eval_charts.make_oblast_skill_chart(res.errors, h), use_container_width=True)
    st.plotly_chart(eval_charts.make_hourly_fact_chart(res.hourly_series, h), use_container_width=True)

    st.markdown("#### Daily alerts per raion, 1–7 days ahead")
    target = st.radio("Target", ("count", "duration"), horizontal=True,
                      format_func={"count": "Alert count", "duration": "Hours under alert"}.get)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(eval_charts.make_mase_chart(res.daily, target), use_container_width=True)
    with c2:
        st.plotly_chart(eval_charts.make_coverage_chart(res.intervals, target), use_container_width=True)
    k = st.slider("Days ahead", 1, 7, 1)
    st.plotly_chart(eval_charts.make_daily_fact_chart(res.daily_series, target, k), use_container_width=True)


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
        render_backtest_tab()
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
        render_backtest_tab()
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
    tab_hist, tab_forecast, tab_backtest = st.tabs([
        "📊 Historical Analytics",
        "🔮 Forecast & Risk Assessment",
        "🧪 Backtest",
    ])

    with tab_hist:
        render_history_tab(df, filters["selected_oblasts"], filters["days_range"])

    with tab_forecast:
        render_forecast_tab(df, filters["selected_oblasts"], filters["forecast_horizon"])

    with tab_backtest:
        render_backtest_tab()


if __name__ == "__main__":
    main()
