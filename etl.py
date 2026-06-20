"""
etl.py — Data Acquisition & Preprocessing Module
==================================================
Resilient ETL pipeline for alerts.in.ua API.

Rate limits:
  - /v1/alerts/active.json          → 8-12 req/min  (cached 5 min)
  - /v1/regions/{uid}/alerts/...    → 2 req/min      (cached 5 min, fetched once)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Oblast UID mapping (oblast-level only, as per the API documentation)
# Source: https://devs.alerts.in.ua — location_uid for oblast-type locations
# ---------------------------------------------------------------------------
OBLAST_UID_MAP: dict[str, str] = {
    "Вінницька область": "1",
    "Волинська область": "2",
    "Дніпропетровська область": "3",
    "Донецька область": "4",
    "Житомирська область": "5",
    "Закарпатська область": "6",
    "Запорізька область": "7",
    "Івано-Франківська область": "8",
    "Київська область": "9",
    "Кіровоградська область": "10",
    "Луганська область": "11",
    "Львівська область": "12",
    "Миколаївська область": "13",
    "Одеська область": "14",
    "Полтавська область": "15",
    "Рівненська область": "16",
    "Сумська область": "17",
    "Тернопільська область": "18",
    "Харківська область": "19",
    "Херсонська область": "20",
    "Хмельницька область": "21",
    "Черкаська область": "22",
    "Чернівецька область": "23",
    "Чернігівська область": "24",
    "м. Київ": "31",
}

# Human-readable English names for UI display
OBLAST_EN_MAP: dict[str, str] = {
    "Вінницька область": "Vinnytsia",
    "Волинська область": "Volyn",
    "Дніпропетровська область": "Dnipropetrovsk",
    "Донецька область": "Donetsk",
    "Житомирська область": "Zhytomyr",
    "Закарпатська область": "Zakarpattia",
    "Запорізька область": "Zaporizhzhia",
    "Івано-Франківська область": "Ivano-Frankivsk",
    "Київська область": "Kyiv Oblast",
    "Кіровоградська область": "Kirovohrad",
    "Луганська область": "Luhansk",
    "Львівська область": "Lviv",
    "Миколаївська область": "Mykolaiv",
    "Одеська область": "Odesa",
    "Полтавська область": "Poltava",
    "Рівненська область": "Rivne",
    "Сумська область": "Sumy",
    "Тернопільська область": "Ternopil",
    "Харківська область": "Kharkiv",
    "Херсонська область": "Kherson",
    "Хмельницька область": "Khmelnytskyi",
    "Черкаська область": "Cherkasy",
    "Чернівецька область": "Chernivtsi",
    "Чернігівська область": "Chernihiv",
    "м. Київ": "Kyiv City",
}

ALERT_TYPE_EN_MAP: dict[str, str] = {
    "air_raid": "Air Raid",
    "artillery_shelling": "Artillery Shelling",
    "urban_fights": "Urban Combat",
    "chemical": "Chemical Threat",
    "nuclear": "Nuclear Threat",
    "unknown": "Unknown",
}

BASE_URL = "https://api.alerts.in.ua/v1"
REQUEST_TIMEOUT = 15  # seconds


def _build_headers(api_token: str) -> dict[str, str]:
    """Construct request headers with Bearer token authentication."""
    return {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json",
        "User-Agent": "UkraineAlertsAnalytics/1.0",
    }


def _handle_response_errors(response: requests.Response, context: str) -> None:
    """
    Raise a descriptive RuntimeError for known HTTP error codes.

    Args:
        response: The HTTP response object.
        context:  Descriptive string for logging context.

    Raises:
        RuntimeError: On 401, 403, 429, 5xx status codes.
    """
    status = response.status_code
    if status == 200:
        return
    if status == 304:
        return  # Not Modified — data unchanged, caller handles this
    error_messages = {
        401: "API token is missing, invalid, revoked, or expired (HTTP 401).",
        403: "Your IP or token is blocked, or the API is unavailable in your region (HTTP 403).",
        429: "Rate limit exceeded — too many requests per minute (HTTP 429). Wait and retry.",
        500: "API server internal error (HTTP 500). Try again later.",
        503: "API service temporarily unavailable (HTTP 503). Try again later.",
    }
    msg = error_messages.get(status, f"Unexpected HTTP error {status} from {context}.")
    raise RuntimeError(msg)


def fetch_active_alerts(api_token: str) -> list[dict]:
    """
    Fetch currently active alerts from the API.

    Args:
        api_token: Valid Bearer token for alerts.in.ua API.

    Returns:
        List of alert dictionaries. Empty list on failure.

    Raises:
        RuntimeError: On authentication or rate-limit errors.
    """
    url = f"{BASE_URL}/alerts/active.json"
    try:
        response = requests.get(
            url,
            headers=_build_headers(api_token),
            timeout=REQUEST_TIMEOUT,
        )
        _handle_response_errors(response, "active alerts endpoint")
        data = response.json()
        return data.get("alerts", [])
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Network connection error: {exc}") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"Request timed out after {REQUEST_TIMEOUT}s: {exc}") from exc
    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from API: {exc}") from exc


def fetch_oblast_history(api_token: str, oblast_uid: str, period: str = "month_ago") -> list[dict]:
    """
    Fetch alert history for a single oblast.

    Note: This endpoint has a strict 2 req/min rate limit.
    Callers must implement appropriate inter-request delays.

    Args:
        api_token:  Valid Bearer token.
        oblast_uid: Location UID (e.g., "4" for Donetsk oblast).
        period:     History period — currently only "month_ago" is supported.

    Returns:
        List of alert dictionaries for the requested oblast + period.
    """
    url = f"{BASE_URL}/regions/{oblast_uid}/alerts/{period}.json"
    try:
        response = requests.get(
            url,
            headers=_build_headers(api_token),
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 304:
            return []
        _handle_response_errors(response, f"history endpoint (uid={oblast_uid})")
        data = response.json()
        # API returns either {"alerts": [...]} or direct list
        if isinstance(data, list):
            return data
        return data.get("alerts", [])
    except RuntimeError:
        raise
    except Exception as exc:
        logger.warning("Failed to fetch history for UID %s: %s", oblast_uid, exc)
        return []


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    """
    Parse an ISO 8601 timestamp string to a timezone-aware UTC datetime.

    Args:
        ts: Timestamp string like "2022-04-04T16:45:39.000Z" or None.

    Returns:
        Timezone-aware datetime in UTC, or None if input is None/invalid.
    """
    if ts is None:
        return None
    try:
        # Handle both 'Z' suffix and '+00:00' offset
        ts_clean = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_clean)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        logger.warning("Could not parse timestamp: %r", ts)
        return None


def _compute_duration(
    started: Optional[datetime],
    finished: Optional[datetime],
    now: datetime,
) -> Optional[float]:
    """
    Compute alert duration in minutes.

    For active (ongoing) alerts where `finished` is None,
    uses `now` as the effective end time.

    Args:
        started:  Alert start datetime (UTC-aware).
        finished: Alert end datetime (UTC-aware), or None if still active.
        now:      Current UTC datetime for active-alert duration estimation.

    Returns:
        Duration in minutes as float, or None if start is unknown.
    """
    if started is None:
        return None
    end = finished if finished is not None else now
    delta = end - started
    minutes = delta.total_seconds() / 60.0
    # Guard against negative durations (data quality issue)
    return max(0.0, minutes)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical and categorical time-series features to the DataFrame.

    Features added:
        - hour_of_day     : 0–23
        - day_of_week     : 0 (Mon) – 6 (Sun)
        - day_name        : Monday, Tuesday, ...
        - is_weekend      : bool
        - month           : 1–12
        - month_name      : January, February, ...
        - week_of_year    : ISO week number
        - date            : date-only (for daily aggregation)
        - hour_sin/cos    : cyclical encoding of hour
        - dow_sin/cos     : cyclical encoding of day-of-week

    Args:
        df: DataFrame with a 'started_at' column of timezone-aware datetimes.

    Returns:
        DataFrame with new feature columns appended.
    """
    dt = df["started_at"].dt
    df = df.copy()
    df["hour_of_day"] = dt.hour
    df["day_of_week"] = dt.dayofweek
    df["day_name"] = dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["month"] = dt.month
    df["month_name"] = dt.month_name()
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["date"] = dt.date
    # Cyclical encodings — preserve periodicity for ML models
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def _normalize_alert_record(record: dict, now: datetime) -> dict:
    """
    Normalize a raw API alert record into a clean, typed dictionary.

    Args:
        record: Raw alert dict from the API response.
        now:    Current UTC datetime for active-alarm duration calculation.

    Returns:
        Normalized dict with typed and computed fields.
    """
    started = _parse_timestamp(record.get("started_at"))
    finished = _parse_timestamp(record.get("finished_at"))
    is_active = finished is None
    duration = _compute_duration(started, finished, now)

    raw_oblast = record.get("location_oblast") or record.get("location_title", "Unknown")
    alert_type_raw = record.get("alert_type", "unknown")

    return {
        "id": record.get("id"),
        "location_uid": record.get("location_uid"),
        "location_oblast_uid": record.get("location_oblast_uid"),
        "location_title": record.get("location_title", "Unknown"),
        "location_type": record.get("location_type", "unknown"),
        "location_oblast": raw_oblast,
        "location_oblast_en": OBLAST_EN_MAP.get(raw_oblast, raw_oblast),
        "location_raion": record.get("location_raion"),
        "alert_type": alert_type_raw,
        "alert_type_en": ALERT_TYPE_EN_MAP.get(alert_type_raw, alert_type_raw.replace("_", " ").title()),
        "started_at": started,
        "finished_at": finished,
        "is_active": is_active,
        "duration_minutes": duration,
        "calculated": record.get("calculated", False),
        "notes": record.get("notes"),
    }


def fetch_and_process_data(
    api_token: str,
    fetch_history: bool = True,
    history_period: str = "month_ago",
    progress_callback=None,
) -> tuple[pd.DataFrame, dict]:
    """
    Main ETL entry point: fetch + preprocess all alert data.

    Fetches active alerts and optionally historical oblast-level data,
    then merges, cleans, and engineers features across both datasets.

    Note on history fetching rate limits:
        The history endpoint allows only 2 req/min. For 25 oblasts,
        this means ~12.5 minutes of sequential fetching with 30s delays.
        In practice, we fetch only oblast-level records (location_type == "oblast")
        to minimise API calls while maximising analytical coverage.

    Args:
        api_token:         Valid Bearer token for alerts.in.ua API.
        fetch_history:     Whether to fetch 30-day historical data.
        history_period:    Period string for history endpoint ("month_ago").
        progress_callback: Optional callable(current, total, label) for
                           Streamlit progress bar integration.

    Returns:
        Tuple of:
            - pd.DataFrame: Processed, feature-engineered alerts DataFrame.
            - dict: API status metadata {"active_count": int, "api_ok": bool,
                    "last_updated": datetime, "error": str|None}.

    Raises:
        RuntimeError: On authentication or critical network failure.
    """
    now = datetime.now(timezone.utc)
    all_records: list[dict] = []
    status: dict = {"api_ok": False, "active_count": 0, "last_updated": now, "error": None}

    # ── Step 1: Fetch active alerts ──────────────────────────────────────────
    try:
        active_raw = fetch_active_alerts(api_token)
        status["active_count"] = len(active_raw)
        status["api_ok"] = True
        for record in active_raw:
            all_records.append(_normalize_alert_record(record, now))
    except RuntimeError as exc:
        status["error"] = str(exc)
        status["api_ok"] = False
        raise

    # ── Step 2: Fetch historical data per oblast ─────────────────────────────
    if fetch_history:
        oblasts = list(OBLAST_UID_MAP.items())
        total = len(oblasts)
        for idx, (oblast_name, uid) in enumerate(oblasts):
            if progress_callback is not None:
                progress_callback(idx + 1, total, f"Loading {OBLAST_EN_MAP.get(oblast_name, oblast_name)}…")
            try:
                history_raw = fetch_oblast_history(api_token, uid, history_period)
                for record in history_raw:
                    # Skip records already in active set (avoid duplicates)
                    normalized = _normalize_alert_record(record, now)
                    all_records.append(normalized)
                # Respect 2 req/min rate limit: wait 31s between requests
                if idx < total - 1:
                    time.sleep(31)
            except RuntimeError as exc:
                logger.warning(
                    "Skipping history for %s (uid=%s): %s", oblast_name, uid, exc
                )
                continue

    # ── Step 3: Build DataFrame ───────────────────────────────────────────────
    if not all_records:
        return pd.DataFrame(), status

    df = pd.DataFrame(all_records)

    # Deduplicate by id + started_at (history and active can overlap)
    df = df.drop_duplicates(subset=["id", "started_at"], keep="first")

    # Convert started_at to UTC-aware pandas Timestamp for .dt accessor compatibility
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)

    # Drop rows with no start time — cannot be placed on timeline
    df = df.dropna(subset=["started_at"])

    # Sort chronologically
    df = df.sort_values("started_at", ascending=True).reset_index(drop=True)

    # ── Step 4: Feature engineering ───────────────────────────────────────────
    df = _engineer_features(df)

    # ── Step 5: Rolling aggregations (oblast-level) ───────────────────────────
    # Compute 7-day rolling frequency (filled daily, grouped by oblast)
    # This is used for the trend chart MA overlay
    df["date"] = pd.to_datetime(df["date"])

    status["last_updated"] = now
    return df, status


def compute_daily_counts(
    df: pd.DataFrame,
    oblast_filter: Optional[list[str]] = None,
    alert_type_filter: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate alert counts and average duration by date.

    Args:
        df:                 Processed alerts DataFrame from fetch_and_process_data.
        oblast_filter:      If provided, filter to only these English oblast names.
        alert_type_filter:  If provided, filter to only these alert_type_en values.

    Returns:
        DataFrame indexed by date with columns:
            count, avg_duration_minutes, total_duration_minutes, rolling_7d_avg
    """
    data = df.copy()

    if oblast_filter:
        data = data[data["location_oblast_en"].isin(oblast_filter)]
    if alert_type_filter:
        data = data[data["alert_type_en"].isin(alert_type_filter)]

    # Filter to oblast-level for clean non-overlapping counts
    data = data[data["location_type"] == "oblast"]

    daily = (
        data.groupby("date")
        .agg(
            count=("id", "count"),
            avg_duration_minutes=("duration_minutes", "mean"),
            total_duration_minutes=("duration_minutes", "sum"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    # Fill missing dates with zero counts for clean time-series
    if not daily.empty:
        full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = daily.set_index("date").reindex(full_range, fill_value=0).reset_index()
        daily.columns = ["date", "count", "avg_duration_minutes", "total_duration_minutes"]
        daily["rolling_7d_avg"] = daily["count"].rolling(window=7, min_periods=1).mean()

    return daily


def compute_heatmap_data(
    df: pd.DataFrame,
    oblast_filter: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Build a pivot table of alert counts by hour-of-day × day-of-week.

    Args:
        df:             Processed alerts DataFrame.
        oblast_filter:  If provided, filter to only these English oblast names.

    Returns:
        Pivot DataFrame with shape (7, 24): rows = day of week (Mon–Sun),
        columns = hour of day (0–23), values = alert count.
    """
    data = df.copy()
    if oblast_filter:
        data = data[data["location_oblast_en"].isin(oblast_filter)]

    data = data[data["location_type"] == "oblast"]

    if data.empty:
        return pd.DataFrame(
            np.zeros((7, 24), dtype=int),
            index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            columns=list(range(24)),
        )

    pivot = (
        data.groupby(["day_name", "hour_of_day"])
        .size()
        .reset_index(name="count")
        .pivot(index="day_name", columns="hour_of_day", values="count")
        .fillna(0)
    )

    # Ensure all days and hours present (fill missing with 0)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(day_order, fill_value=0)
    for hour in range(24):
        if hour not in pivot.columns:
            pivot[hour] = 0
    pivot = pivot[sorted(pivot.columns)]

    return pivot.astype(int)


def compute_vulnerability_score(
    df: pd.DataFrame,
    oblast_filter: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute a normalized vulnerability score per oblast.

    Score formula:
        vulnerability = (alert_count / max_count) * 0.6
                       + (avg_duration / max_avg_dur) * 0.4

    Args:
        df:             Processed alerts DataFrame.
        oblast_filter:  If provided, filter to only these English oblast names.

    Returns:
        DataFrame with columns: location_oblast_en, alert_count,
        avg_duration_minutes, vulnerability_score. Sorted descending by score.
    """
    data = df[df["location_type"] == "oblast"].copy()
    if oblast_filter:
        data = data[data["location_oblast_en"].isin(oblast_filter)]

    if data.empty:
        return pd.DataFrame(columns=["location_oblast_en", "alert_count", "avg_duration_minutes", "vulnerability_score"])

    agg = (
        data.groupby("location_oblast_en")
        .agg(
            alert_count=("id", "count"),
            avg_duration_minutes=("duration_minutes", "mean"),
        )
        .reset_index()
    )

    max_count = agg["alert_count"].max() or 1
    max_dur = agg["avg_duration_minutes"].max() or 1

    agg["vulnerability_score"] = (
        (agg["alert_count"] / max_count) * 0.6
        + (agg["avg_duration_minutes"].fillna(0) / max_dur) * 0.4
    ).round(4)

    return agg.sort_values("vulnerability_score", ascending=False).reset_index(drop=True)
