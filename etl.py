"""
etl.py — Data Acquisition & Preprocessing Module
==================================================
Resilient ETL pipeline for alerts.in.ua API.

Rate limits:
  - /v1/alerts/active.json          → 8-12 req/min  (cached 5 min)
  - /v1/regions/{uid}/alerts/...    → 2 req/min      (cached 5 min, fetched once)

Time conventions (PLAN.md 1.2, 3.1):
  - timestamps stay UTC; calendar fields (date, hour, weekday) are Europe/Kyiv local time
  - duration_minutes is what is known at the fetch time `as_of`: for alerts still
    active it is censored (is_active=True) and is excluded from average durations
  - daily aggregates flag days that are not fully observed (is_complete=False)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

KYIV_TZ = "Europe/Kyiv"

# ---------------------------------------------------------------------------
# Oblast UID mapping (oblast-level only, as per the API documentation)
# Source: https://devs.alerts.in.ua — location_uid for oblast-type locations
# (verified 2026-10-11 against the docs table and live API responses)
# ---------------------------------------------------------------------------
OBLAST_UID_MAP: dict[str, str] = {
    "Вінницька область": "4",
    "Волинська область": "8",
    "Дніпропетровська область": "9",
    "Донецька область": "28",
    "Житомирська область": "10",
    "Закарпатська область": "11",
    "Запорізька область": "12",
    "Івано-Франківська область": "13",
    "Київська область": "14",
    "Кіровоградська область": "15",
    "Луганська область": "16",
    "Львівська область": "27",
    "Миколаївська область": "17",
    "Одеська область": "18",
    "Полтавська область": "19",
    "Рівненська область": "5",
    "Сумська область": "20",
    "Тернопільська область": "21",
    "Харківська область": "22",
    "Херсонська область": "23",
    "Хмельницька область": "3",
    "Черкаська область": "24",
    "Чернівецька область": "26",
    "Чернігівська область": "25",
    "м. Київ": "31",
}

# Human-readable English names for UI display
OBLAST_EN_MAP: dict[str, str] = {
    "Вінницька область": "Vinnytsia",
    "Волинська область": "Volyn",
    "Дніпропетровська область": "Dnipro",
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
    Alert duration in minutes as known at `now`: min(finished, now) - started.

    For an alert still active at `now` this is a lower bound (censored), not
    its duration; the record's is_active flag marks that case. Nothing after
    `now` is ever used.

    Args:
        started:  Alert start datetime (UTC-aware).
        finished: Alert end datetime (UTC-aware), or None if still active.
        now:      The as-of time (fetch time).

    Returns:
        Duration in minutes as float, or None if start is unknown.
    """
    if started is None:
        return None
    end = min(finished, now) if finished is not None else now
    delta = end - started
    minutes = delta.total_seconds() / 60.0
    # Guard against negative durations (data quality issue)
    return max(0.0, minutes)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical and categorical time-series features to the DataFrame.
    All calendar fields are Kyiv local time (a UTC date shifts the day by 2-3 hours).

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
    dt = df["started_at"].dt.tz_convert(KYIV_TZ).dt
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
        The history endpoint allows only 2 req/min. For the 24 counted oblasts
        (Luhansk is skipped), this means ~12 minutes of sequential fetching with
        31 s delays. Each response holds alerts of every level (oblast, raion,
        hromada, city); the aggregates count raion alerts (see raion_alerts).

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
    status: dict = {
        "api_ok": False,
        "active_count": 0,
        "last_updated": now,
        "error": None,
        "history_failures": {},  # oblast_en -> error message, for regions whose history fetch failed
    }

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
        # Only oblasts the aggregates count: Luhansk (permanent alert since 2022) is
        # excluded from every figure, so its history request would only cost 31 s
        counted = set(raion_units())
        oblasts = [(name, uid) for name, uid in OBLAST_UID_MAP.items() if name in counted]
        total = len(oblasts)
        for idx, (oblast_name, uid) in enumerate(oblasts):
            if progress_callback is not None:
                progress_callback(idx + 1, total, f"Loading {OBLAST_EN_MAP.get(oblast_name, oblast_name)}…")
            try:
                history_raw = fetch_oblast_history(api_token, uid, history_period)
                if not history_raw:
                    # Empty response is not necessarily an error (region may simply
                    # have had zero alerts), but we still note it for diagnostics.
                    status["history_failures"].setdefault(
                        OBLAST_EN_MAP.get(oblast_name, oblast_name),
                        "No records returned (may be genuinely zero alerts, or a silent API issue).",
                    )
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
                status["history_failures"][OBLAST_EN_MAP.get(oblast_name, oblast_name)] = str(exc)
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

    # The fetch time: the last day is only observed up to here (see compute_daily_counts)
    df.attrs["as_of"] = pd.Timestamp(now)
    # Start of the observed history (month_ago = the last 30 days); before it only alerts
    # still running at fetch time are known, so counts before this point are incomplete
    df.attrs["history_start"] = pd.Timestamp(now - timedelta(days=HISTORY_DAYS)) if fetch_history else pd.Timestamp(now)
    status["last_updated"] = now
    return df, status


# ---------------------------------------------------------------------------
# Raion alerts: the unit every aggregate below counts
# ---------------------------------------------------------------------------
# Since December 2025 alerts are declared mostly per raion, not per oblast, so
# counting only location_type == "oblast" records misses most alerts. Every
# aggregate counts *raion alerts* instead: an oblast-level alert puts each raion
# of the oblast under alert, a raion-level alert its own raion, and overlapping
# announcements for one raion are merged (the same unit as the forecasting study,
# src/data/merge.py). Hromada / city-level alerts are not counted; Luhansk oblast
# (permanent alert since 2022) is excluded.

RAION_ALERT = "raion_alert"  # location_type of rows produced by raion_alerts()
HISTORY_DAYS = 30  # the month_ago history endpoint
RAION_UNITS_PATH = Path(__file__).resolve().parent / "src" / "data" / "raion_units.json"


def raion_units() -> dict[str, list[str]]:
    """Raions per oblast (Ukrainian names; Kyiv City has none and is its own unit)."""
    return json.loads(RAION_UNITS_PATH.read_text(encoding="utf-8"))


def counted_oblasts_en() -> list[str]:
    """English names of the oblasts the aggregates count (Luhansk is excluded), sorted."""
    return sorted(OBLAST_EN_MAP.get(ob, ob) for ob in raion_units())


def raion_count(oblasts_en: Optional[list[str]] = None) -> int:
    """Number of raion units in the given oblasts (English names), or in the whole country."""
    units = raion_units()
    wanted = None if oblasts_en is None else set(oblasts_en)
    return sum(max(len(rs), 1) for ob, rs in units.items()
               if wanted is None or OBLAST_EN_MAP.get(ob, ob) in wanted)


def raion_alerts(df: pd.DataFrame, as_of: Optional[datetime] = None, by_type: bool = False) -> pd.DataFrame:
    """
    Raw API records -> one row per raion alert, in the columns the aggregates use.

    Args:
        df:      fetch_and_process_data() output.
        as_of:   fetch time (defaults to df.attrs["as_of"]); alerts still running are
                 censored here and flagged is_active.
        by_type: keep alert types apart (for the type breakdown); otherwise a raion
                 under any alert counts once.

    Returns:
        DataFrame with location_type = "raion_alert", location_oblast(_en),
        location_title / location_raion (the raion), alert_type_en, started_at,
        finished_at (NaT while active), is_active, duration_minutes, n_events and the
        calendar features of _engineer_features. attrs["as_of"] is kept.
    """
    from src.data.merge import RAION_RENAMES, expand_to_raions, union_intervals

    columns = ["id", "location_type", "location_oblast", "location_oblast_en", "location_title",
               "location_raion", "alert_type_en", "started_at", "finished_at", "is_active",
               "duration_minutes", "n_events"]
    as_of = as_of if as_of is not None else df.attrs.get("as_of")
    if df.empty or "location_type" not in df:
        out = pd.DataFrame(columns=columns + ["date", "hour_of_day", "day_name"])
        out.attrs["as_of"] = as_of
        return out
    if as_of is None:
        as_of = df["started_at"].max()
    as_of = pd.Timestamp(as_of)

    ev = df[df["location_type"].isin(["oblast", "raion"])]
    raion_name = (ev["location_title"].astype("string").str.replace("’", "'", regex=False)
                  .replace(RAION_RENAMES))
    events = pd.DataFrame({
        "oblast": ev["location_oblast"],
        "raion": raion_name.where(ev["location_type"] == "raion"),
        "level": ev["location_type"],
        "started_at": pd.to_datetime(ev["started_at"], utc=True),
        "finished_at": pd.to_datetime(ev["finished_at"], utc=True),
        "type": ev["alert_type_en"] if by_type else "any",
    })
    raion_map = {ob: list(rs) for ob, rs in raion_units().items()}
    seen = events.loc[events["level"] == "raion", ["oblast", "raion"]].dropna().drop_duplicates()
    for ob, r in seen.itertuples(index=False):  # a raion missing from the static list still counts
        if r not in raion_map.setdefault(ob, []):
            raion_map[ob].append(r)

    parts = []
    for alert_type, g in events.groupby("type"):
        units = expand_to_raions(g, raion_map).assign(type=alert_type)
        parts.append(units)
    if not parts:
        out = pd.DataFrame(columns=columns + ["date", "hour_of_day", "day_name"])
        out.attrs["as_of"] = as_of
        return out
    units = pd.concat(parts, ignore_index=True)
    units["key"] = units["unit"] + "\x1f" + units["type"] + "\x1f" + units["oblast"]
    merged = union_intervals(units[["key", "started_at", "finished_at"]], "key", open_until=as_of)
    merged[["unit", "type", "oblast"]] = merged["key"].str.split("\x1f", expand=True)

    out = pd.DataFrame({
        "id": np.arange(len(merged)),
        "location_type": RAION_ALERT,
        "location_oblast": merged["oblast"],
        "location_oblast_en": merged["oblast"].map(lambda o: OBLAST_EN_MAP.get(o, o)),
        "location_title": merged["unit"],
        "location_raion": merged["unit"],
        "alert_type_en": merged["type"] if by_type else "Any",
        "started_at": merged["started_at"],
        "finished_at": merged["finished_at"].where(merged["finished_at"] < as_of),
        "is_active": merged["finished_at"] >= as_of,
        "duration_minutes": (merged["finished_at"].clip(upper=as_of) - merged["started_at"]).dt.total_seconds() / 60,
        "n_events": merged["n_events"],
    })
    out = _engineer_features(out.sort_values("started_at").reset_index(drop=True))
    out["date"] = pd.to_datetime(out["date"])
    out.attrs = {**df.attrs, "as_of": as_of}
    return out


def _raion_level(df: pd.DataFrame) -> pd.DataFrame:
    """Raion alerts as they are, raw records converted (see raion_alerts)."""
    if "location_type" in df and not df.empty and (df["location_type"] == RAION_ALERT).all():
        return df
    return raion_alerts(df)


def mean_finished_duration(df: pd.DataFrame) -> float:
    """Mean duration (minutes) of finished alerts; active ones are censored at `as_of`."""
    finished = df.loc[~df["is_active"].astype(bool), "duration_minutes"] if "is_active" in df else df["duration_minutes"]
    return float(finished.mean()) if finished.notna().any() else 0.0


def compute_daily_counts(
    df: pd.DataFrame,
    oblast_filter: Optional[list[str]] = None,
    alert_type_filter: Optional[list[str]] = None,
    as_of: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Aggregate alert counts and average duration by Kyiv-local date.

    Args:
        df:                 raion_alerts() output, or raw records (converted here).
        oblast_filter:      If provided, filter to only these English oblast names.
        alert_type_filter:  If provided, filter to only these alert_type_en values.
        as_of:              Fetch time; defaults to df.attrs["as_of"] (set by the ETL).

    Returns:
        DataFrame with columns:
            date, count (raion alerts started that Kyiv day), avg_duration_minutes (finished alerts only),
            total_duration_minutes (known part, up to as_of), rolling_7d_avg,
            is_complete — False for the first day (the history window starts
            mid-day) and for the day of as_of (still running). Forecasting must
            use complete days only.
    """
    as_of = as_of if as_of is not None else df.attrs.get("as_of")
    if as_of is None and not df.empty and "started_at" in df:
        # Conservative fallback: treat the day of the latest alert as still running
        logger.warning("compute_daily_counts: no as_of; marking the last day incomplete")
        as_of = df["started_at"].max()
    # Count raion alerts (an oblast-level record alone misses most alerts since Dec 2025)
    data = _raion_level(df) if not alert_type_filter else raion_alerts(df, by_type=True)

    if oblast_filter:
        data = data[data["location_oblast_en"].isin(oblast_filter)]
    if alert_type_filter:
        data = data[data["alert_type_en"].isin(alert_type_filter)]

    finished_duration = data["duration_minutes"].where(~data["is_active"].astype(bool)) if "is_active" in data else data["duration_minutes"]
    daily = (
        data.assign(finished_duration=finished_duration)
        .groupby("date")
        .agg(
            count=("id", "count"),
            avg_duration_minutes=("finished_duration", "mean"),
            total_duration_minutes=("duration_minutes", "sum"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    # Gap-fill against the full unfiltered dataset's range: the window the data
    # covers does not depend on the filter, and a quiet region (few or zero
    # alerts) must still get its zero days. An empty frame here would propagate
    # into prepare_series_for_region() as a misleading "insufficient data".
    if not df.empty and "date" in df.columns:
        full_dates = pd.to_datetime(df["date"])
        range_start, range_end = full_dates.min(), full_dates.max()
    elif not daily.empty:
        range_start, range_end = daily["date"].min(), daily["date"].max()
    else:
        range_start, range_end = None, None

    if range_start is not None and range_end is not None:
        full_range = pd.date_range(range_start, range_end, freq="D")
        daily = (
            daily.set_index("date")
            .reindex(full_range, fill_value=0)
            .reset_index()
        )
        daily.columns = ["date", "count", "avg_duration_minutes", "total_duration_minutes"]
        daily["rolling_7d_avg"] = daily["count"].rolling(window=7, min_periods=1).mean()
        daily["is_complete"] = daily["date"] > daily["date"].min()
        if as_of is not None:
            as_of_date = pd.Timestamp(as_of).tz_convert(KYIV_TZ).tz_localize(None).normalize()
            daily["is_complete"] &= daily["date"] < as_of_date
    else:
        # Truly no data anywhere (e.g. df itself is empty) — return a
        # well-formed empty frame with the expected columns rather than a
        # bare DataFrame() that's missing 'rolling_7d_avg'.
        daily = pd.DataFrame(
            columns=["date", "count", "avg_duration_minutes", "total_duration_minutes", "rolling_7d_avg", "is_complete"]
        )

    return daily


def compute_heatmap_data(
    df: pd.DataFrame,
    oblast_filter: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Build a pivot table of raion-alert starts by hour-of-day × day-of-week (Kyiv time).

    Args:
        df:             raion_alerts() output, or raw records (converted here).
        oblast_filter:  If provided, filter to only these English oblast names.

    Returns:
        Pivot DataFrame with shape (7, 24): rows = day of week (Mon–Sun),
        columns = hour of day (0–23), values = alert count.
    """
    data = _raion_level(df)
    if oblast_filter:
        data = data[data["location_oblast_en"].isin(oblast_filter)]

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
    Compute a normalized vulnerability score per oblast, per raion so that oblasts
    with many raions are not favoured.

    Score formula:
        vulnerability = (alerts_per_raion / max) * 0.6
                       + (share_under_alert / max) * 0.4

    where alerts_per_raion = raion alerts / raions in the oblast, and
    share_under_alert = raion-hours under alert / (raions x hours in the window).

    Args:
        df:             raion_alerts() output, or raw records (converted here).
        oblast_filter:  If provided, filter to only these English oblast names.

    Returns:
        DataFrame with columns: location_oblast_en, alert_count, alerts_per_raion,
        share_under_alert, avg_duration_minutes, vulnerability_score. Sorted descending by score.
    """
    cols = ["location_oblast_en", "alert_count", "alerts_per_raion", "share_under_alert",
            "avg_duration_minutes", "vulnerability_score"]
    data = _raion_level(df)
    if oblast_filter:
        data = data[data["location_oblast_en"].isin(oblast_filter)]
    if data.empty:
        return pd.DataFrame(columns=cols)

    as_of = data.attrs.get("as_of") or data["started_at"].max()
    window_minutes = max((pd.Timestamp(as_of) - data["started_at"].min()).total_seconds() / 60, 1.0)
    finished_duration = data["duration_minutes"].where(~data["is_active"].astype(bool))
    agg = (
        data.assign(finished_duration=finished_duration)
        .groupby("location_oblast_en")
        .agg(
            alert_count=("id", "count"),
            minutes=("duration_minutes", "sum"),
            avg_duration_minutes=("finished_duration", "mean"),
        )
        .reset_index()
    )
    n_raions = agg["location_oblast_en"].map(lambda ob: raion_count([ob])).clip(lower=1)
    agg["alerts_per_raion"] = agg["alert_count"] / n_raions
    agg["share_under_alert"] = (agg["minutes"] / (n_raions * window_minutes)).clip(upper=1)

    max_rate = agg["alerts_per_raion"].max() or 1
    max_share = agg["share_under_alert"].max() or 1
    agg["vulnerability_score"] = (
        (agg["alerts_per_raion"] / max_rate) * 0.6
        + (agg["share_under_alert"] / max_share) * 0.4
    ).round(4)

    return agg[cols].sort_values("vulnerability_score", ascending=False).reset_index(drop=True)
