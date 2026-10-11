"""
ETL fixes from PLAN.md stage 3: Kyiv-local calendar, censored durations, partial days.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from etl import _compute_duration, _engineer_features, compute_daily_counts, mean_finished_duration
from forecasting import prepare_series_for_region

U = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)  # noqa: E731


def test_calendar_fields_are_kyiv_local():
    df = _engineer_features(pd.DataFrame({"started_at": pd.to_datetime(["2026-10-10 22:30", "2026-01-15 21:59"], utc=True)}))
    # UTC+3 in summer time, UTC+2 in winter
    assert str(df["date"].iloc[0]) == "2026-10-11" and df["hour_of_day"].iloc[0] == 1
    assert str(df["date"].iloc[1]) == "2026-01-15" and df["hour_of_day"].iloc[1] == 23
    assert df["day_name"].iloc[0] == "Sunday"


def test_duration_never_reaches_past_as_of():
    now = U("2026-10-11 10:00")
    assert _compute_duration(U("2026-10-11 09:00"), None, now) == 60  # active: censored at now
    assert _compute_duration(U("2026-10-11 09:00"), U("2026-10-11 09:30"), now) == 30
    assert _compute_duration(U("2026-10-11 09:00"), U("2026-10-11 12:00"), now) == 60
    assert _compute_duration(None, None, now) is None


def alerts(as_of="2026-10-11 10:00") -> pd.DataFrame:
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "location_type": "oblast",
        "location_oblast_en": ["Kyiv Oblast", "Kyiv Oblast", "Kharkiv Oblast", "Kharkiv Oblast", "Kyiv Oblast"],
        "alert_type_en": "Air Raid",
        "started_at": pd.to_datetime(["2026-10-08 14:00", "2026-10-09 08:00", "2026-10-09 09:00",
                                      "2026-10-10 05:00", "2026-10-11 08:00"], utc=True),
        "is_active": [False, False, False, False, True],
        "duration_minutes": [30.0, 90.0, 60.0, 20.0, 120.0],
    })
    df = _engineer_features(df)
    df["date"] = pd.to_datetime(df["date"])
    df.attrs["as_of"] = pd.Timestamp(as_of, tz="UTC")
    return df


def test_daily_counts_flag_partial_days_and_ignore_censored_durations():
    daily = compute_daily_counts(alerts()).set_index("date")
    assert list(daily.index.strftime("%Y-%m-%d")) == ["2026-10-08", "2026-10-09", "2026-10-10", "2026-10-11"]
    assert list(daily["is_complete"]) == [False, True, True, False]  # window start, today
    assert daily.loc["2026-10-11", "count"] == 1
    assert pd.isna(daily.loc["2026-10-11", "avg_duration_minutes"])  # only an active alert
    assert daily.loc["2026-10-09", "avg_duration_minutes"] == 75


def test_quiet_region_gets_zero_days_over_the_whole_window():
    daily = compute_daily_counts(alerts(), oblast_filter=["Kharkiv Oblast"]).set_index("date")
    assert len(daily) == 4
    assert daily.loc["2026-10-08", "count"] == 0


def test_forecast_series_uses_complete_days_only():
    series = prepare_series_for_region(compute_daily_counts(alerts()))
    assert list(series.index.strftime("%Y-%m-%d")) == ["2026-10-09", "2026-10-10"]
    assert list(series) == [2.0, 1.0]


def test_mean_duration_excludes_active_alerts():
    assert mean_finished_duration(alerts()) == pytest.approx((30 + 90 + 60 + 20) / 4)
