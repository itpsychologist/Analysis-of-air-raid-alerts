"""
ETL fixes from PLAN.md stage 3: Kyiv-local calendar, censored durations, partial days.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from etl import (
    OBLAST_EN_MAP, RAION_ALERT, _compute_duration, _engineer_features, compute_daily_counts,
    compute_heatmap_data, compute_vulnerability_score, mean_finished_duration, raion_alerts, raion_count,
    raion_units,
)
from forecasting import compute_risk_forecast_score, prepare_series_for_region

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
        "location_type": RAION_ALERT,  # already raion alerts (see raion_alerts below)
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



# ---------------------------------------------------------------------------
# Raion alerts: since Dec 2025 alerts are mostly raion-level, so every aggregate
# counts raion alerts, not oblast-level records
# ---------------------------------------------------------------------------

KHARKIV = "Харківська область"


def record(i, level, title, start, end, oblast=KHARKIV, alert_type="Air Raid"):
    return {"id": i, "location_type": level, "location_title": title, "location_oblast": oblast,
            "location_oblast_en": OBLAST_EN_MAP.get(oblast, oblast), "alert_type_en": alert_type,
            "location_raion": None,
            "started_at": pd.Timestamp(start, tz="UTC"),
            "finished_at": pd.Timestamp(end, tz="UTC") if end else pd.NaT}


def raw(rows, as_of="2026-10-11 10:00"):
    df = _engineer_features(pd.DataFrame(rows))
    df["date"] = pd.to_datetime(df["date"])
    df.attrs["as_of"] = pd.Timestamp(as_of, tz="UTC")
    return df


def test_oblast_alert_reaches_every_raion_and_overlaps_merge():
    raions = raion_units()[KHARKIV]
    df = raw([
        record(1, "oblast", KHARKIV, "2026-10-09 08:00", "2026-10-09 09:00"),
        record(2, "raion", "Харківський район", "2026-10-09 08:30", "2026-10-09 10:00"),  # overlaps → merged
        record(3, "raion", "Ізюмський район", "2026-10-09 12:00", "2026-10-09 12:30"),
        record(4, "hromada", "Балаклійська територіальна громада", "2026-10-09 13:00", "2026-10-09 14:00"),
        record(5, "oblast", "Луганська область", "2022-04-04 00:00", None, oblast="Луганська область"),
    ])
    ra = raion_alerts(df)
    assert (ra["location_type"] == RAION_ALERT).all()
    assert len(ra) == len(raions) + 1  # the oblast alert per raion, plus Izium's second alert
    kh = ra[ra["location_title"] == "Харківський район"].iloc[0]
    assert kh["n_events"] == 2 and kh["duration_minutes"] == 120  # 08:00–10:00, merged
    assert "Луганська область" not in set(ra["location_oblast"])  # permanent alert excluded
    assert set(ra["location_oblast_en"]) == {OBLAST_EN_MAP[KHARKIV]}


def test_active_alert_is_censored_at_fetch_time():
    ra = raion_alerts(raw([record(1, "raion", "Ізюмський район", "2026-10-11 09:00", None)]))
    row = ra.iloc[0]
    assert bool(row["is_active"]) and pd.isna(row["finished_at"]) and row["duration_minutes"] == 60


def test_raion_names_are_normalised_and_types_kept_apart_on_request():
    rows = [record(1, "raion", "Куп’янський район", "2026-10-09 08:00", "2026-10-09 09:00"),
            record(2, "raion", "Куп'янський район", "2026-10-09 08:30", "2026-10-09 09:30", alert_type="Artillery")]
    assert len(raion_alerts(raw(rows))) == 1  # same raion, overlapping: one alert
    by_type = raion_alerts(raw(rows), by_type=True)
    assert sorted(by_type["alert_type_en"]) == ["Air Raid", "Artillery"]


def test_aggregates_count_raion_alerts_from_raw_records():
    df = raw([record(1, "raion", "Ізюмський район", "2026-10-09 08:00", "2026-10-09 09:00"),
              record(2, "raion", "Харківський район", "2026-10-09 08:10", "2026-10-09 08:40"),
              record(3, "hromada", "Балаклійська територіальна громада", "2026-10-09 13:00", "2026-10-09 14:00")])
    daily = compute_daily_counts(df).set_index("date")
    assert daily.loc["2026-10-09", "count"] == 2  # both raions, the hromada alert not counted
    assert compute_heatmap_data(df).to_numpy().sum() == 2


def test_vulnerability_is_per_raion():
    # One raion alert in Kyiv City (1 unit) vs one in Kharkiv oblast (many raions)
    df = raw([record(1, "raion", "Ізюмський район", "2026-10-09 08:00", "2026-10-09 09:00"),
              record(2, "oblast", "м. Київ", "2026-10-09 08:00", "2026-10-09 09:00", oblast="м. Київ")])
    v = compute_vulnerability_score(df).set_index("location_oblast_en")
    kyiv = OBLAST_EN_MAP["м. Київ"]
    assert v.loc[kyiv, "alerts_per_raion"] == 1
    assert v.loc[OBLAST_EN_MAP[KHARKIV], "alerts_per_raion"] == pytest.approx(1 / raion_count([OBLAST_EN_MAP[KHARKIV]]))
    assert v.index[0] == kyiv


def test_raion_units_and_risk_per_raion():
    assert raion_count() == 119  # 118 raions + Kyiv City (Luhansk excluded)
    fc = pd.DataFrame({"forecast": [119.0, 119.0]})
    assert compute_risk_forecast_score(fc, n_units=119) == pytest.approx(0.5)
