import numpy as np
import pandas as pd
import pytest

from src.models.baselines import hour_of_day_rate, recent_rate, seasonal_naive, unit_rate


def Y_from(values, start="2026-01-01"):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(values), freq="h")
    return pd.DataFrame({"u": np.asarray(values, dtype=float)}, index=idx)


def test_recent_rate_uses_only_labels_known_h_hours_ago():
    Y = Y_from([1] * 12 + [0] * 12 + [1] * 3)
    p = recent_rate(Y, h=3, window_hours=24)["u"]
    # at row 26 the known labels are rows 0..23 -> 12 of 24 positive
    assert p.iloc[26] == pytest.approx(0.5)
    assert np.isnan(p.iloc[5])  # fewer than window/2 known labels


def test_seasonal_naive_is_last_week_same_hour():
    Y = Y_from(np.arange(200) % 2)
    assert seasonal_naive(Y, h=1)["u"].iloc[180] == Y["u"].iloc[180 - 168]


def test_hour_of_day_rate_averages_same_hour_on_previous_days():
    values = np.zeros(24 * 30)
    values[np.arange(10, len(values), 24)] = 1.0  # always an alert start at 10:00 UTC
    p = hour_of_day_rate(Y_from(values), h=1, window_days=28)["u"]
    assert p.iloc[24 * 29 + 10] == 1.0 and p.iloc[24 * 29 + 11] == 0.0
    assert np.isnan(p.iloc[24 * 5 + 10])  # fewer than 14 previous days


def test_unit_rate_and_irregular_index_is_rejected():
    Y = Y_from([1.0, 0.0] * 24 * 20)
    assert unit_rate(Y, h=1, window_days=14)["u"].iloc[-1] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="regular hourly"):
        recent_rate(Y.drop(Y.index[5]), h=1)
