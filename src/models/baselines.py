"""
baselines.py — Reference forecasts any model has to beat (PLAN.md, section 4.2).
================================================================================
Each baseline maps the label matrix Y (origins × units, from features.panel.labels)
to a probability matrix of the same shape.

No-leakage rule: the label of origin t' is only known at t' + h, so a forecast
made at t may use Y[t'] only for t' <= t - h. Every baseline therefore shifts Y by
at least h rows before aggregating. Gap hours are NaN and are skipped by the
rolling means (they reduce the sample, they are never read as "no alert").
"""

from __future__ import annotations

import math

import pandas as pd

HOURS_PER_WEEK = 168


def seasonal_naive(Y: pd.DataFrame, h: int) -> pd.DataFrame:
    """Same origin hour one week earlier."""
    _check_hourly(Y.index)
    assert h <= HOURS_PER_WEEK
    return Y.shift(HOURS_PER_WEEK)


def recent_rate(Y: pd.DataFrame, h: int, window_hours: int = 24) -> pd.DataFrame:
    """Share of positive labels among the last `window_hours` known origins ("persistence")."""
    _check_hourly(Y.index)
    return Y.shift(h).rolling(window_hours, min_periods=window_hours // 2).mean()


def unit_rate(Y: pd.DataFrame, h: int, window_days: int = 28) -> pd.DataFrame:
    """Per-unit share of positive labels over the last `window_days` days."""
    _check_hourly(Y.index)
    w = window_days * 24
    return Y.shift(h).rolling(w, min_periods=w // 2).mean()


def hour_of_day_rate(Y: pd.DataFrame, h: int, window_days: int = 28) -> pd.DataFrame:
    """
    Per-unit share of positive labels at the same hour of day over the last
    `window_days` days ("historical frequency" in PLAN.md).

    Uses exact 24-hour lags of the regular hourly index rather than grouping by
    local Kyiv hour: on the autumn DST change a local hour occurs twice, one hour
    apart, and a "previous same hour" could then be a label not yet known at t.
    The first lag used is ceil(h / 24) days, so every used origin t' has t' + h <= t.
    """
    _check_hourly(Y.index)
    first_lag = math.ceil(h / 24)
    lags = [Y.shift(24 * k) for k in range(first_lag, first_lag + window_days)]
    total = sum(lag.fillna(0.0) for lag in lags)
    count = sum(lag.notna().astype(float) for lag in lags)
    return (total / count).where(count >= window_days // 2)


def _check_hourly(index: pd.DatetimeIndex) -> None:
    if len(index) > 1 and not (index[1:] - index[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("baselines need a regular hourly index (shift = hours)")


BASELINES = {
    "seasonal_naive": seasonal_naive,
    "recent_rate_24h": recent_rate,
    "unit_rate_28d": unit_rate,
    "hour_of_day_rate_28d": hour_of_day_rate,
}


def predict_all(Y: pd.DataFrame, h: int) -> dict[str, pd.DataFrame]:
    return {name: fn(Y, h) for name, fn in BASELINES.items()}
