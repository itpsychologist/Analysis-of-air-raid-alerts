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

import numpy as np

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


# ---------------------------------------------------------------------------
# Target B: daily count / duration per unit (src/features/daily.py)
# ---------------------------------------------------------------------------
# Each maps the daily target matrix Yd (Kyiv days × units, NaN = not covered) to a
# forecast indexed by target day D for horizon k days. The origin is D - k + 1, so
# the latest usable day is D - k: every baseline shifts by >= k rows (days).


def daily_last_day(Yd: pd.DataFrame, k: int) -> pd.DataFrame:
    """Persistence: the last known day."""
    return Yd.shift(k)


def daily_seasonal_naive(Yd: pd.DataFrame, k: int) -> pd.DataFrame:
    """Same weekday one week before (known for k <= 7)."""
    assert k <= 7
    return Yd.shift(7)


def daily_mean(Yd: pd.DataFrame, k: int, window_days: int) -> pd.DataFrame:
    """Mean of the last `window_days` known days."""
    return Yd.shift(k).rolling(window_days, min_periods=window_days // 2).mean()


DAILY_BASELINES = {
    "seasonal_naive": daily_seasonal_naive,
    "last_day": daily_last_day,
    "mean_7d": lambda Yd, k: daily_mean(Yd, k, 7),
    "mean_28d": lambda Yd, k: daily_mean(Yd, k, 28),
}


def holt_winters_daily(Yd: pd.DataFrame, origins: pd.DatetimeIndex, max_k: int = 7,
                       history_days: int = 56) -> dict[int, dict[str, pd.DataFrame]]:
    """
    The dashboard's model (forecasting.train_and_forecast) refit per unit and origin
    on the last `history_days` known days before the origin (the trailing covered run;
    coverage is the same for all units). Returns {k: {"mean", "lower", "upper"}}
    indexed by target day; lower/upper are its 95% interval.
    """
    import logging

    from forecasting import train_and_forecast

    logging.getLogger("forecasting").setLevel(logging.ERROR)
    days = Yd.index
    out = {k: {q: pd.DataFrame(np.nan, index=days, columns=Yd.columns) for q in ("mean", "lower", "upper")}
           for k in range(1, max_k + 1)}
    known = Yd.notna().any(axis=1).to_numpy()
    for o in origins:
        i = days.get_loc(o)
        lo = i
        while lo > 0 and known[lo - 1] and i - lo < history_days:
            lo -= 1
        if i - lo < 3:
            continue
        hist = Yd.iloc[lo:i]
        index = pd.date_range("2000-01-01", periods=len(hist), freq="D", tz="UTC")  # values only
        for u in Yd.columns:
            fc = train_and_forecast(pd.Series(hist[u].to_numpy(), index=index), horizon=max_k)
            for k in range(1, max_k + 1):
                if i + k - 1 < len(days):
                    row = fc.iloc[k - 1]
                    for q, col in (("mean", "forecast"), ("lower", "ci_lower"), ("upper", "ci_upper")):
                        out[k][q].iat[i + k - 1, out[k][q].columns.get_loc(u)] = row[col]
    return out
