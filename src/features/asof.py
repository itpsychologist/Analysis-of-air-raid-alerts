"""
asof.py — Features "as of" the forecast origin t (PLAN.md, section 5.1).
=========================================================================
Every feature is a wide matrix (index = origin hours t, columns = units) and the
value at row t uses only:

  - panel hours strictly before t (rows <= t-1 of starts / active_minutes),
  - the alert state at the instant t (panel.active_at: started before t, not yet over),
  - labels of origins t' with t' + h <= t (through the baselines, which shift by h),
  - the calendar of t itself (known in advance).

Hours that were not covered by any source are NaN, never "no alert", and so are
features whose window reaches into them. tests/test_leakage.py perturbs everything
after t and checks that no feature at t changes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.features.geo import group_matrices
from src.features.panel import Panel
from src.models.baselines import hour_of_day_rate, recent_rate, unit_rate

KYIV_TZ = "Europe/Kyiv"
SINCE_CAP_HOURS = 168
UNIT_WINDOWS = (3, 6, 24, 168)
SPATIAL_WINDOWS = (1, 3)


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w == 1:
        return x
    return pd.DataFrame(x).rolling(w, min_periods=max(1, w // 2)).mean().to_numpy()


def _hours_since(event: np.ndarray, covered_before: np.ndarray) -> np.ndarray:
    """
    log1p(hours since the last hour with an event), counting hour t-1 as 0, capped
    at SINCE_CAP_HOURS. `event` and `covered_before` are aligned so that row t
    describes hour t-1. NaN when an uncovered hour is more recent than the last
    event (the true value is then unknown).
    """
    n = len(event)
    rows = np.arange(n)[:, None]
    last_event = pd.DataFrame(np.where(event, rows, np.nan)).ffill().to_numpy()
    last_hole = pd.Series(np.where(covered_before, np.nan, np.arange(n))).ffill().to_numpy()[:, None]
    since = rows - last_event
    since = np.where(np.isnan(since), SINCE_CAP_HOURS, np.minimum(since, SINCE_CAP_HOURS))
    unknown = ~np.isnan(last_hole) & ((np.isnan(last_event)) | (last_hole > last_event)) & (rows - last_hole < SINCE_CAP_HOURS)
    return np.where(unknown, np.nan, np.log1p(since))


def _country_mean(x: np.ndarray) -> np.ndarray:
    """Mean over all units, broadcast back to every unit (NaN rows stay NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows: uncovered hours
        return np.repeat(np.nanmean(x, axis=1, keepdims=True), x.shape[1], axis=1)


def calendar(hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Kyiv-local calendar of each origin (known in advance; DST handled by tz_convert)."""
    local = hours.tz_convert(KYIV_TZ)
    hour, dow = local.hour.to_numpy(), local.dayofweek.to_numpy()
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_night": ((hour >= 22) | (hour < 6)).astype(float),
    }, index=hours)


def history_features(panel: Panel, unit_oblast: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Horizon-independent features from the alert history (unit, oblast, neighbours, country)."""
    if panel.active_at is None:
        raise ValueError("panel has no active_at matrix; rebuild it with build_panel")
    hours, units = panel.hours, panel.units
    cov = panel.covered.to_numpy()
    cov_before = np.concatenate([[False], cov[:-1]])  # row t: was hour t-1 covered?

    def before(x: np.ndarray) -> np.ndarray:
        """Row t <- hour t-1, NaN where that hour was not covered."""
        shifted = np.vstack([np.full((1, x.shape[1]), np.nan), x[:-1]])
        return np.where(cov_before[:, None], shifted, np.nan)

    starts = before(panel.starts.to_numpy())
    active = before(panel.active_minutes.to_numpy() / 60)
    now = np.where(cov_before[:, None], panel.active_at.to_numpy(), np.nan)

    same, nbr = group_matrices(units, unit_oblast)
    f: dict[str, np.ndarray] = {
        "unit_active_now": now,
        "unit_starts_1h": starts,
        "unit_active_1h": active,
        "unit_since_start": _hours_since(np.nan_to_num(starts) > 0, cov_before),
        "unit_since_alert": _hours_since(np.nan_to_num(active) > 0, cov_before),
    }
    for w in UNIT_WINDOWS:
        f[f"unit_starts_{w}h"] = _rolling_mean(starts, w)
        f[f"unit_active_{w}h"] = _rolling_mean(active, w)

    # Spatial: same oblast, bordering oblasts, whole country
    for name, agg in [("oblast", lambda x: x @ same.T), ("nbr", lambda x: x @ nbr.T), ("country", _country_mean)]:
        f[f"{name}_active_now"] = agg(now)
        for w in SPATIAL_WINDOWS:
            f[f"{name}_starts_{w}h"] = agg(_rolling_mean(starts, w))
            f[f"{name}_active_{w}h"] = agg(_rolling_mean(active, w))
    # Intensity regime ("massive phase") over the last day and week
    for w in (24, 168):
        f[f"country_starts_{w}h"] = _country_mean(_rolling_mean(starts, w))
        f[f"country_active_{w}h"] = _country_mean(_rolling_mean(active, w))

    out = {k: pd.DataFrame(v, index=hours, columns=units) for k, v in f.items()}
    cal = calendar(hours)
    for c in cal:
        out[f"cal_{c}"] = pd.DataFrame(np.repeat(cal[[c]].to_numpy(), len(units), axis=1), index=hours, columns=units)
    return out


def label_features(Y: pd.DataFrame, h: int) -> dict[str, pd.DataFrame]:
    """Past label rates for horizon h (the baselines; each shifts labels by >= h rows)."""
    return {
        "rate_recent_24h": recent_rate(Y, h),
        "rate_unit_28d": unit_rate(Y, h),
        "rate_hour_of_day_28d": hour_of_day_rate(Y, h),
    }


def build_features(panel: Panel, unit_oblast: dict[str, str], Y: pd.DataFrame, h: int,
                   history: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """All features for horizon h; pass `history` to reuse history_features across horizons."""
    history = history if history is not None else history_features(panel, unit_oblast)
    return history | label_features(Y, h)


def stack(features: dict[str, pd.DataFrame], rows: np.ndarray) -> pd.DataFrame:
    """
    Long design matrix for the selected origin rows: one row per (origin, unit),
    ordered origin-major like `matrix[rows].ravel()`.
    """
    first = next(iter(features.values()))
    index = pd.MultiIndex.from_product([first.index[rows], first.columns], names=["origin", "unit"])
    return pd.DataFrame(
        {k: v.to_numpy()[rows].ravel().astype(np.float32) for k, v in features.items()}, index=index
    )
