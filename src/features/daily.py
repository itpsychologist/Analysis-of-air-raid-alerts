"""
daily.py — Daily unit panel, target B and its "as of origin" features.
=======================================================================
Target B (PLAN.md 1.1): for unit u and Kyiv-local day D,

  count[D, u]     number of alert starts in D
  duration[D, u]  hours under alert in D (an alert crossing midnight is split)

NaN when any hour of D is not covered. Days follow Europe/Kyiv (23 or 25 hours on
DST changes), built from the hourly panel, so the leakage rules carry over.

A forecast is made at the start of day d (Kyiv midnight, the origin) for target
days D = d + k - 1, horizon k = 1..7 days. At the origin the latest known day is
d - 1, so:

  - daily features at row d use days <= d - 1 plus the hourly "as of t" features
    at the origin hour (src/features/asof.py);
  - for horizon k the design row of target day D is features row D - k + 1
    (`for_horizon`), plus the calendar of D and the same weekday a week before D
    (D - 7 <= d - 1 for k <= 7);
  - a model scored on target days from S on may train on target days <= S - k
    (`DaySplit.train_end`), exactly the labels known at the first origin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.asof import KYIV_TZ, history_features
from src.features.geo import group_matrices
from src.features.panel import Panel

MAX_HORIZON_DAYS = 7
# Hourly as-of features taken at the origin hour (Kyiv midnight)
ORIGIN_HOURLY = (
    "unit_active_now", "unit_active_24h", "unit_active_168h", "unit_starts_24h", "unit_since_alert",
    "nbr_active_now", "country_active_now", "country_active_24h", "country_active_168h",
)
TARGETS = ("count", "duration")


@dataclass
class DailyPanel:
    count: pd.DataFrame  # index: Kyiv dates (tz-naive midnight), columns: units; NaN = not covered
    duration: pd.DataFrame
    origin_hour: pd.Series  # UTC hour of each day's Kyiv midnight

    @property
    def days(self) -> pd.DatetimeIndex:
        return self.count.index

    def target(self, name: str) -> pd.DataFrame:
        return getattr(self, name)


def build_daily(panel: Panel) -> DailyPanel:
    hours = panel.hours
    day = pd.DatetimeIndex(hours.tz_convert(KYIV_TZ).tz_localize(None).normalize(), name="day")
    n_hours = pd.Series(1, index=hours).groupby(day).sum()
    days = pd.date_range(day.min(), day.max(), freq="D", name="day")
    midnight = days.tz_localize(KYIV_TZ)
    # Calendar next midnight (a fixed 24 h shift would be wrong on DST days: 23 / 25 h)
    next_midnight = (days + pd.Timedelta(days=1)).tz_localize(KYIV_TZ)
    expected = pd.Series((next_midnight - midnight) / pd.Timedelta(hours=1), index=days)
    covered = (panel.covered.groupby(day).all() & (n_hours == expected.reindex(n_hours.index))).reindex(days, fill_value=False)

    def per_day(x: pd.DataFrame) -> pd.DataFrame:
        out = x.groupby(day).sum().reindex(days)
        return out.where(covered, axis=0)

    return DailyPanel(
        count=per_day(panel.starts),
        duration=per_day(panel.active_minutes / 60),
        origin_hour=pd.Series(midnight.tz_convert("UTC"), index=days),
    )


def _mean(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return x.rolling(w, min_periods=max(1, w // 2)).mean()


def origin_features(daily: DailyPanel, panel: Panel, unit_oblast: dict[str, str],
                    hourly: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """Features known at the start of each day (row d uses days <= d - 1 and hours < origin)."""
    units = list(daily.count.columns)
    hourly = hourly if hourly is not None else history_features(panel, unit_oblast)
    f: dict[str, pd.DataFrame] = {}
    for name in ORIGIN_HOURLY:
        f[f"h_{name}"] = hourly[name].reindex(daily.origin_hour.to_numpy()).set_axis(daily.days)

    same, nbr = group_matrices(units, unit_oblast)
    for target in TARGETS:
        prev = daily.target(target).shift(1)  # row d <- day d - 1
        f[f"{target}_1d"] = prev
        f[f"{target}_2d"] = prev.shift(1)
        for w in (7, 28):
            f[f"{target}_{w}d"] = _mean(prev, w)
        for name, M in (("oblast", same), ("nbr", nbr)):
            for w in (1, 7):
                m = prev if w == 1 else _mean(prev, w)
                f[f"{name}_{target}_{w}d"] = pd.DataFrame(m.to_numpy() @ M.T, index=daily.days, columns=units)
        country = _mean(prev, 7).mean(axis=1)
        f[f"country_{target}_7d"] = pd.DataFrame(np.repeat(country.to_numpy()[:, None], len(units), 1),
                                                 index=daily.days, columns=units)
    return f


def for_horizon(F: dict[str, pd.DataFrame], daily: DailyPanel, k: int) -> dict[str, pd.DataFrame]:
    """Design matrices indexed by target day D for horizon k (origin D - k + 1)."""
    if not 1 <= k <= MAX_HORIZON_DAYS:
        raise ValueError(f"horizon must be 1..{MAX_HORIZON_DAYS} days")
    out = {name: m.shift(k - 1) for name, m in F.items()}
    units = daily.count.columns
    for target in TARGETS:
        out[f"{target}_same_dow"] = daily.target(target).shift(7)  # day D - 7 <= origin - 1
    dow = daily.days.dayofweek.to_numpy()
    for name, v in (("dow_sin", np.sin(2 * np.pi * dow / 7)), ("dow_cos", np.cos(2 * np.pi * dow / 7)),
                    ("is_weekend", (dow >= 5).astype(float))):
        out[f"cal_{name}"] = pd.DataFrame(np.repeat(v[:, None], len(units), 1), index=daily.days, columns=units)
    return out


@dataclass(frozen=True)
class DaySplit:
    name: str
    start: pd.Timestamp  # first target day (inclusive)
    end: pd.Timestamp  # last target day (exclusive)

    def train_end(self, k: int) -> pd.Timestamp:
        """Exclusive bound on training target days: D <= start - k (known at the first origin)."""
        return self.start - pd.Timedelta(days=k - 1)

    def mask(self, days: pd.DatetimeIndex) -> np.ndarray:
        return (days >= self.start) & (days < self.end)

    def train_mask(self, days: pd.DatetimeIndex, k: int, window_days: int) -> np.ndarray:
        """Target days a model scored on this split may train on (last `window_days` before train_end)."""
        end = self.train_end(k)
        return (days < end) & (days >= end - pd.Timedelta(days=window_days))


def make_daily_splits(
    day_covered: pd.Series, frozen: dict, val_days: int = 28, n_folds: int = 6, fold_days: int = 7
) -> tuple[DaySplit, DaySplit, list[DaySplit]]:
    """
    Final test = the Kyiv days lying entirely inside the frozen test window (never
    scored by the backtest); validation = the last `val_days` fully covered days
    before the Kyiv day containing test_start (a partial day overlapping the test
    window belongs to no split); folds = `fold_days`-long blocks before validation.
    """
    start = pd.Timestamp(frozen["test_start"]).tz_convert(KYIV_TZ)
    end = pd.Timestamp(frozen["test_end"]).tz_convert(KYIV_TZ)
    first = start.normalize() + (pd.Timedelta(days=1) if start != start.normalize() else pd.Timedelta(0))
    last = end.normalize()  # exclusive: day `last` ends after `end` unless end is midnight
    test = DaySplit("test", first.tz_localize(None), last.tz_localize(None))

    days = day_covered.index
    val_end = start.normalize().tz_localize(None)
    while True:
        window = day_covered[(days >= val_end - pd.Timedelta(days=val_days)) & (days < val_end)]
        holes = window.index[~window.to_numpy()]
        if holes.empty:
            break
        val_end = holes[-1]
        while val_end - pd.Timedelta(days=1) in day_covered.index and not day_covered[val_end - pd.Timedelta(days=1)]:
            val_end -= pd.Timedelta(days=1)
    validation = DaySplit("validation", val_end - pd.Timedelta(days=val_days), val_end)
    folds = [
        DaySplit(f"fold{n_folds - i}", validation.start - pd.Timedelta(days=fold_days * (i + 1)),
                 validation.start - pd.Timedelta(days=fold_days * i))
        for i in range(n_folds)
    ]
    return test, validation, folds[::-1]
