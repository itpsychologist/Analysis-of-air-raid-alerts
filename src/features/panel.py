"""
panel.py — Hourly unit panel and forecast targets.
==================================================
Wide matrices (index = UTC hours, columns = units) built from the processed
raion intervals (see src/data/build.py):

  starts          number of alert *starts* (calm -> alert transitions) in the hour
  active_minutes  minutes under alert in the hour
  covered         whether the whole hour lies inside one source window

Target A for a forecast made at hour boundary t with horizon h:

  y_h(t) = 1  if a new alert starts in the unit during [t, t+h)
           0  otherwise
           NaN if any hour of [t, t+h) is not covered (gap / source boundary)

The label for origin t is only *known* at t + h. Anything that uses labels as
inputs at time t may only use origins t' with t' + h <= t (see models/baselines.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.merge import hourly_activity

@dataclass
class Panel:
    starts: pd.DataFrame
    active_minutes: pd.DataFrame
    covered: pd.Series
    data_cutoff: pd.Timestamp

    @property
    def hours(self) -> pd.DatetimeIndex:
        return self.starts.index

    @property
    def units(self) -> list[str]:
        return list(self.starts.columns)

    def truncate(self, end: pd.Timestamp) -> "Panel":
        """The panel as it would have looked with data only up to `end` (exclusive)."""
        keep = self.hours < end
        return Panel(
            starts=self.starts[keep],
            active_minutes=self.active_minutes[keep],
            covered=self.covered[keep] & (self.hours[keep] + pd.Timedelta(hours=1) <= end),
            data_cutoff=min(self.data_cutoff, end),
        )


def _covered_hours(hours: pd.DatetimeIndex, windows: list[dict]) -> pd.Series:
    covered = pd.Series(False, index=hours)
    for w in windows:
        if w["source"] == "gap":
            continue
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        covered |= (hours >= start.ceil("h")) & (hours + pd.Timedelta(hours=1) <= end)
    return covered


def build_panel(
    intervals: pd.DataFrame, coverage: dict, start: pd.Timestamp | None = None
) -> Panel:
    """
    Args:
        intervals: processed disjoint per-unit intervals (raion_intervals.parquet).
        coverage:  coverage manifest (windows, units, data_cutoff).
        start:     first hour of the panel; defaults to the first covered hour.
    """
    windows = coverage["windows"]
    cutoff = pd.Timestamp(coverage["data_cutoff"])
    start = (start or pd.Timestamp(windows[0]["start"])).ceil("h")
    end = cutoff.floor("h")
    units = coverage["units"]

    active = hourly_activity(intervals, start, end, key="unit").reindex(columns=units, fill_value=0.0)

    # An interval that begins exactly at a source-window start was clipped there;
    # its start is an artefact of stitching, not a real calm -> alert transition.
    window_starts = {pd.Timestamp(w["start"]) for w in windows if w["source"] != "gap"}
    real = intervals[~intervals["started_at"].isin(window_starts)]
    real = real[(real["started_at"] >= start) & (real["started_at"] < end)]
    counts = (
        real.assign(hour=real["started_at"].dt.floor("h"))
        .groupby(["hour", "unit"]).size().unstack(fill_value=0)
    )
    starts = counts.reindex(index=active.index, columns=units, fill_value=0).astype(float)

    return Panel(
        starts=starts,
        active_minutes=active,
        covered=_covered_hours(active.index, windows),
        data_cutoff=cutoff,
    )


def labels(panel: Panel, h: int) -> pd.DataFrame:
    """
    Target A: 1.0 / 0.0 / NaN per (origin hour, unit), see module docstring.
    Origins whose window [t, t+h) runs past the panel end are NaN.
    """
    starts = panel.starts.to_numpy()
    covered = panel.covered.to_numpy().astype(float)
    n = len(starts)
    csum = np.vstack([np.zeros((1, starts.shape[1])), np.cumsum(starts, axis=0)])
    ccov = np.concatenate([[0.0], np.cumsum(covered)])

    y = np.full(starts.shape, np.nan)
    t = np.arange(n - h + 1)
    window_starts = csum[t + h] - csum[t]
    fully_covered = (ccov[t + h] - ccov[t]) == h
    y[t] = np.where(fully_covered[:, None], (window_starts > 0).astype(float), np.nan)
    return pd.DataFrame(y, index=panel.hours, columns=panel.units)
