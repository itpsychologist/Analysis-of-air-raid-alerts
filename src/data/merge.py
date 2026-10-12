"""
merge.py — Bring all alert sources to one schema and to per-unit alert intervals.
==================================================================================
Unified event schema (one row per raw alert):

    oblast | raion | hromada | level | alert_type | started_at | finished_at | end_imputed | source

  - oblast:      canonical Ukrainian name, as in etl.OBLAST_UID_MAP keys
  - level:       "oblast" | "raion" | "hromada"
  - started_at / finished_at: tz-aware UTC Timestamps
  - end_imputed: True when the end time is not observed (volunteer `naive` rows,
                 where the dataset sets finished_at = started_at + 30 min)
  - source:      "official" | "volunteer" | "alerts_in_ua"

The forecast unit is the raion: `expand_to_raions` copies oblast-level alerts (the norm
before Dec 2025) onto every raion of the oblast, and `union_intervals` merges overlapping
alerts per unit, so both alerting regimes map onto one target definition.
`to_oblast_intervals` is the same union at oblast level, used to compare sources.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from etl import OBLAST_UID_MAP

EVENT_COLUMNS = [
    "oblast", "raion", "hromada", "level", "alert_type",
    "started_at", "finished_at", "end_imputed", "source",
]

# Volunteer data spells Kyiv City differently from the API / official data
OBLAST_ALIASES = {"Київ": "м. Київ"}

# Raions renamed in 2024: official dataset keeps old names, alerts.in.ua uses new ones
RAION_RENAMES = {
    "Володимир-Волинський район": "Володимирський район",
    "Красноградський район": "Берестинський район",
    "Червоноградський район": "Шептицький район",
}

# Permanent alert since 2022-04-04 (not listed in the datasets) — carries no signal
EXCLUDED_OBLASTS = {"Луганська область"}


def _finalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    df["oblast"] = df["oblast"].replace(OBLAST_ALIASES)
    if "raion" in df:
        # Sources disagree on the apostrophe (’ vs ')
        df["raion"] = df["raion"].astype("string").str.replace("’", "'", regex=False).replace(RAION_RENAMES)
    unknown = set(df["oblast"]) - set(OBLAST_UID_MAP)
    if unknown:
        raise ValueError(f"{source}: unknown oblast names {sorted(unknown)}")
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    df["source"] = source
    for col in EVENT_COLUMNS:
        if col not in df:
            df[col] = pd.NA
    df = df[EVENT_COLUMNS].dropna(subset=["started_at"])
    return df.sort_values(["started_at", "oblast"], kind="stable").reset_index(drop=True)


def normalize_official(path: Path) -> pd.DataFrame:
    """
    Official dataset. Drops exact duplicate rows: the 2026-10-11 snapshot contains
    the whole 2022-03..2026-01 history appended twice (113,845 duplicate rows).
    """
    df = pd.read_csv(path).drop_duplicates()
    df["alert_type"] = "air_raid"  # dataset does not distinguish types
    df["end_imputed"] = False
    return _finalize(df, "official")


def normalize_volunteer(path: Path) -> pd.DataFrame:
    """Volunteer (eTryvoga) dataset: oblast level only, some end times imputed."""
    df = pd.read_csv(path).rename(columns={"region": "oblast", "naive": "end_imputed"})
    df["level"] = "oblast"
    df["alert_type"] = "air_raid"
    return _finalize(df, "volunteer")


def normalize_api_snapshot(snapshot_dir: Path) -> pd.DataFrame:
    """
    alerts.in.ua month_ago history from a raw snapshot directory (see fetch_api.py).
    Alerts still active at download time keep finished_at = NaT.
    """
    records = []
    for path in sorted(snapshot_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    raw = pd.DataFrame(records).drop_duplicates(subset=["id"])
    level_map = {"oblast": "oblast", "raion": "raion", "hromada": "hromada", "city": "hromada"}
    # The raion/hromada name is in location_title; there is no location_raion field
    df = pd.DataFrame({
        "oblast": raw["location_oblast"],
        "raion": raw["location_title"].where(raw["location_type"] == "raion"),
        "hromada": raw["location_title"].where(raw["location_type"].isin(["hromada", "city"])),
        "level": raw["location_type"].map(level_map).fillna(raw["location_type"]),
        "alert_type": raw["alert_type"],
        "started_at": raw["started_at"],
        "finished_at": raw["finished_at"],
        "end_imputed": False,
    })
    return _finalize(df, "alerts_in_ua")


def to_oblast_intervals(
    events: pd.DataFrame,
    levels: tuple[str, ...] = ("oblast", "raion"),
    open_until: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Union of overlapping alert intervals per oblast.

    Args:
        events:     unified events.
        levels:     which alert levels count as "the oblast is under alert".
        open_until: end time to use for alerts still open (finished_at NaT). This must
                    be the data cutoff, never a later time, or a feature/target could
                    see the future.

    Returns:
        DataFrame [oblast, started_at, finished_at, n_events] of disjoint intervals.
    """
    df = events[events["level"].isin(levels) & ~events["oblast"].isin(EXCLUDED_OBLASTS)]
    return union_intervals(df[["oblast", "started_at", "finished_at"]], "oblast", open_until)


def union_intervals(
    df: pd.DataFrame, key: str, open_until: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Merge overlapping [started_at, finished_at] intervals within each `key` group."""
    df = df.copy()
    if open_until is not None:
        df["finished_at"] = df["finished_at"].fillna(open_until)
    df = df.dropna(subset=["finished_at"])
    df = df[df["finished_at"] >= df["started_at"]]
    df = df.sort_values([key, "started_at"])

    # A new interval starts where started_at exceeds the running max end so far
    running_end = df.groupby(key)["finished_at"].cummax()
    prev_end = running_end.groupby(df[key]).shift()
    group = (df["started_at"] > prev_end) | prev_end.isna()
    df["grp"] = group.groupby(df[key]).cumsum()
    merged = df.groupby([key, "grp"]).agg(
        started_at=("started_at", "min"),
        finished_at=("finished_at", "max"),
        n_events=("started_at", "size"),
    )
    return merged.reset_index().drop(columns="grp")


def raions_by_oblast(*event_frames: pd.DataFrame) -> dict[str, list[str]]:
    """Raion list per oblast, as seen in raion-level alerts of the given sources."""
    df = pd.concat([e[e["level"] == "raion"][["oblast", "raion"]] for e in event_frames])
    df = df.dropna().drop_duplicates()
    return {ob: sorted(g["raion"]) for ob, g in df.groupby("oblast")}


def expand_to_raions(events: pd.DataFrame, raion_map: dict[str, list[str]]) -> pd.DataFrame:
    """
    Map oblast- and raion-level events onto raion units.

    An oblast-level alert (the norm before Dec 2025) put every raion of the oblast
    under alert, so it is copied to each raion; a raion-level alert stays with its
    raion. Kyiv City has no raions in the data and is kept as a single unit.
    Hromada-level alerts are ignored.

    Returns events with columns [unit, oblast, started_at, finished_at].
    """
    ev = events[~events["oblast"].isin(EXCLUDED_OBLASTS)]
    raion_ev = ev[ev["level"] == "raion"].assign(unit=lambda d: d["raion"])
    oblast_ev = ev[ev["level"] == "oblast"]
    units = pd.DataFrame(
        [(ob, r) for ob, rs in raion_map.items() for r in rs], columns=["oblast", "unit"]
    )
    broadcast = oblast_ev.merge(units, on="oblast", how="left")
    broadcast["unit"] = broadcast["unit"].fillna(broadcast["oblast"])  # e.g. м. Київ
    cols = ["unit", "oblast", "started_at", "finished_at"]
    parts = [p[cols] for p in (raion_ev, broadcast) if not p.empty]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    return out.dropna(subset=["unit"])


def clip_intervals(intervals: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Keep the parts of intervals that fall inside [start, end)."""
    df = intervals[(intervals["finished_at"] > start) & (intervals["started_at"] < end)].copy()
    df["started_at"] = df["started_at"].clip(lower=start)
    df["finished_at"] = df["finished_at"].clip(upper=end)
    return df


def hourly_activity(
    intervals: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, key: str = "oblast"
) -> pd.DataFrame:
    """
    Minutes under alert per (unit, UTC hour) in [start, end).

    Intervals must be disjoint within a unit (see union_intervals), otherwise
    overlapping minutes are counted twice.

    Returns a wide DataFrame: index = hourly UTC timestamps, columns = values of `key`.
    """
    hours = pd.date_range(start.floor("h"), end.ceil("h"), freq="h", tz="UTC", inclusive="left")
    units = sorted(intervals[key].unique())
    out = np.zeros((len(hours), len(units)))
    iv = clip_intervals(intervals, start, end)
    if not iv.empty:
        # Minutes since the first hour; each interval is split into the hours it touches
        s = ((iv["started_at"] - hours[0]).dt.total_seconds() / 60).to_numpy()
        e = ((iv["finished_at"] - hours[0]).dt.total_seconds() / 60).to_numpy()
        first = np.floor(s / 60).astype(int)
        n_hours = np.maximum(np.ceil(e / 60).astype(int) - first, 0)
        row = np.repeat(np.arange(len(iv)), n_hours)
        hour = first[row] + np.arange(n_hours.sum()) - np.repeat(np.cumsum(n_hours) - n_hours, n_hours)
        minutes = np.minimum(e[row], (hour + 1) * 60) - np.maximum(s[row], hour * 60)
        col = pd.Index(units).get_indexer(iv[key])[row]
        np.add.at(out, (hour, col), minutes)
    return pd.DataFrame(out, index=hours, columns=units)
