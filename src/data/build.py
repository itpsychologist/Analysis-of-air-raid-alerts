"""
build.py — Combine source snapshots into the processed raion-level dataset.
============================================================================
Sources are stitched by time, never mixed within a period (they differ in
coverage and semantics, see PLAN.md, stage 1 findings):

  official dataset   [first record,       official_end)
  (gap — no data)    [official_end,       api_start)
  alerts.in.ua API   [api_start,          api_fetched_at)

Outputs in data/processed/:
  raion_intervals.parquet  disjoint alert intervals per unit (raion or м. Київ)
  coverage.json            source windows, gap, snapshot provenance
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.fetch_api import latest_api_snapshot
from src.data.load_dataset import latest_snapshot, snapshot_info
from src.data.merge import (
    clip_intervals,
    expand_to_raions,
    normalize_api_snapshot,
    normalize_official,
    raions_by_oblast,
    union_intervals,
)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# alerts.in.ua month_ago covers ~30 days before the download, with the exact start
# varying per oblast; coverage starts at the first full UTC day after that
API_HISTORY_DAYS = 30


def api_coverage_start(fetched_at: pd.Timestamp) -> pd.Timestamp:
    return (fetched_at - pd.Timedelta(days=API_HISTORY_DAYS)).ceil("D")


def stitch(
    official: pd.DataFrame, api: pd.DataFrame, api_fetched_at: pd.Timestamp
) -> tuple[pd.DataFrame, list[dict], dict[str, str]]:
    """
    Turn unified official + API events into disjoint per-unit intervals, each source
    clipped to its own time window.

    Returns (intervals, windows, unit_oblast).
    """
    official_start = official["started_at"].min()
    # Records stop arriving at the last start; later-ending alerts are still complete
    official_end = official["started_at"].max()
    api_start = max(api_coverage_start(api_fetched_at), official_end)

    raion_map = raions_by_oblast(official, api)
    parts = []
    for events, source, start, end in [
        (official, "official", official_start, official_end),
        (api, "alerts_in_ua", api_start, api_fetched_at),
    ]:
        units = expand_to_raions(events, raion_map)
        # Alerts still open at download time are closed at the cutoff, never later
        iv = union_intervals(units[["unit", "started_at", "finished_at"]], "unit", open_until=end)
        iv = clip_intervals(iv, start, end)
        iv["source"] = source
        parts.append(iv)

    unit_oblast = {r: ob for ob, rs in raion_map.items() for r in rs} | {"м. Київ": "м. Київ"}
    intervals = pd.concat(parts, ignore_index=True)
    intervals["oblast"] = intervals["unit"].map(unit_oblast)
    intervals = intervals[["unit", "oblast", "started_at", "finished_at", "n_events", "source"]]
    intervals = intervals.sort_values(["unit", "started_at"]).reset_index(drop=True)

    windows = [
        {"source": "official", "start": official_start.isoformat(), "end": official_end.isoformat()},
        {"source": "gap", "start": official_end.isoformat(), "end": api_start.isoformat()},
        {"source": "alerts_in_ua", "start": api_start.isoformat(), "end": api_fetched_at.isoformat()},
    ]
    return intervals, windows, unit_oblast


def build() -> tuple[pd.DataFrame, dict]:
    """Stitch the latest raw snapshots; returns (intervals, coverage manifest)."""
    official_path = latest_snapshot("official_data_uk")
    api_dir = latest_api_snapshot()
    api_fetched_at = pd.Timestamp(json.loads((api_dir / "manifest.json").read_text())["fetched_at"])
    intervals, windows, unit_oblast = stitch(
        normalize_official(official_path), normalize_api_snapshot(api_dir), api_fetched_at
    )
    windows[0] |= {"snapshot": official_path.name, **snapshot_info(official_path)}
    windows[2] |= {"snapshot": api_dir.name}
    coverage = {
        "windows": windows,
        "data_cutoff": api_fetched_at.isoformat(),
        # Panels must use this list: a unit with no alerts has no intervals at all
        "units": sorted(unit_oblast),
        "unit_oblast": unit_oblast,
        "levels_used": ["oblast (broadcast to its raions)", "raion"],
    }
    return intervals, coverage


def save(intervals: pd.DataFrame, coverage: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    intervals.to_parquet(PROCESSED_DIR / "raion_intervals.parquet", index=False)
    (PROCESSED_DIR / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load() -> tuple[pd.DataFrame, dict]:
    intervals = pd.read_parquet(PROCESSED_DIR / "raion_intervals.parquet")
    coverage = json.loads((PROCESSED_DIR / "coverage.json").read_text(encoding="utf-8"))
    return intervals, coverage
