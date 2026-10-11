"""
build.py — Combine source snapshots into the processed raion-level dataset.
============================================================================
Sources are stitched by time, never mixed within a period (they differ in
coverage and semantics, see PLAN.md, stage 1 findings):

  official dataset   [first record,       official_end)   minus source outages
  (gap — no data)    [official_end,       api_start)
  alerts.in.ua API   [api_start,          api_fetched_at)

A source outage is a stretch of more than OUTAGE_MIN_GAP without a single alert start
anywhere in Ukraine: the feed stopped, it was not quiet (2026-08-29 .. 09-02, 105 h; the
longest real national lull in 2022-2026 is 30 h). It becomes a gap window, so it is
neither training data nor "no alert" labels.

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

# The official feed writes an alert only once it has finished (no open rows; nothing
# ends more than minutes after the last start). Alerts still running when the feed
# stopped are therefore missing, so its last day is treated as not covered: only
# alerts longer than this (<1% of raion alerts in 2026) can still be missing.
OFFICIAL_TAIL_TRIM = pd.Timedelta(hours=24)


OUTAGE_MIN_GAP = pd.Timedelta(hours=48)


def find_outages(starts: pd.Series, min_gap: pd.Timedelta = OUTAGE_MIN_GAP) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(last start before, first start after) for every national gap between alert starts > min_gap."""
    t = starts.drop_duplicates().sort_values().reset_index(drop=True)
    gap = t.diff()
    return [(t[i - 1], t[i]) for i in gap.index[gap > min_gap]]


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
    official_end = official["started_at"].max() - OFFICIAL_TAIL_TRIM
    api_start = max(api_coverage_start(api_fetched_at), official_end)
    outages = find_outages(official.loc[official["started_at"] < official_end, "started_at"])
    edges = [official_start] + [x for o in outages for x in o] + [official_end]
    official_pieces = list(zip(edges[::2], edges[1::2]))

    raion_map = raions_by_oblast(official, api)
    parts = []
    for events, source, pieces, end in [
        (official, "official", official_pieces, official_end),
        (api, "alerts_in_ua", [(api_start, api_fetched_at)], api_fetched_at),
    ]:
        units = expand_to_raions(events, raion_map)
        # Alerts still open at download time are closed at the cutoff, never later
        iv = union_intervals(units[["unit", "started_at", "finished_at"]], "unit", open_until=end)
        for start, stop in pieces:
            piece = clip_intervals(iv, start, stop)
            piece["source"] = source
            parts.append(piece)

    unit_oblast = {r: ob for ob, rs in raion_map.items() for r in rs} | {"м. Київ": "м. Київ"}
    intervals = pd.concat(parts, ignore_index=True)
    intervals["oblast"] = intervals["unit"].map(unit_oblast)
    intervals = intervals[["unit", "oblast", "started_at", "finished_at", "n_events", "source"]]
    intervals = intervals.sort_values(["unit", "started_at"]).reset_index(drop=True)

    windows = []
    for i, (start, stop) in enumerate(official_pieces):
        windows.append({"source": "official", "start": start.isoformat(), "end": stop.isoformat()})
        if i < len(outages):
            windows.append({"source": "gap", "reason": "source outage", "start": stop.isoformat(),
                            "end": outages[i][1].isoformat()})
    windows += [
        {"source": "gap", "reason": "between sources", "start": official_end.isoformat(), "end": api_start.isoformat()},
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
    for w in windows:
        if w["source"] == "official":
            w |= {"snapshot": official_path.name, **snapshot_info(official_path)}
        elif w["source"] == "alerts_in_ua":
            w |= {"snapshot": api_dir.name}
    coverage = {
        "windows": windows,
        "data_cutoff": api_fetched_at.isoformat(),
        # Panels must use this list: a unit with no alerts has no intervals at all
        "units": sorted(unit_oblast),
        "unit_oblast": unit_oblast,
        "levels_used": ["oblast (broadcast to its raions)", "raion"],
    }
    return intervals, coverage


def windows_fingerprint(coverage: dict) -> str:
    """sha256 of the coverage windows: ties a model selection to the dataset build it ran on."""
    import hashlib

    return hashlib.sha256(json.dumps(coverage["windows"], sort_keys=True).encode()).hexdigest()


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
