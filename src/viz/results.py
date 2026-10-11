"""
results.py — Read the committed evaluation results (reports/results/) for the HTML
report and the dashboard's Backtest tab, and build the small data-overview tables.

Nothing here fits or scores a model: the report shows exactly what
scripts/run_final_test.py computed on the final test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "results"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
KYIV_TZ = "Europe/Kyiv"


@dataclass
class Results:
    marker: dict  # final_test.json: when and on what the test was scored
    scores: pd.DataFrame  # target A on the test
    reliability: pd.DataFrame
    errors: pd.DataFrame
    bootstrap: pd.DataFrame
    daily: pd.DataFrame  # target B on the test
    intervals: pd.DataFrame
    hourly_series: pd.DataFrame
    daily_series: pd.DataFrame
    backtest: pd.DataFrame  # folds + validation (stages 2-4), for comparison
    backtest_daily: pd.DataFrame
    overview: dict[str, pd.DataFrame | list]
    config: dict  # model_config.json (selected settings), {} if absent


def load_results(results_dir: Path = RESULTS_DIR) -> Results | None:
    """The final-test results, or None if the test has not been scored yet."""
    d = Path(results_dir)
    marker = d / "final_test.json"
    if not marker.exists():
        return None
    csv = lambda name: pd.read_csv(d / f"{name}.csv")  # noqa: E731
    hourly = pd.read_parquet(d / "final_test_hourly.parquet")
    daily_series = pd.read_parquet(d / "final_test_daily.parquet")
    return Results(
        marker=json.loads(marker.read_text(encoding="utf-8")),
        scores=csv("final_test"),
        reliability=csv("final_test_reliability"),
        errors=csv("final_test_errors"),
        bootstrap=csv("final_test_bootstrap"),
        daily=csv("final_test_daily"),
        intervals=csv("final_test_intervals_daily"),
        hourly_series=hourly,
        daily_series=daily_series,
        backtest=csv("backtest"),
        backtest_daily=csv("backtest_daily"),
        overview=load_overview(d),
        config=json.loads((d / "model_config.json").read_text(encoding="utf-8"))
        if (d / "model_config.json").exists() else {},
    )


# ---------------------------------------------------------------------------
# Data overview (descriptive only; computed from data/processed when available)
# ---------------------------------------------------------------------------

def compute_overview(intervals: pd.DataFrame, coverage: dict, recent_days: int = 365) -> dict:
    """
    Tables for the report's data section:
      daily    Kyiv day -> mean hours under alert per raion, alert starts (NaN in gaps)
      heatmap  alert starts by Kyiv weekday x hour over the last `recent_days`
      oblast   share of hours under alert per oblast (mean over its raions), same window
      splits   the evaluation layout for h = 1 (train / folds / validation / gap / test)
    """
    from src.evaluation.splits import load_frozen_test, make_splits
    from src.features.panel import build_panel

    panel = build_panel(intervals, coverage)
    local = panel.hours.tz_convert(KYIV_TZ)
    day = local.normalize().tz_localize(None)
    covered = panel.covered.to_numpy()
    hours_active = panel.active_minutes.to_numpy().sum(axis=1) / 60 / panel.active_minutes.shape[1]
    starts = panel.starts.to_numpy().sum(axis=1)
    by_day = pd.DataFrame({"day": day, "h": hours_active, "s": starts, "cov": covered}).groupby("day")
    daily = by_day.agg(hours_per_raion=("h", "sum"), alert_starts=("s", "sum"), covered=("cov", "all"))
    daily.loc[~daily["covered"], ["hours_per_raion", "alert_starts"]] = np.nan
    daily = daily.drop(columns="covered").reset_index()

    recent = (panel.hours >= panel.hours[-1] - pd.Timedelta(days=recent_days)) & covered
    heat = pd.DataFrame({"weekday": local.dayofweek[recent], "hour": local.hour[recent], "s": starts[recent]})
    heatmap = heat.pivot_table(index="weekday", columns="hour", values="s", aggfunc="sum").reindex(range(7))
    heatmap.index = WEEKDAYS

    share = panel.active_minutes[recent].mean() / 60
    unit_oblast = coverage["unit_oblast"]
    oblast = share.groupby(share.index.map(unit_oblast)).mean().rename("share").rename_axis("oblast").reset_index()

    test, validation, folds = make_splits(panel.covered, 1, frozen=load_frozen_test())
    gap = next(w for w in coverage["windows"] if w.get("reason") == "between sources")
    windows = [
        {"label": "training (window ≤ 180 d)", "start": validation.start - pd.Timedelta(days=180),
         "end": folds[0].start, "kind": "train", "row": "model selection"},
        {"label": "6 weekly folds", "start": folds[0].start, "end": folds[-1].end, "kind": "fold",
         "row": "model selection"},
        {"label": "validation", "start": validation.start, "end": validation.end, "kind": "validation",
         "row": "model selection"},
        {"label": "training (incl. validation)", "start": test.start - pd.Timedelta(days=180),
         "end": validation.end, "kind": "train", "row": "final test"},
        {"label": "no data", "start": gap["start"], "end": gap["end"], "kind": "gap", "row": "final test"},
        {"label": "final test", "start": test.start, "end": test.end, "kind": "test", "row": "final test"},
    ]
    splits = [{**w, "start": str(pd.Timestamp(w["start"])), "end": str(pd.Timestamp(w["end"]))} for w in windows]
    return {"daily": daily, "heatmap": heatmap, "oblast": oblast, "splits": splits}


def save_overview(overview: dict, results_dir: Path = RESULTS_DIR) -> None:
    d = Path(results_dir)
    overview["daily"].to_csv(d / "overview_daily.csv", index=False, float_format="%.3f")
    overview["heatmap"].to_csv(d / "overview_heatmap.csv", float_format="%.0f")
    overview["oblast"].to_csv(d / "overview_oblast.csv", index=False, float_format="%.4f")
    (d / "overview_splits.json").write_text(json.dumps(overview["splits"], indent=2), encoding="utf-8")


def load_overview(results_dir: Path = RESULTS_DIR) -> dict:
    d = Path(results_dir)
    if not (d / "overview_daily.csv").exists():
        return {}
    heatmap = pd.read_csv(d / "overview_heatmap.csv", index_col=0)
    heatmap.columns = heatmap.columns.astype(int)
    return {
        "daily": pd.read_csv(d / "overview_daily.csv", parse_dates=["day"]),
        "heatmap": heatmap,
        "oblast": pd.read_csv(d / "overview_oblast.csv"),
        "splits": json.loads((d / "overview_splits.json").read_text(encoding="utf-8")),
    }
