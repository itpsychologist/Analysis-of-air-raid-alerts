"""
splits.py — Chronological evaluation splits with an embargo.
============================================================
Layout (origins are hourly forecast times t; the label of t covers [t, t+h)):

  ... | fold 1 | fold 2 | ... | fold k | validation | (gap) | final test |
                                                             ^ last 28 days of data

- Final test: frozen in config/evaluation.json (origins [test_start, test_end - h + 1)),
  so new data snapshots never move it. It is evaluated once, at the end of the
  project (PLAN.md, section 4.1); without the config it is the last `test_days` days.
- Validation: the `val_days` days of covered data before the final test, used for
  model selection and calibration.
- Rolling-origin folds: `fold_days`-long test blocks walking back from validation.

Training data for a split: origins t with t + h <= split start - embargo (default
embargo = h). Consecutive splits are placed the same way: each one ends at the next
one's `train_end`, so a model fit on a whole split keeps the full embargo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FROZEN_CONFIG = Path(__file__).resolve().parents[2] / "config" / "evaluation.json"


@dataclass(frozen=True)
class Split:
    name: str
    start: pd.Timestamp  # first origin (inclusive)
    end: pd.Timestamp  # last origin (exclusive)

    def train_end(self, h: int, embargo_hours: int | None = None) -> pd.Timestamp:
        """
        Exclusive upper bound for training origins: t + h <= start - embargo, so
        no training label window comes within `embargo` hours of the split.
        """
        embargo = h if embargo_hours is None else embargo_hours
        return self.start - pd.Timedelta(hours=h + embargo - 1)

    def mask(self, index: pd.DatetimeIndex):
        return (index >= self.start) & (index < self.end)


def load_frozen_test() -> dict | None:
    """The committed final-test window and the snapshots it was defined on."""
    if not FROZEN_CONFIG.exists():
        return None
    return json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))


def make_splits(
    covered: pd.Series,
    h: int,
    test_days: int = 28,
    val_days: int = 28,
    n_folds: int = 6,
    fold_days: int = 7,
    frozen: dict | None = None,
) -> tuple[Split, Split, list[Split]]:
    """
    Args:
        covered: panel.covered (hourly bool), used to place validation before any gap.
        h:       forecast horizon in hours.
        frozen:  {"test_start", "test_end"} (ISO timestamps); see load_frozen_test().

    Returns (final_test, validation, folds) with folds ordered oldest first.
    """
    hours = covered.index
    if frozen:
        start, end = pd.Timestamp(frozen["test_start"]), pd.Timestamp(frozen["test_end"])
        if hours[-1] + pd.Timedelta(hours=1) < end:
            raise ValueError(f"panel ends before the frozen test window ({end})")
        test = Split("test", start, end - pd.Timedelta(hours=h - 1))
    else:
        last_origin_end = hours[-1] + pd.Timedelta(hours=1) - pd.Timedelta(hours=h - 1)
        test = Split("test", last_origin_end - pd.Timedelta(days=test_days), last_origin_end)

    # Validation ends at the test's train_end and must contain no uncovered
    # hours: if it would, move it to end where the latest uncovered stretch begins.
    val_end = test.train_end(h)
    while True:
        window = covered[(hours >= val_end - pd.Timedelta(days=val_days)) & (hours < val_end)]
        holes = window.index[~window.to_numpy()]
        if holes.empty:
            break
        run_start = holes[-1]
        while run_start - pd.Timedelta(hours=1) in covered.index and not covered[run_start - pd.Timedelta(hours=1)]:
            run_start -= pd.Timedelta(hours=1)
        val_end = run_start - pd.Timedelta(hours=h - 1)  # last label window ends at the hole
    validation = Split("validation", val_end - pd.Timedelta(days=val_days), val_end)

    folds = []
    nxt = validation
    for i in range(n_folds):
        end = nxt.train_end(h)
        fold = Split(f"fold{n_folds - i}", end - pd.Timedelta(days=fold_days), end)
        folds.append(fold)
        nxt = fold
    return test, validation, folds[::-1]
