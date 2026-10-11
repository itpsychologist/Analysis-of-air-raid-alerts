"""
Leakage tests (PLAN.md, section 3.2). They must pass before any model result is trusted.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.splits import make_splits
from src.features.panel import Panel, build_panel, labels
from src.models.baselines import BASELINES

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
HORIZONS = [1, 3, 6, 24]


def random_labels(start="2025-09-01", days=70, units=4, seed=0) -> pd.DataFrame:
    """Regular hourly label matrix spanning the autumn DST change (2025-10-26)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(T(start), periods=days * 24, freq="h")
    y = (rng.random((len(idx), units)) < 0.3).astype(float)
    y[rng.random(y.shape) < 0.05] = np.nan  # gaps
    return pd.DataFrame(y, index=idx, columns=[f"u{i}" for i in range(units)])


ORIGINS = [T("2025-10-20 13:00"), T("2025-10-26 01:00"), T("2025-10-26 02:00"), T("2025-11-05 23:00")]


@pytest.mark.parametrize("name", list(BASELINES))
@pytest.mark.parametrize("h", HORIZONS)
def test_baseline_ignores_labels_not_yet_known(name, h):
    """Future-perturbation test: labels of origins t' with t' + h > t must not matter at t."""
    fn = BASELINES[name]
    Y = random_labels()
    rng = np.random.default_rng(1)
    for t in ORIGINS:
        unknown = Y.index + pd.Timedelta(hours=h) > t
        Y_perturbed = Y.copy()
        Y_perturbed[unknown] = rng.choice([0.0, 1.0, np.nan], size=Y_perturbed[unknown].shape)
        before, after = fn(Y, h).loc[t], fn(Y_perturbed, h).loc[t]
        pd.testing.assert_series_equal(before, after, obj=f"{name} h={h} at {t}")


def toy_panel() -> Panel:
    hours = pd.date_range(T("2026-09-01"), periods=48, freq="h")
    starts = pd.DataFrame(0.0, index=hours, columns=["A"])
    starts.loc[T("2026-09-01 05:00"), "A"] = 1
    starts.loc[T("2026-09-01 20:00"), "A"] = 1
    covered = pd.Series(True, index=hours)
    covered[(hours >= T("2026-09-01 10:00")) & (hours < T("2026-09-01 14:00"))] = False  # gap
    return Panel(starts=starts, active_minutes=starts * 0, covered=covered, data_cutoff=hours[-1])


def test_labels_look_forward_only_and_are_nan_across_gaps():
    y = labels(toy_panel(), h=3)["A"]
    assert y[T("2026-09-01 03:00")] == 1.0  # start at 05:00 is inside [03, 06)
    assert y[T("2026-09-01 05:00")] == 1.0
    assert y[T("2026-09-01 06:00")] == 0.0  # the past start does not count
    assert np.isnan(y[T("2026-09-01 08:00")])  # window [08, 11) touches the gap
    assert y[T("2026-09-01 14:00")] == 0.0
    assert np.isnan(y.iloc[-1])  # window runs past the end of data


def test_truncated_panel_gives_identical_labels_where_fully_known():
    """Future-perturbation for the target itself: data after T only affects windows past T."""
    panel = toy_panel()
    cut = T("2026-09-01 21:00")
    full, trunc = labels(panel, 3), labels(panel.truncate(cut), 3)
    known = trunc.index + pd.Timedelta(hours=3) <= cut
    pd.testing.assert_frame_equal(full[full.index < cut][known], trunc[known])


def test_stitching_boundary_does_not_create_fake_alert_starts():
    intervals = pd.DataFrame({
        "unit": ["A", "A"],
        # first one is an alert that was ongoing when the API window began (clipped)
        "started_at": [T("2026-09-12 00:00"), T("2026-09-12 05:30")],
        "finished_at": [T("2026-09-12 02:00"), T("2026-09-12 06:00")],
    })
    coverage = {
        "windows": [
            {"source": "official", "start": "2026-09-01T00:00:00+00:00", "end": "2026-09-07T05:28:00+00:00"},
            {"source": "gap", "start": "2026-09-07T05:28:00+00:00", "end": "2026-09-12T00:00:00+00:00"},
            {"source": "alerts_in_ua", "start": "2026-09-12T00:00:00+00:00", "end": "2026-09-13T00:00:00+00:00"},
        ],
        "data_cutoff": "2026-09-13T00:00:00+00:00",
        "units": ["A", "B"],
    }
    panel = build_panel(intervals, coverage)
    assert panel.starts["A"].sum() == 1  # only the 05:30 start is real
    assert panel.starts.loc[T("2026-09-12 05:00"), "A"] == 1
    assert not panel.covered[T("2026-09-07 05:00")]  # partially covered hour
    assert not panel.covered[T("2026-09-10 00:00")]
    assert list(panel.units) == ["A", "B"]  # unit without alerts is still in the panel


def test_rebuilding_panel_with_future_events_keeps_known_labels():
    """Future perturbation on raw data: alerts after T must not change any label known at T."""
    coverage = {
        "windows": [{"source": "official", "start": "2026-09-01T00:00:00+00:00", "end": "2026-09-04T00:00:00+00:00"}],
        "data_cutoff": "2026-09-04T00:00:00+00:00",
        "units": ["A", "B"],
    }
    base = pd.DataFrame({
        "unit": ["A", "B", "A"],
        "started_at": [T("2026-09-01 03:10"), T("2026-09-01 20:00"), T("2026-09-02 07:45")],
        "finished_at": [T("2026-09-01 04:00"), T("2026-09-02 02:00"), T("2026-09-02 09:00")],
    })
    cut = T("2026-09-02 12:00")
    future = pd.DataFrame({
        "unit": ["A", "B", "B"],
        "started_at": [T("2026-09-02 12:30"), T("2026-09-02 13:00"), T("2026-09-03 01:00")],
        "finished_at": [T("2026-09-02 14:00"), T("2026-09-02 15:00"), T("2026-09-03 02:00")],
    })
    with_future = build_panel(pd.concat([base, future], ignore_index=True), coverage)
    without = build_panel(base, coverage)
    for h in HORIZONS:
        a, b = labels(with_future, h), labels(without, h)
        known = a.index + pd.Timedelta(hours=h) <= cut
        pd.testing.assert_frame_equal(a[known], b[known], obj=f"labels h={h}")


@pytest.mark.parametrize("h", HORIZONS)
@pytest.mark.parametrize("frozen", [None, {"test_start": "2026-09-13T10:00:00+00:00", "test_end": "2026-10-11T00:00:00+00:00"}])
def test_splits_are_ordered_embargoed_and_validation_avoids_gaps(h, frozen):
    hours = pd.date_range(T("2026-03-01"), T("2026-10-11"), freq="h", inclusive="left")
    covered = pd.Series(True, index=hours)
    covered[(hours >= T("2026-09-06 05:00")) & (hours < T("2026-09-12"))] = False
    test, val, folds = make_splits(covered, h, frozen=frozen)

    seq = folds + [val, test]
    for a, b in zip(seq, seq[1:]):
        # PLAN 3.2: the last label window of a ends at least h hours before b starts,
        # so a model fit on all of `a` keeps the full embargo when scored on `b`
        last_origin_a = a.end - pd.Timedelta(hours=1)
        assert last_origin_a + pd.Timedelta(hours=h) <= b.start - pd.Timedelta(hours=h)
        assert a.end <= b.train_end(h)
        assert not (a.mask(hours) & b.mask(hours)).any()

    # every label window of a validation origin lies in covered hours
    last_val_window_end = val.end - pd.Timedelta(hours=1) + pd.Timedelta(hours=h)
    assert covered[(hours >= val.start) & (hours < last_val_window_end)].all()
    if frozen:
        assert test.start == T(frozen["test_start"][:16])
    else:
        assert test.end == hours[-1] + pd.Timedelta(hours=1) - pd.Timedelta(hours=h - 1)


def test_frozen_test_window_does_not_move_with_new_data():
    frozen = {"test_start": "2026-09-13T10:00:00+00:00", "test_end": "2026-10-11T10:00:00+00:00"}
    short = pd.Series(True, index=pd.date_range(T("2026-05-01"), T("2026-10-11 10:00"), freq="h", inclusive="left"))
    longer = pd.Series(True, index=pd.date_range(T("2026-05-01"), T("2026-11-20"), freq="h", inclusive="left"))
    assert make_splits(short, 3, frozen=frozen)[0] == make_splits(longer, 3, frozen=frozen)[0]
    too_short = short[short.index < T("2026-10-01")]
    with pytest.raises(ValueError, match="frozen test window"):
        make_splits(too_short, 3, frozen=frozen)
