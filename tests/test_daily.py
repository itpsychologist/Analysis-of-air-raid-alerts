"""
Target B (daily count / duration): Kyiv-day aggregation, leakage of daily features and
baselines, daily splits, regression metrics.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import interval_scores, pinball, poisson_deviance, regression_scores
from src.features.daily import DaySplit, build_daily, for_horizon, make_daily_splits, origin_features
from src.features.panel import build_panel
from src.models.baselines import DAILY_BASELINES, holt_winters_daily
from tests.test_features import COVERAGE, UNIT_OBLAST, perturb_future, random_intervals

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
D = pd.Timestamp


def test_days_follow_kyiv_time_and_split_alerts_at_midnight():
    iv = pd.DataFrame({
        "unit": ["A1", "A1"],
        # 2025-10-02 23:30 -> 2025-10-03 01:00 Kyiv (UTC+3): 0.5 h on Oct 2, 1 h on Oct 3
        "started_at": [T("2025-10-02 20:30"), T("2025-10-26 10:00")],
        "finished_at": [T("2025-10-02 22:00"), T("2025-10-26 11:00")],
    })
    daily = build_daily(build_panel(iv, COVERAGE))
    a = daily.count["A1"]
    assert a[D("2025-10-02")] == 1 and a[D("2025-10-03")] == 0
    assert daily.duration["A1"][D("2025-10-02")] == pytest.approx(0.5)
    assert daily.duration["A1"][D("2025-10-03")] == pytest.approx(1.0)
    # the DST day has 25 hours and is still a covered day
    assert a[D("2025-10-26")] == 1
    assert daily.origin_hour[D("2025-10-26")] == T("2025-10-25 21:00")
    assert daily.origin_hour[D("2025-10-27")] == T("2025-10-26 22:00")
    # days touching the gap or the window end are not covered
    for day in ("2025-10-20", "2025-10-21", "2025-10-22"):
        assert np.isnan(a[D(day)])
    assert not np.isnan(a[D("2025-10-23")])


ORIGIN_DAYS = [D("2025-10-15"), D("2025-10-23"), D("2025-10-26"), D("2025-10-27"), D("2025-11-02")]


def kyiv_midnight(day: pd.Timestamp) -> pd.Timestamp:
    return day.tz_localize("Europe/Kyiv").tz_convert("UTC")


@pytest.mark.parametrize("k", [1, 3, 7])
def test_daily_features_and_baselines_ignore_everything_after_the_origin(k):
    """Future-perturbation test (PLAN 3.2): nothing after the origin may change the design row."""
    iv = random_intervals()

    def design(intervals):
        panel = build_panel(intervals, COVERAGE)
        daily = build_daily(panel)
        Fk = for_horizon(origin_features(daily, panel, UNIT_OBLAST), daily, k)
        for name, fn in DAILY_BASELINES.items():
            Fk[f"baseline_{name}"] = fn(daily.count, k)
        return Fk

    F = design(iv)
    checked = set()
    for i, origin in enumerate(ORIGIN_DAYS):
        target = origin + pd.Timedelta(days=k - 1)
        if target not in F["count_1d"].index:
            continue
        F2 = design(perturb_future(iv, kyiv_midnight(origin), seed=i))
        for name in F:
            pd.testing.assert_series_equal(F[name].loc[target], F2[name].loc[target], obj=f"{name} k={k} D={target}")
            if F[name].loc[target].notna().any():
                checked.add(name)
    assert checked == set(F), f"never checked on real values: {set(F) - checked}"


def test_holt_winters_baseline_uses_only_days_before_the_origin():
    daily = build_daily(build_panel(random_intervals(), COVERAGE))
    Y = daily.count[["A1", "H1"]]
    origin = D("2025-11-01")
    Y2 = Y.copy()
    Y2[Y2.index >= origin] = 99.0
    a, b = holt_winters_daily(Y, pd.DatetimeIndex([origin])), holt_winters_daily(Y2, pd.DatetimeIndex([origin]))
    for k in (1, 3):
        day = origin + pd.Timedelta(days=k - 1)
        assert a[k]["mean"].loc[day].notna().all()
        pd.testing.assert_series_equal(a[k]["mean"].loc[day], b[k]["mean"].loc[day])


def test_daily_splits_keep_the_test_closed_and_validation_covered():
    days = pd.date_range("2026-03-01", "2026-10-10", freq="D")
    covered = pd.Series(True, index=days)
    covered[(days >= D("2026-09-06")) & (days <= D("2026-09-12"))] = False
    frozen = {"test_start": "2026-09-13T10:00:00+00:00", "test_end": "2026-10-11T10:00:00+00:00"}
    test, val, folds = make_daily_splits(covered, frozen)
    assert (test.start, test.end) == (D("2026-09-14"), D("2026-10-11"))  # whole Kyiv days only
    assert val.end == D("2026-09-06") and val.start == D("2026-08-09")
    assert covered[val.mask(days)].all()
    seq = folds + [val]
    for a, b in zip(seq, seq[1:]):
        assert a.end == b.start
    assert not any((s.mask(days) & test.mask(days)).any() for s in seq)
    # training target days for horizon k end at start - k: the last label known at the first origin
    assert val.train_end(1) == val.start and val.train_end(7) == val.start - pd.Timedelta(days=6)


def test_partial_day_overlapping_the_test_window_is_in_no_split():
    """Without a data gap, validation must still stop before the Kyiv day containing test_start."""
    days = pd.date_range("2026-03-01", "2026-10-10", freq="D")
    frozen = {"test_start": "2026-09-13T10:00:00+00:00", "test_end": "2026-10-11T10:00:00+00:00"}
    test, val, _ = make_daily_splits(pd.Series(True, index=days), frozen)
    assert test.start == D("2026-09-14")
    assert val.end == D("2026-09-13")  # exclusive: 09-13 (13:00 Kyiv onwards is test) is not validation


@pytest.mark.parametrize("k", [1, 4, 7])
def test_training_days_are_known_at_the_first_origin(k):
    days = pd.date_range("2026-01-01", "2026-09-30", freq="D")
    split = DaySplit("val", D("2026-08-09"), D("2026-09-06"))
    rows = split.train_mask(days, k, window_days=90)
    assert days[rows].max() == split.start - pd.Timedelta(days=k)  # origin = start - k + 1 knows day start - k
    assert rows.sum() == 90
    assert not (rows & split.mask(days)).any()


def test_regression_metrics():
    y, p = np.array([0.0, 2.0, 4.0]), np.array([1.0, 2.0, 2.0])
    s = regression_scores(y, p, reference_mae=2.0)
    assert s["mae"] == pytest.approx(1.0) and s["mase"] == pytest.approx(0.5)
    assert s["bias"] == pytest.approx(-1 / 3)
    assert poisson_deviance(y, y + 0.0) == pytest.approx(0.0, abs=1e-5)
    assert poisson_deviance(np.array([0.0]), np.array([1.0])) == pytest.approx(2.0)
    assert pinball(np.array([1.0]), np.array([0.0]), 0.9) == pytest.approx(0.9)
    assert interval_scores(y, y - 1, y + 1)["coverage"] == 1.0
