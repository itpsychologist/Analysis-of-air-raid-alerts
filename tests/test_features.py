"""
Leakage and unit tests for the "as of t" features (PLAN.md, section 3.2) and model training rows.
"""

import numpy as np
import pandas as pd
import pytest

from etl import OBLAST_UID_MAP
from src.data.merge import EXCLUDED_OBLASTS
from src.evaluation.splits import Split
from src.features.asof import build_features, stack
from src.features.geo import group_matrices, neighbors
from src.features.panel import build_panel, labels
from src.models.tabular import train_rows

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731

UNIT_OBLAST = {
    "A1": "Київська область", "A2": "Київська область", "K": "м. Київ",
    "P1": "Полтавська область", "H1": "Харківська область", "H2": "Харківська область",
}
COVERAGE = {
    "windows": [
        {"source": "official", "start": "2025-10-01T00:00:00+00:00", "end": "2025-10-20T07:30:00+00:00"},
        {"source": "gap", "start": "2025-10-20T07:30:00+00:00", "end": "2025-10-22T00:00:00+00:00"},
        {"source": "alerts_in_ua", "start": "2025-10-22T00:00:00+00:00", "end": "2025-11-05T00:00:00+00:00"},
    ],
    "data_cutoff": "2025-11-05T00:00:00+00:00",
    "units": sorted(UNIT_OBLAST),
    "unit_oblast": UNIT_OBLAST,
}


def random_intervals(seed=0, start=T("2025-10-01"), end=T("2025-11-05")) -> pd.DataFrame:
    """Disjoint random alert intervals per unit, minute resolution."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in UNIT_OBLAST:
        t = start + pd.Timedelta(minutes=int(rng.integers(0, 600)))
        while t < end:
            dur = pd.Timedelta(minutes=int(rng.integers(5, 300)))
            rows.append((unit, t, min(t + dur, end)))
            t = t + dur + pd.Timedelta(minutes=int(rng.integers(10, 900)))
    return pd.DataFrame(rows, columns=["unit", "started_at", "finished_at"])


def perturb_future(iv: pd.DataFrame, t: pd.Timestamp, seed: int) -> pd.DataFrame:
    """
    Rewrite everything that is unknown at t: alerts running at t get a random end
    (still after t), alerts starting at or after t are replaced by random new ones.
    """
    rng = np.random.default_rng(seed)
    past = iv[iv["started_at"] < t].copy()
    running = past["finished_at"] > t
    past.loc[running, "finished_at"] = t + pd.to_timedelta(rng.integers(1, 2000, running.sum()), unit="min")
    new = random_intervals(seed + 100, start=t, end=T("2025-11-05"))
    last_end = past.groupby("unit")["finished_at"].max()
    new = new[new["started_at"] > new["unit"].map(last_end).fillna(t)]
    return pd.concat([past, new], ignore_index=True) if len(new) else past.reset_index(drop=True)


ORIGINS = [
    T("2025-10-14 09:00"), T("2025-10-20 07:00"), T("2025-10-20 08:00"),  # around the window end 07:30
    T("2025-10-22 00:00"), T("2025-10-22 01:00"), T("2025-10-22 03:00"),  # first hours after the gap
    T("2025-10-26 01:00"), T("2025-10-26 02:00"),  # autumn DST change
    T("2025-11-02 18:00"), T("2025-11-04 23:00"),  # deep in history, last hour before cutoff
]


def boundary_intervals() -> pd.DataFrame:
    """Random intervals plus an alert starting exactly on an hour and one ending exactly on it."""
    iv = random_intervals()
    iv = iv[~((iv["unit"] == "K") & (iv["finished_at"] > T("2025-11-02 12:00")) & (iv["started_at"] < T("2025-11-03")))]
    extra = pd.DataFrame({"unit": ["K", "K"],
                          "started_at": [T("2025-11-02 15:00"), T("2025-11-02 18:20")],
                          "finished_at": [T("2025-11-02 18:00"), T("2025-11-02 19:00")]})
    return pd.concat([iv, extra], ignore_index=True)


@pytest.mark.parametrize("h", [1, 3, 6])
def test_features_at_t_ignore_everything_after_t(h):
    """Future-perturbation test (PLAN 3.2) for every feature, on raw intervals."""
    iv = boundary_intervals()
    panel = build_panel(iv, COVERAGE)
    F = build_features(panel, UNIT_OBLAST, labels(panel, h), h)
    # Guard: every feature is actually checked somewhere (not NaN at all tested origins)
    for name, mat in F.items():
        assert mat.loc[ORIGINS].notna().any().any(), f"{name} is NaN at every tested origin"
    for i, t in enumerate(ORIGINS):
        for seed in (i, i + 50):
            p2 = build_panel(perturb_future(iv, t, seed=seed), COVERAGE)
            F2 = build_features(p2, UNIT_OBLAST, labels(p2, h), h)
            for name in F:
                pd.testing.assert_series_equal(F[name].loc[t], F2[name].loc[t], obj=f"{name} h={h} at {t}")


def test_perturbation_actually_changes_the_future():
    """Guard for the test above: the perturbation must change data after t."""
    iv = random_intervals()
    t = ORIGINS[0]
    a, b = build_panel(iv, COVERAGE), build_panel(perturb_future(iv, t, 0), COVERAGE)
    after = a.hours >= t
    assert not a.starts[after].equals(b.starts[after])
    F, F2 = build_features(a, UNIT_OBLAST, labels(a, 3), 3), build_features(b, UNIT_OBLAST, labels(b, 3), 3)
    later = t + pd.Timedelta(hours=6)
    assert not F["unit_active_3h"].loc[later].equals(F2["unit_active_3h"].loc[later])


def test_active_at_is_strictly_inside_the_interval():
    iv = pd.DataFrame({
        "unit": ["A1", "A1"],
        "started_at": [T("2025-10-02 03:00"), T("2025-10-02 06:20")],
        "finished_at": [T("2025-10-02 05:00"), T("2025-10-02 07:10")],
    })
    act = build_panel(iv, COVERAGE).active_at["A1"]
    assert act[T("2025-10-02 03:00")] == 0  # starts exactly at t: that is the label, not the state
    assert act[T("2025-10-02 04:00")] == 1
    assert act[T("2025-10-02 05:00")] == 0  # finished exactly at t
    assert act[T("2025-10-02 07:00")] == 1
    assert act[T("2025-10-02 08:00")] == 0


def test_features_are_nan_right_after_a_gap_and_use_only_the_previous_hour():
    iv = pd.DataFrame({
        "unit": ["A1"], "started_at": [T("2025-10-22 05:10")], "finished_at": [T("2025-10-22 05:40")],
    })
    panel = build_panel(iv, COVERAGE)
    F = build_features(panel, UNIT_OBLAST, labels(panel, 1), 1)
    assert np.isnan(F["unit_starts_1h"].loc[T("2025-10-22 00:00"), "A1"])  # hour before was the gap
    assert np.isnan(F["unit_since_start"].loc[T("2025-10-22 03:00"), "A1"])  # last start unknown
    assert F["unit_starts_1h"].loc[T("2025-10-22 05:00"), "A1"] == 0  # start in hour 05 is not visible at 05:00
    assert F["unit_starts_1h"].loc[T("2025-10-22 06:00"), "A1"] == 1
    assert F["unit_since_start"].loc[T("2025-10-22 06:00"), "A1"] == 0
    assert F["unit_since_start"].loc[T("2025-10-22 09:00"), "A1"] == pytest.approx(np.log1p(3))
    # neighbours of P1 (Poltava) include Kyiv oblast (A1, A2) and Kharkiv (H1, H2): 1 start of 4 units
    assert F["nbr_starts_1h"].loc[T("2025-10-22 06:00"), "P1"] == pytest.approx(0.25)
    assert F["oblast_starts_1h"].loc[T("2025-10-22 06:00"), "A2"] == pytest.approx(0.5)


def test_calendar_uses_kyiv_local_time():
    panel = build_panel(random_intervals(), COVERAGE)
    F = build_features(panel, UNIT_OBLAST, labels(panel, 1), 1)
    # 21:00 UTC is midnight in Kyiv in summer time (UTC+3) -> hour 0
    assert F["cal_hour_sin"].loc[T("2025-10-01 21:00"), "A1"] == pytest.approx(0.0)
    assert F["cal_hour_cos"].loc[T("2025-10-01 21:00"), "A1"] == pytest.approx(1.0)


def test_stack_is_origin_major():
    panel = build_panel(random_intervals(), COVERAGE)
    F = build_features(panel, UNIT_OBLAST, labels(panel, 1), 1)
    rows = np.zeros(len(panel.hours), bool)
    rows[100:103] = True
    X = stack(F, rows)
    assert len(X) == 3 * len(UNIT_OBLAST)
    assert X.loc[(panel.hours[101], "H2"), "unit_active_24h"] == pytest.approx(F["unit_active_24h"].iloc[101]["H2"])


def test_oblast_adjacency_is_symmetric_and_covers_all_units():
    adj = neighbors()
    valid = set(OBLAST_UID_MAP) - EXCLUDED_OBLASTS
    assert set(adj) == valid
    for a, bs in adj.items():
        assert bs <= valid and a not in bs
        assert all(a in adj[b] for b in bs)
    same, nbr = group_matrices(list(UNIT_OBLAST), UNIT_OBLAST)
    np.testing.assert_allclose(same.sum(1), 1)
    np.testing.assert_allclose(nbr.sum(1), 1)


@pytest.mark.parametrize("h", [1, 3, 6, 24])
def test_training_rows_keep_the_embargo(h):
    hours = pd.date_range(T("2026-01-01"), T("2026-09-01"), freq="h", inclusive="left")
    split = Split("val", T("2026-08-09 03:00"), T("2026-08-16 03:00"))
    rows = train_rows(hours, split, h, window_days=90)
    last = hours[rows].max()
    assert last + pd.Timedelta(hours=h) <= split.start - pd.Timedelta(hours=h)  # label window + embargo
    assert hours[rows].min() >= split.train_end(h) - pd.Timedelta(days=90)
