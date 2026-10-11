import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    block_sums, bootstrap_skill, brier, calibration_table, paired, poisson_deviance, poisson_deviance_cells, score,
    within_unit_auc,
)
from src.models.tabular import conformal_margin


def test_score_perfect_and_reference_skill():
    y = np.array([0, 0, 1, 1.0])
    perfect = score(y, y, reference_brier=0.25)
    assert perfect["brier"] == 0 and perfect["bss"] == 1 and perfect["roc_auc"] == 1
    assert np.isfinite(perfect["log_loss"])  # 0/1 forecasts are clipped, not infinite
    climatology = score(y, np.full(4, 0.5), reference_brier=0.25)
    assert climatology["bss"] == pytest.approx(0) and climatology["roc_auc"] == 0.5


def test_paired_drops_cells_missing_on_either_side_and_respects_mask():
    y = pd.DataFrame({"a": [1.0, np.nan, 0.0], "b": [0.0, 1.0, 1.0]})
    p = pd.DataFrame({"a": [0.9, 0.5, np.nan], "b": [0.1, 0.8, 0.7]})
    yv, pv = paired(y, p)
    assert yv.tolist() == [1.0, 0.0, 1.0, 1.0] and pv.tolist() == [0.9, 0.1, 0.8, 0.7]
    mask = np.array([[True, False], [True, False], [True, True]])
    assert paired(y, p, mask)[0].tolist() == [1.0, 1.0]
    assert brier(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == 0.25


def test_within_unit_auc_ignores_between_unit_differences():
    # Unit "hot" is always forecast higher than "cold" (perfect pooled ranking signal),
    # but within each unit the forecast is anti-correlated with the outcome.
    y = pd.DataFrame({"hot": [1.0, 1.0, 0.0, 1.0], "cold": [0.0, 1.0, 0.0, 0.0]})
    p = pd.DataFrame({"hot": [0.8, 0.8, 0.9, 0.8], "cold": [0.2, 0.1, 0.2, 0.2]})
    cells = np.ones(y.shape, dtype=bool)
    assert within_unit_auc(y, p, cells) == 0.0
    yv, pv = paired(y, p)
    assert score(yv, pv)["roc_auc"] > 0.5


def test_calibration_table_bins():
    t = calibration_table(np.array([0, 1, 1, 0.0]), np.array([0.05, 0.95, 0.91, 0.12]), bins=10)
    assert t.loc[9, "observed"] == 1.0 and t.loc[9, "n"] == 2
    assert t.loc[0, "observed"] == 0.0 and t.loc[1, "n"] == 1


def test_block_bootstrap_skill_resamples_whole_days():
    loss_ref = np.array([1.0, 0, 0, 1.0])
    blocks = np.array(["d1", "d1", "d2", "d2"])
    sums, labels = block_sums(loss_ref, blocks)
    assert sums.tolist() == [1.0, 1.0] and labels.tolist() == ["d1", "d2"]

    rng = np.random.default_rng(1)
    ref = rng.uniform(1, 2, 60)
    better = bootstrap_skill(0.8 * ref, ref)
    assert better["skill"] == pytest.approx(0.2) and better["ci_low"] == pytest.approx(0.2)
    assert better["share_not_better"] == 0 and better["n_blocks"] == 60
    noisy = bootstrap_skill(ref * rng.uniform(0.5, 1.5, 60), ref)
    assert noisy["ci_low"] < noisy["skill"] < noisy["ci_high"]


def test_conformal_margin_reaches_the_level_on_exchangeable_data():
    rng = np.random.default_rng(0)
    y_cal, y_new = rng.normal(size=2000), rng.normal(size=20000)
    narrow = (-0.5, 0.5)  # covers ~38%: the margin has to widen it
    m = conformal_margin(y_cal, np.full(2000, narrow[0]), np.full(2000, narrow[1]), 0.9)
    assert m > 0
    cover = np.mean((y_new >= narrow[0] - m) & (y_new <= narrow[1] + m))
    assert 0.88 < cover < 0.92
    assert conformal_margin(y_cal, np.full(2000, -5.0), np.full(2000, 5.0), 0.9) < 0  # over-wide: narrowed


def test_poisson_deviance_cells_average_to_the_mean():
    y, mu = np.array([0, 2, 3.5]), np.array([0.5, 2, 1.0])
    cells = poisson_deviance_cells(y, mu)
    assert cells[1] == 0 and poisson_deviance(y, mu) == pytest.approx(cells.mean())
