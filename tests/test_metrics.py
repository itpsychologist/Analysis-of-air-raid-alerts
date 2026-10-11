import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import brier, calibration_table, paired, score, within_unit_auc


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
