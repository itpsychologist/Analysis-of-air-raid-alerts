"""CLI: score baseline forecasts of target A on the rolling folds and the validation period.

The final test period is deliberately not scored here; it is opened once, at the
end of the project (PLAN.md, section 4.1).

Usage:
    .venv/bin/python scripts/run_backtest.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build import load  # noqa: E402
from src.evaluation.metrics import brier, paired, score, within_unit_auc  # noqa: E402
from src.evaluation.splits import load_frozen_test, make_splits  # noqa: E402
from src.features.panel import build_panel, labels  # noqa: E402
from src.models.baselines import predict_all  # noqa: E402

HORIZONS = (1, 3, 6)
REFERENCE = "unit_rate_28d"  # climatology-style reference for the Brier skill score
RESULTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "results"


def evaluate(panel, h: int) -> pd.DataFrame:
    Y = labels(panel, h)
    preds = predict_all(Y, h)
    # Compare every baseline on exactly the same cells
    common = np.logical_and.reduce([~np.isnan(p.to_numpy()) for p in preds.values()])
    test, validation, folds = make_splits(panel.covered, h, frozen=load_frozen_test())

    rows = []
    for split_name, splits in [("folds", folds), ("validation", [validation])]:
        rowmask = np.zeros(len(Y), dtype=bool)
        for s in splits:
            rowmask |= s.mask(Y.index)
        cells = common & rowmask[:, None]
        # The final test is scored once, at the end of the project — never here
        assert not (cells & test.mask(Y.index)[:, None]).any(), "refusing to score the final test"
        ref_brier = brier(*paired(Y, preds[REFERENCE], cells))
        for name, p in preds.items():
            y, pv = paired(Y, p, cells)
            rows.append({"h": h, "split": split_name, "model": name, **score(y, pv, ref_brier),
                         "roc_auc_within": within_unit_auc(Y, p, cells)})
    return pd.DataFrame(rows)


def main() -> None:
    intervals, coverage = load()
    frozen = load_frozen_test()
    if frozen and frozen["data_cutoff"] != coverage["data_cutoff"]:
        print(f"note: data cutoff {coverage['data_cutoff']} differs from the one the test "
              f"was frozen on ({frozen['data_cutoff']}); the test window stays fixed")
    panel = build_panel(intervals, coverage)
    results = pd.concat([evaluate(panel, h) for h in HORIZONS], ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "baselines.csv"
    results.to_csv(out, index=False, float_format="%.4f")

    cols = ["h", "split", "model", "base_rate", "brier", "bss", "log_loss", "roc_auc", "roc_auc_within", "pr_auc"]
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(results[cols].to_string(index=False))
    print(f"\nn per (h, split): {results.groupby(['h', 'split'])['n'].first().to_dict()}")
    print(f"saved {out}; data cutoff {coverage['data_cutoff']}")


if __name__ == "__main__":
    main()
