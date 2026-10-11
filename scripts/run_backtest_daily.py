"""CLI: backtest target B — daily alert count and hours under alert per raion, 1-7 days ahead.

For each target (count, duration) and horizon k (days):
  1. model selection on validation: training window (and LightGBM's tree size and,
     by early stopping, number of trees) per model, chosen by validation Poisson deviance;
  2. every split (6 weekly folds + validation) gets its own models, trained only on
     target days <= split.start - k (the labels known at the split's first origin);
  3. point forecasts are scored against baselines (incl. the dashboard's Holt-Winters)
     on the same cells; 80/95% intervals from LightGBM quantile models and the
     Holt-Winters 95% interval are checked for coverage.

As for target A, validation scores of models are optimistic (selection) and the folds
reuse settings chosen on the later validation period; only the final test is
unbiased, and it is never scored here (PLAN.md, section 4.1).

Usage:
    .venv/bin/python scripts/run_backtest_daily.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build import load, windows_fingerprint  # noqa: E402
from src.evaluation.metrics import interval_scores, paired, pinball, poisson_deviance, regression_scores  # noqa: E402
from src.evaluation.splits import load_frozen_test  # noqa: E402
from src.features.asof import history_features, stack  # noqa: E402
from src.features.daily import (  # noqa: E402
    MAX_HORIZON_DAYS, TARGETS, build_daily, for_horizon, make_daily_splits, origin_features,
)
from src.features.panel import build_panel  # noqa: E402
from src.models.baselines import DAILY_BASELINES, holt_winters_daily  # noqa: E402
from src.models.tabular import LGBMSpec, fit_lgbm_early_stopping, make_poisson_glm, predict, to_wide  # noqa: E402

HORIZONS = range(1, MAX_HORIZON_DAYS + 1)
REFERENCE = "seasonal_naive"  # MASE denominator (PLAN.md 4.3)
TRAIN_WINDOWS_DAYS = (90, 180, 365)
LGBM_NUM_LEAVES = (7, 15)
QUANTILES = (0.025, 0.1, 0.9, 0.975)
WARMUP_DAYS = 60
RESULTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "results"

SPEC = LGBMSpec()
SPEC.params.update(objective="poisson", min_child_samples=200)


def design(F, Y, rows):
    X = stack(F, rows)
    y = Y.to_numpy()[rows].ravel()
    keep = ~np.isnan(y)
    return X[keep], y[keep]


def select(F, Y, validation, k):
    Xv, yv = design(F, Y, validation.mask(Y.index))
    rows, best = [], {}
    for window in TRAIN_WINDOWS_DAYS:
        X, y = design(F, Y, validation.train_mask(Y.index, k, window))
        assert X.index.get_level_values(0).max() <= validation.start - pd.Timedelta(days=k)
        candidates = [("poisson_glm", make_poisson_glm().fit(X, y), {"window_days": window})]
        for leaves in LGBM_NUM_LEAVES:
            m, n_trees = fit_lgbm_early_stopping(SPEC, X, y, Xv, yv, num_leaves=leaves)
            candidates.append(("lgbm", m, {"window_days": window, "num_leaves": leaves, "n_estimators": n_trees}))
        for name, model, cfg in candidates:
            dev = poisson_deviance(yv, predict(model, Xv))
            rows.append({"k": k, "model": name, **cfg, "val_poisson_dev": dev})
            if name not in best or dev < best[name][0]:
                best[name] = (dev, cfg)
    return {m: v[1] for m, v in best.items()}, pd.DataFrame(rows)


def fit_split(F, Y, split, k, configs):
    """Point models + LightGBM quantile models for one split; returns long predictions."""
    out = {}
    X_pred = stack(F, split.mask(Y.index))
    for name, cfg in configs.items():
        X, y = design(F, Y, split.train_mask(Y.index, k, cfg["window_days"]))
        last = X.index.get_level_values(0).max()
        assert last <= split.start - pd.Timedelta(days=k), "training label not known at the first origin"
        if name == "poisson_glm":
            out[name] = predict(make_poisson_glm().fit(X, y), X_pred)
        else:
            params = {"num_leaves": cfg["num_leaves"]}
            out[name] = predict(SPEC.make(cfg["n_estimators"], **params).fit(X, y), X_pred)
            for a in QUANTILES:
                q = SPEC.make(cfg["n_estimators"], objective="quantile", alpha=a, **params).fit(X, y)
                out[f"q{a}"] = predict(q, X_pred)
    return {name: pd.Series(p, index=X_pred.index) for name, p in out.items()}


def evaluate_target(target, daily, F, splits):
    test, validation, folds = splits
    Y = daily.target(target)
    origins = Y.index[(Y.index >= folds[0].start - pd.Timedelta(days=MAX_HORIZON_DAYS)) & (Y.index < validation.end)]
    hw = holt_winters_daily(Y, origins)
    scores, intervals, selections, configs_k = [], [], [], {}
    for k in HORIZONS:
        Fk = for_horizon(F, daily, k)
        configs, sel = select(Fk, Y, validation, k)
        selections.append(sel.assign(target=target))
        configs_k[k] = configs
        long = {}
        for split in folds + [validation]:
            for name, s in fit_split(Fk, Y, split, k, configs).items():
                long.setdefault(name, []).append(s)
        model_preds = {name: to_wide(pd.concat(v).to_numpy(), pd.concat(v).index, Y) for name, v in long.items()}

        preds = {name: fn(Y, k) for name, fn in DAILY_BASELINES.items()}
        preds["holt_winters"] = hw[k]["mean"]
        preds["poisson_glm"], preds["lgbm"] = model_preds["poisson_glm"], model_preds["lgbm"]
        common = Y.notna().to_numpy() & np.logical_and.reduce([p.notna().to_numpy() for p in preds.values()])
        for split_name, group in (("folds", folds), ("validation", [validation])):
            rowmask = np.logical_or.reduce([s.mask(Y.index) for s in group])
            cells = common & rowmask[:, None]
            # The final test is scored once, at the end of the project — never here
            assert not (cells & test.mask(Y.index)[:, None]).any(), "refusing to score the final test"
            y_ref, p_ref = paired(Y, preds[REFERENCE], cells)
            ref_mae = float(np.mean(np.abs(p_ref - y_ref)))
            for name, p in preds.items():
                y, pv = paired(Y, p, cells)
                scores.append({"target": target, "k": k, "split": split_name, "model": name,
                               **regression_scores(y, pv, ref_mae)})
            y = Y.to_numpy()[cells]
            q = {a: model_preds[f"q{a}"].to_numpy()[cells] for a in QUANTILES}
            for level, (lo, hi) in ((80, (0.1, 0.9)), (95, (0.025, 0.975))):
                intervals.append({"target": target, "k": k, "split": split_name, "model": "lgbm_quantile",
                                  "level": level, **interval_scores(y, q[lo], q[hi]),
                                  "pinball": np.mean([pinball(y, q[lo], lo), pinball(y, q[hi], hi)])})
            intervals.append({"target": target, "k": k, "split": split_name, "model": "holt_winters", "level": 95,
                              **interval_scores(y, hw[k]["lower"].to_numpy()[cells], hw[k]["upper"].to_numpy()[cells]),
                              "pinball": np.mean([pinball(y, hw[k]["lower"].to_numpy()[cells], 0.025),
                                                  pinball(y, hw[k]["upper"].to_numpy()[cells], 0.975)])})
        print(f"{target} k={k}: {configs}", flush=True)
    return pd.DataFrame(scores), pd.DataFrame(intervals), pd.concat(selections), configs_k


def main() -> None:
    intervals, coverage = load()
    frozen = load_frozen_test()
    if frozen is None:
        raise SystemExit("config/evaluation.json (frozen final test) is required")
    full_daily = build_daily(build_panel(intervals, coverage))
    splits = make_daily_splits(full_daily.count.notna().any(axis=1), frozen)
    test, validation, folds = splits
    print(f"folds {folds[0].start.date()}..{folds[-1].end.date()}, validation {validation.start.date()}.."
          f"{validation.end.date()}, test (not scored) {test.start.date()}..{test.end.date()}")

    start = (folds[0].start - pd.Timedelta(days=max(TRAIN_WINDOWS_DAYS) + WARMUP_DAYS)).tz_localize("Europe/Kyiv")
    panel = build_panel(intervals, coverage, start=start.tz_convert("UTC"))
    daily = build_daily(panel)
    F = origin_features(daily, panel, coverage["unit_oblast"], history_features(panel, coverage["unit_oblast"]))

    out = [evaluate_target(t, daily, F, splits) for t in TARGETS]
    scores = pd.concat([o[0] for o in out], ignore_index=True)
    ivals = pd.concat([o[1] for o in out], ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS_DIR / "backtest_daily.csv", index=False, float_format="%.4f")
    ivals.to_csv(RESULTS_DIR / "intervals_daily.csv", index=False, float_format="%.4f")
    pd.concat([o[2] for o in out], ignore_index=True).to_csv(
        RESULTS_DIR / "model_selection_daily.csv", index=False, float_format="%.4f")
    config = {
        "data_cutoff": coverage["data_cutoff"],
        "coverage_windows_sha256": windows_fingerprint(coverage),
        "reference": REFERENCE,
        "lgbm_params": SPEC.params,
        "quantiles": QUANTILES,
        "targets": {t: {str(k): c for k, c in o[3].items()} for t, o in zip(TARGETS, out)},
    }
    (RESULTS_DIR / "model_config_daily.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        for (target, split), g in scores.groupby(["target", "split"]):
            print(f"\n{target} / {split}: MASE by horizon (vs {REFERENCE})")
            print(g.pivot(index="model", columns="k", values="mase").to_string())
        print("\nInterval coverage (folds):")
        print(ivals[ivals.split == "folds"].pivot_table(index=["target", "model", "level"], columns="k",
                                                         values="coverage").to_string())
    print(f"\nsaved {RESULTS_DIR}; data cutoff {coverage['data_cutoff']}")


if __name__ == "__main__":
    main()
