"""CLI: backtest baselines and models for target A on the rolling folds and the validation period.

For each horizon h:
  1. model selection on validation: training window (and LightGBM's tree size and,
     by early stopping, number of trees) per model, chosen by validation Brier;
  2. every split (6 weekly folds + validation) gets its own model, trained only on
     origins before split.train_end(h), and is scored on the same cells as the baselines;
  3. calibration check (calibrators fit on fold predictions, applied to validation),
     reliability tables, errors by oblast and by Kyiv hour, feature importance.

Text ablation (PLAN.md stage 4): when a snapshot of the Air Force channel exists
(scripts/fetch_telegram.py), every model is also run as `<model>_text` with the text
features added, selected and refit by the same protocol (`--no-text` skips it).

Validation scores of the models are optimistic (the configuration was selected on
validation). The folds are not used for selection, but they reuse settings chosen
on the later validation period, so only the final test is an unbiased estimate. The final test period is
deliberately not scored here; it is opened once, at the end of the project
(PLAN.md, section 4.1). The selected configuration goes to model_config.json for that run.

Usage:
    .venv/bin/python scripts/run_backtest.py [--no-text]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build import load  # noqa: E402
from src.data.telegram import latest_snapshot, load_messages  # noqa: E402
from src.evaluation.metrics import brier, calibration_table, paired, score, within_unit_auc  # noqa: E402
from src.evaluation.splits import load_frozen_test, make_splits  # noqa: E402
from src.features.asof import KYIV_TZ, build_features, history_features, stack  # noqa: E402
from src.features.panel import build_panel, labels  # noqa: E402
from src.features.text import tag_messages, text_features  # noqa: E402
from src.models.baselines import predict_all  # noqa: E402
from src.models.tabular import (  # noqa: E402
    Calibrator, LGBMSpec, fit_lgbm_early_stopping, make_logit, predict, to_wide, train_rows,
)

HORIZONS = (1, 3, 6)
REFERENCE = "unit_rate_28d"  # climatology-style reference for the Brier skill score
TRAIN_WINDOWS_DAYS = (60, 90, 180, 365)
LGBM_NUM_LEAVES = (7, 15)
WARMUP_DAYS = 35  # longest feature window (28-day rates) plus slack
MODEL_KINDS = ("logit", "lgbm")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "results"


def design(F, Y, rows):
    """Labelled long rows of the given origin rows."""
    X = stack(F, rows)
    y = Y.to_numpy()[rows].ravel()
    keep = ~np.isnan(y)
    return X[keep], y[keep]


def fit(model_name, config, F, Y, split, h, spec):
    X, y = design(F, Y, train_rows(Y.index, split, h, config["window_days"]))
    # Training label windows end at least h hours before the split starts (embargo)
    assert X.index.get_level_values("origin").max() + pd.Timedelta(hours=2 * h - 1) < split.start
    if model_name.startswith("logit"):
        return make_logit().fit(X, y)
    return spec.make(config["n_estimators"], num_leaves=config["num_leaves"]).fit(X, y)


def select(F, Y, validation, h, spec, suffix=""):
    """Training window (and tree settings) per model by validation Brier; names get `suffix`."""
    Xv, yv = design(F, Y, validation.mask(Y.index))
    rows, best = [], {}
    for window in TRAIN_WINDOWS_DAYS:
        X, y = design(F, Y, train_rows(Y.index, validation, h, window))
        candidates = [(f"logit{suffix}", make_logit().fit(X, y), {"window_days": window})]
        for leaves in LGBM_NUM_LEAVES:
            lgbm, n_trees = fit_lgbm_early_stopping(spec, X, y, Xv, yv, num_leaves=leaves)
            candidates.append((f"lgbm{suffix}", lgbm,
                               {"window_days": window, "num_leaves": leaves, "n_estimators": n_trees}))
        for name, model, cfg in candidates:
            b = brier(yv, predict(model, Xv))
            rows.append({"h": h, "model": name, **cfg, "val_brier": b})
            if name not in best or b < best[name][0]:
                best[name] = (b, cfg, model, Xv.columns)
    return {k: v[1] for k, v in best.items()}, best[f"lgbm{suffix}"][2], pd.DataFrame(rows)


def score_splits(Y, preds, cells_by_split, h):
    rows = []
    for split_name, cells in cells_by_split.items():
        ref_brier = brier(*paired(Y, preds[REFERENCE], cells))
        for name, p in preds.items():
            y, pv = paired(Y, p, cells)
            rows.append({"h": h, "split": split_name, "model": name, **score(y, pv, ref_brier),
                         "roc_auc_within": within_unit_auc(Y, p, cells)})
    return rows


def breakdown(Y, preds, cells, unit_oblast, h, model):
    """Brier of `model` vs the reference per oblast and per Kyiv hour of origin."""
    df = pd.DataFrame({
        "y": Y.to_numpy()[cells],
        "p": preds[model].reindex_like(Y).to_numpy()[cells],
        "ref": preds[REFERENCE].reindex_like(Y).to_numpy()[cells],
        "oblast": np.broadcast_to(np.array([unit_oblast[u] for u in Y.columns]), cells.shape)[cells],
        "kyiv_hour": np.broadcast_to(Y.index.tz_convert(KYIV_TZ).hour.to_numpy()[:, None], cells.shape)[cells],
    })
    df["se"], df["se_ref"] = (df["p"] - df["y"]) ** 2, (df["ref"] - df["y"]) ** 2
    out = []
    for key in ("oblast", "kyiv_hour"):
        g = df.groupby(key).agg(n=("y", "size"), base_rate=("y", "mean"), brier=("se", "mean"),
                                brier_ref=("se_ref", "mean"))
        g["bss"] = 1 - g["brier"] / g["brier_ref"]
        out.append(g.reset_index().assign(h=h, model=model, by=key).rename(columns={key: "group"}))
    return pd.concat(out, ignore_index=True)


def load_text():
    """Tagged Air Force posts and the start of their coverage, or None without a snapshot."""
    try:
        path = latest_snapshot("kpszsu")
    except FileNotFoundError:
        print("no Telegram snapshot: text ablation skipped (scripts/fetch_telegram.py)")
        return None
    # Edited posts stay (presence, report timing); their text-derived tags are NaN
    msgs = load_messages(path, drop_edited=False)
    print(f"text: {path.name}, {len(msgs)} posts ({msgs['edited'].sum()} edited: content ignored), "
          f"from {msgs['date_utc'].min()}")
    return tag_messages(msgs), msgs["date_utc"].min()


def evaluate(full_covered, intervals, coverage, h, spec, text=None):
    test, validation, folds = make_splits(full_covered, h, frozen=load_frozen_test())
    start = folds[0].train_end(h) - pd.Timedelta(days=max(TRAIN_WINDOWS_DAYS) + WARMUP_DAYS)
    panel = build_panel(intervals, coverage, start=start)
    unit_oblast = coverage["unit_oblast"]
    Y = labels(panel, h)
    F = build_features(panel, unit_oblast, Y, h, history_features(panel, unit_oblast))
    feature_sets = {"": F}
    if text is not None:
        feature_sets["_text"] = F | text_features(text[0], panel, unit_oblast, text[1])

    configs, lgbm_val, selections = {}, {}, []
    for suffix, Fs in feature_sets.items():
        cfg, lgbm_val[f"lgbm{suffix}"], sel = select(Fs, Y, validation, h, spec, suffix)
        configs |= cfg
        selections.append(sel)
    selection = pd.concat(selections, ignore_index=True)
    print(f"h={h}: selected {configs}", flush=True)
    models = [f"{kind}{suffix}" for suffix in feature_sets for kind in MODEL_KINDS]

    preds = predict_all(Y, h)
    model_preds = {m: [] for m in models}
    for split in folds + [validation]:
        rows = split.mask(Y.index)
        for suffix, Fs in feature_sets.items():
            X = stack(Fs, rows)
            for kind in MODEL_KINDS:
                m = f"{kind}{suffix}"
                model = fit(m, configs[m], Fs, Y, split, h, spec)
                model_preds[m].append(pd.Series(predict(model, X), index=X.index))
    for m in models:
        long = pd.concat(model_preds[m])
        preds[m] = to_wide(long.to_numpy(), long.index, Y)

    common = np.logical_and.reduce([~np.isnan(p.to_numpy()) for p in preds.values()])
    cells_by_split = {}
    for split_name, splits in [("folds", folds), ("validation", [validation])]:
        rowmask = np.logical_or.reduce([s.mask(Y.index) for s in splits])
        cells_by_split[split_name] = common & rowmask[:, None]
        # The final test is scored once, at the end of the project — never here
        assert not (cells_by_split[split_name] & test.mask(Y.index)[:, None]).any(), "refusing to score the final test"

    results = pd.DataFrame(score_splits(Y, preds, cells_by_split, h))

    # Calibration: fit on out-of-sample fold forecasts, check on validation
    cal_rows, rel = [], []
    for m in models:
        y_f, p_f = paired(Y, preds[m], cells_by_split["folds"])
        y_v, p_v = paired(Y, preds[m], cells_by_split["validation"])
        row = {"h": h, "model": m, "brier_raw": brier(y_v, p_v)}
        for method in ("platt", "isotonic"):
            row[f"brier_{method}"] = brier(y_v, Calibrator(method).fit(p_f, y_f).transform(p_v))
        cal_rows.append(row)
        for split_name, cells in cells_by_split.items():
            y, p = paired(Y, preds[m], cells)
            rel.append(calibration_table(y, p).reset_index().assign(h=h, model=m, split=split_name))

    all_cells = cells_by_split["folds"] | cells_by_split["validation"]
    errors = pd.concat([breakdown(Y, preds, all_cells, unit_oblast, h, m) for m in models], ignore_index=True)
    importance = []
    for name, booster in ((n, m.booster_) for n, m in lgbm_val.items()):
        imp = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
        importance.append((imp / imp.sum()).rename("gain_share").rename_axis("feature").reset_index().assign(model=name))
    importance = pd.concat(importance, ignore_index=True)

    return {
        "backtest": results,
        "model_selection": selection,
        "calibration_check": pd.DataFrame(cal_rows),
        "reliability": pd.concat(rel, ignore_index=True),
        "errors": errors,
        "feature_importance": importance.assign(h=h).sort_values("gain_share", ascending=False),
        "config": configs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-text", action="store_true", help="skip the text ablation")
    args = ap.parse_args()
    intervals, coverage = load()
    frozen = load_frozen_test()
    if frozen and frozen["data_cutoff"] != coverage["data_cutoff"]:
        print(f"note: data cutoff {coverage['data_cutoff']} differs from the one the test "
              f"was frozen on ({frozen['data_cutoff']}); the test window stays fixed")
    full_covered = build_panel(intervals, coverage).covered
    spec = LGBMSpec()
    text = None if args.no_text else load_text()
    out = [evaluate(full_covered, intervals, coverage, h, spec, text) for h in HORIZONS]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for key in ("backtest", "model_selection", "calibration_check", "reliability", "errors", "feature_importance"):
        pd.concat([o[key] for o in out], ignore_index=True).to_csv(
            RESULTS_DIR / f"{key}.csv", index=False, float_format="%.4f")
    config = {
        "data_cutoff": coverage["data_cutoff"],
        "reference": REFERENCE,
        "lgbm_params": spec.params,
        "horizons": {str(h): o["config"] for h, o in zip(HORIZONS, out)},
    }
    if text is not None:
        # Tagging rules are frozen for the final test (PLAN.md stage 4): any edit changes this hash
        text_py = Path(__file__).resolve().parents[1] / "src" / "features" / "text.py"
        config["text"] = {"snapshot": latest_snapshot("kpszsu").name,
                          "text_py_sha256": hashlib.sha256(text_py.read_bytes()).hexdigest()}
    (RESULTS_DIR / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = pd.concat([o["backtest"] for o in out], ignore_index=True)
    cols = ["h", "split", "model", "base_rate", "brier", "bss", "log_loss", "roc_auc", "roc_auc_within", "pr_auc"]
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(results[cols].to_string(index=False))
        print(pd.concat([o["calibration_check"] for o in out]).to_string(index=False))
    print(f"\nn per (h, split): {results.groupby(['h', 'split'])['n'].first().to_dict()}")
    print(f"saved {RESULTS_DIR}; data cutoff {coverage['data_cutoff']}")


if __name__ == "__main__":
    main()
