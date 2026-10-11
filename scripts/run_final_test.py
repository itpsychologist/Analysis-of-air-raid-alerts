"""CLI: the one-time evaluation on the frozen final test (PLAN.md, section 4.1, stage 5).

Nothing is selected here. Every model uses the configuration written by the backtests
(reports/results/model_config.json, model_config_daily.json) and is trained exactly
like a backtest split, with the final test as the split:

  - target A (h = 1, 3, 6 hours): origins t + h <= test start - h (embargo), within the
    selected training window;
  - target B (k = 1..7 days): target days <= first test day - k; LightGBM quantile
    intervals are widened by split-conformal margins fit on the validation period
    (the validation-split models' out-of-sample intervals), as planned in stage 3.

Scored cells: the test starts a week after the data gap (2026-09-06 .. 09-12), so
"same hour / day last week" and a few short-window baselines are undefined on the
first test days. Models and the reference are scored on every test cell; a forecast
that is undefined somewhere is scored on its own subset, against the reference on
that same subset (`n` says how many cells). Target B also has a "common" cell set
(every forecast defined) where MASE vs the seasonal naive is computed.

Significance: block bootstrap over Kyiv days (1-day and 3-day blocks) of the skill
vs a reference. The text ablation (`*_text`) runs only if src/features/text.py still
has the hash frozen in model_config.json.

The test is opened once: the run refuses to start with uncommitted code in src/,
scripts/ or config/, writes reports/results/final_test.json ("opened") before it
scores anything, and refuses to run again once that file exists (`--force` re-runs,
archives the earlier outputs and is recorded in the file). `--rehearse DIR` runs the
whole pipeline with the backtest's validation window (h = 1) in place of the test and
writes to DIR: a check that never opens the test; for h = 1 it reproduces the
validation rows of backtest.csv.

Usage:
    .venv/bin/python scripts/run_final_test.py [--rehearse DIR]
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_backtest import MODEL_KINDS, REFERENCE, TRAIN_WINDOWS_DAYS, WARMUP_DAYS, breakdown, fit  # noqa: E402
from scripts import run_backtest_daily as daily_bt  # noqa: E402
from src.data.build import find_outages, load, windows_fingerprint  # noqa: E402
from src.data.telegram import RAW_DIR as TELEGRAM_DIR, load_messages  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    block_sums, bootstrap_skill, brier, calibration_table, interval_scores, paired, pinball,
    poisson_deviance_cells, regression_scores, score, within_unit_auc,
)
from src.evaluation.splits import load_frozen_test, make_splits  # noqa: E402
from src.features.asof import KYIV_TZ, build_features, history_features, stack  # noqa: E402
from src.features.daily import TARGETS, build_daily, for_horizon, make_daily_splits, origin_features  # noqa: E402
from src.features.panel import build_panel, labels  # noqa: E402
from src.features.text import tag_messages, text_features  # noqa: E402
from src.models.baselines import DAILY_BASELINES, holt_winters_daily, predict_all  # noqa: E402
from src.models.tabular import LGBMSpec, conformal_margin, predict, to_wide  # noqa: E402

HORIZONS = (1, 3, 6)
RESULTS_DIR = ROOT / "reports" / "results"
MARKER = RESULTS_DIR / "final_test.json"
DAY_HOURS = range(7, 22)  # Kyiv origin hours counted as "day" (07:00-21:59)
N_BOOT = 2000
BLOCK_DAYS = (1, 3)
DAILY_REF = "mean_28d"  # reference for target B on all test days (MASE needs the seasonal naive)
FROZEN_CODE = ("src", "scripts", "config")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Checks: the run must use exactly what was frozen
# ---------------------------------------------------------------------------

def check_inputs(coverage, frozen, config, config_daily):
    used = {w["snapshot"] for w in coverage["windows"] if "snapshot" in w}
    if used != set(frozen["snapshots"].values()):
        raise SystemExit(f"data snapshots {sorted(used)} differ from the frozen {frozen['snapshots']}")
    for name, c in (("model_config.json", config), ("model_config_daily.json", config_daily)):
        if c["data_cutoff"] != coverage["data_cutoff"]:
            raise SystemExit(f"{name} was selected on data cut at {c['data_cutoff']}, not {coverage['data_cutoff']}")
        # Same snapshots and cutoff are not enough: the selection must have run on this dataset build
        if c.get("coverage_windows_sha256") != windows_fingerprint(coverage):
            raise SystemExit(f"{name} was selected on a different dataset build (coverage windows); "
                             "re-run the backtests")
    if LGBMSpec().params != config["lgbm_params"]:
        raise SystemExit("LightGBM parameters differ from model_config.json")
    if daily_bt.SPEC.params != config_daily["lgbm_params"]:
        raise SystemExit("LightGBM parameters differ from model_config_daily.json")
    text_ok = False
    if "text" in config:
        text_ok = sha256(ROOT / "src" / "features" / "text.py") == config["text"]["text_py_sha256"]
        if not text_ok:
            print("WARNING: src/features/text.py changed since selection: text ablation skipped")
    return text_ok


def test_period_outages(intervals, frozen) -> list[list[str]]:
    """
    The rule that masks official-feed outages (build.find_outages, > 48 h without a start
    anywhere), applied to the API data of the test window. Fixed before scoring: any hit
    aborts the run, because a dead feed would be scored as "no alert".
    """
    t0, t1 = pd.Timestamp(frozen["test_start"]), pd.Timestamp(frozen["test_end"])
    api = intervals[intervals["source"] == "alerts_in_ua"]
    starts = pd.concat([pd.Series([t0]), api.loc[(api.started_at >= t0) & (api.started_at < t1), "started_at"],
                        pd.Series([t1])], ignore_index=True)
    return [[str(a), str(b)] for a, b in find_outages(starts)]


def uncommitted_code() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain", "--", *FROZEN_CODE], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def load_text(config):
    path = TELEGRAM_DIR / "kpszsu" / config["text"]["snapshot"]
    msgs = load_messages(path, drop_edited=False)
    return tag_messages(msgs), msgs["date_utc"].min()


# ---------------------------------------------------------------------------
# Shared scoring helpers
# ---------------------------------------------------------------------------

def defined(p: pd.DataFrame, like: pd.DataFrame) -> np.ndarray:
    return ~np.isnan(p.reindex_like(like).to_numpy())


def bootstrap_rows(loss: dict[str, np.ndarray], days: np.ndarray, ok: dict[str, np.ndarray],
                   comparisons, **key) -> list[dict]:
    """
    Block bootstrap of 1 - sum(loss[m]) / sum(loss[ref]) over the cells where both are
    defined (`ok`); blocks are runs of `BLOCK_DAYS` consecutive days.
    `loss`, `days` and `ok[m]` are aligned flat arrays over the scored cells.
    """
    day_no = ((days - days.min()) // np.timedelta64(1, "D")).astype(int)
    rows = []
    for m, ref in comparisons:
        if m not in loss or ref not in loss:
            continue
        both = ok[m] & ok[ref]
        for block_days in BLOCK_DAYS:
            blocks = day_no[both] // block_days
            rows.append({**key, "model": m, "vs": ref, "block_days": block_days,
                         **bootstrap_skill(block_sums(loss[m][both], blocks)[0],
                                           block_sums(loss[ref][both], blocks)[0], N_BOOT)})
    return rows


# ---------------------------------------------------------------------------
# Target A
# ---------------------------------------------------------------------------

def kyiv_day_and_hour(index: pd.DatetimeIndex, shape) -> tuple[np.ndarray, np.ndarray]:
    local = index.tz_convert(KYIV_TZ)
    day = np.broadcast_to(local.normalize().tz_localize(None).to_numpy()[:, None], shape)
    hour = np.broadcast_to(local.hour.to_numpy()[:, None], shape)
    return day, hour


def score_hourly(Y, preds, cells, ok, h):
    """Every forecast on `cells` where it is defined, BSS vs the reference on the same cells."""
    day, _ = kyiv_day_and_hour(Y.index, cells.shape)
    rows = []
    for m, p in preds.items():
        c = cells & ok[m]
        y, pv = paired(Y, p, c)
        _, p_ref = paired(Y, preds[REFERENCE], c)
        rows.append({"h": h, "split": "test", "model": m, **score(y, pv, brier(y, p_ref)),
                     "roc_auc_within": within_unit_auc(Y, p, c), "n_days": len(np.unique(day[c])),
                     "first_origin": str(Y.index[c.any(axis=1)][0]) if c.any() else None})
    return pd.DataFrame(rows)


def by_period(Y, preds, cells, models, h):
    """Brier and BSS vs the reference for Kyiv day / night origins."""
    _, hour = kyiv_day_and_hour(Y.index, cells.shape)
    rows = []
    for period, sel in (("day", np.isin(hour, DAY_HOURS)), ("night", ~np.isin(hour, DAY_HOURS))):
        c = cells & sel
        y, p_ref = paired(Y, preds[REFERENCE], c)
        ref = brier(y, p_ref)
        for m in models:
            y, p = paired(Y, preds[m], c)
            b = brier(y, p)
            rows.append({"h": h, "model": m, "by": "period", "group": period, "n": len(y),
                         "base_rate": float(y.mean()), "brier": b, "brier_ref": ref, "bss": 1 - b / ref})
    return pd.DataFrame(rows)


def oblast_hourly(Y, preds, cells, unit_oblast, models, h):
    """Per oblast (and country) and origin hour: share of raions with a start vs mean forecast."""
    oblast = pd.Series({u: unit_oblast[u] for u in Y.columns})
    rows = cells.any(axis=1)
    out = []
    for name, frame in [("y", Y)] + [(m, preds[m].reindex_like(Y)) for m in models]:
        v = frame.where(cells).iloc[rows]
        g = v.T.groupby(oblast).mean().T
        g["Україна"] = v.mean(axis=1)
        out.append(g.stack().rename(name))
    df = pd.concat(out, axis=1).rename_axis(["origin", "oblast"]).reset_index()
    return df.assign(h=h)


def evaluate_hourly(full_covered, intervals, coverage, frozen, h, config, text):
    test, validation, folds = make_splits(full_covered, h, frozen=frozen)
    cfg = config["horizons"][str(h)]
    start = folds[0].train_end(h) - pd.Timedelta(days=max(TRAIN_WINDOWS_DAYS) + WARMUP_DAYS)  # as in the backtest
    panel = build_panel(intervals, coverage, start=start)
    unit_oblast = coverage["unit_oblast"]
    Y = labels(panel, h)
    F = build_features(panel, unit_oblast, Y, h, history_features(panel, unit_oblast))
    feature_sets = {"": F}
    if text is not None:
        feature_sets["_text"] = F | text_features(text[0], panel, unit_oblast, text[1])

    spec = LGBMSpec()
    preds = predict_all(Y, h)
    rows = test.mask(Y.index)
    models = []
    for suffix, Fs in feature_sets.items():
        X = stack(Fs, rows)
        for kind in MODEL_KINDS:
            m = f"{kind}{suffix}"
            model = fit(m, cfg[m], Fs, Y, test, h, spec)
            preds[m] = to_wide(predict(model, X), X.index, Y)
            models.append(m)

    known = ~np.isnan(Y.to_numpy()) & rows[:, None]
    ok_full = {m: defined(p, Y) for m, p in preds.items()}
    cells = known & np.logical_and.reduce([ok_full[m] for m in models + [REFERENCE]])
    print(f"h={h}: test origins {test.start} .. {test.end}, models {models}; scored "
          f"{cells.sum()} of {known.sum()} labelled cells", flush=True)
    assert cells.any() and not (cells & ~rows[:, None]).any()
    assert cells.sum() >= 0.99 * known.sum(), "models / reference undefined on part of the test"

    scores = score_hourly(Y, preds, cells, ok_full, h)
    rel = pd.concat([calibration_table(*paired(Y, preds[m], cells)).reset_index().assign(h=h, model=m)
                     for m in [REFERENCE] + models], ignore_index=True)
    errors = pd.concat([breakdown(Y, preds, cells, unit_oblast, h, m) for m in models]
                       + [by_period(Y, preds, cells, models, h)], ignore_index=True)

    day, _ = kyiv_day_and_hour(Y.index, cells.shape)
    y = Y.to_numpy()[cells]
    loss = {m: np.nan_to_num((p.reindex_like(Y).to_numpy()[cells] - y) ** 2) for m, p in preds.items()}
    ok = {m: ok_full[m][cells] for m in preds}
    comparisons = [(m, REFERENCE) for m in preds if m != REFERENCE] + [
        ("lgbm", "logit"), ("lgbm_text", "lgbm"), ("logit_text", "logit")]
    boot = bootstrap_rows(loss, day[cells], ok, comparisons, target="A", h=h, metric="brier")
    series = oblast_hourly(Y, preds, cells, unit_oblast, ["lgbm", REFERENCE], h)
    return {"scores": scores, "reliability": rel, "errors": errors, "bootstrap": boot, "series": series,
            "window": [str(test.start), str(test.end)], "scored_cells": [int(cells.sum()), int(known.sum())]}


# ---------------------------------------------------------------------------
# Target B
# ---------------------------------------------------------------------------

def wide(long: dict[str, pd.Series], Y: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: to_wide(s.to_numpy(), s.index, Y) for name, s in long.items()}


def score_daily(Y, preds, scored, ok, ref, **key):
    """Point scores on `scored` cells where the forecast and `ref` are defined; rel_mae = MAE / MAE(ref)."""
    rows = []
    for name, p in preds.items():
        c = scored & ok[name] & ok[ref]
        y, pv = paired(Y, p, c)
        y_ref, p_ref = paired(Y, preds[ref], c)
        out = regression_scores(y, pv, float(np.mean(np.abs(p_ref - y_ref))))
        out["rel_mae"] = out.pop("mase")
        rows.append({**key, "model": name, "ref": ref, **out, "n_days": int(c.any(axis=1).sum())})
    return rows


def evaluate_daily(intervals, coverage, frozen, config_daily):
    full_daily = build_daily(build_panel(intervals, coverage))
    test, validation, folds = make_daily_splits(full_daily.count.notna().any(axis=1), frozen)
    print(f"target B: test days {test.start.date()} .. {test.end.date()} (exclusive), "
          f"conformal calibration on {validation.start.date()} .. {validation.end.date()}", flush=True)
    start = (folds[0].start - pd.Timedelta(days=max(daily_bt.TRAIN_WINDOWS_DAYS) + daily_bt.WARMUP_DAYS)
             ).tz_localize(KYIV_TZ)  # as in the backtest
    panel = build_panel(intervals, coverage, start=start.tz_convert("UTC"))
    daily = build_daily(panel)
    unit_oblast = coverage["unit_oblast"]
    F = origin_features(daily, panel, unit_oblast, history_features(panel, unit_oblast))

    scores, ivals, boot, series = [], [], [], []
    for target in TARGETS:
        Y = daily.target(target)
        origins = Y.index[(Y.index >= test.start - pd.Timedelta(days=daily_bt.MAX_HORIZON_DAYS))
                          & (Y.index < test.end)]
        hw = holt_winters_daily(Y, origins)
        rows = test.mask(Y.index)
        for k in daily_bt.HORIZONS:
            Fk = for_horizon(F, daily, k)
            cfg = config_daily["targets"][target][str(k)]
            p_test = wide(daily_bt.fit_split(Fk, Y, test, k, cfg), Y)
            p_val = wide(daily_bt.fit_split(Fk, Y, validation, k, cfg), Y)

            preds = {name: fn(Y, k) for name, fn in DAILY_BASELINES.items()}
            preds["poisson_glm"], preds["lgbm"] = p_test["poisson_glm"], p_test["lgbm"]
            preds["holt_winters"] = hw[k]["mean"]
            ok = {m: defined(p, Y) for m, p in preds.items()}
            known = Y.notna().to_numpy() & rows[:, None]
            base = known & ok["lgbm"] & ok["poisson_glm"] & ok[DAILY_REF]
            assert base.sum() >= 0.99 * known.sum(), "models / reference undefined on part of the test"
            common = base & np.logical_and.reduce(list(ok.values()))
            scores += score_daily(Y, preds, base, ok, DAILY_REF, target=target, k=k, cells="all_days")
            scores += score_daily(Y, preds, common, ok, daily_bt.REFERENCE, target=target, k=k, cells="common")

            # Intervals on all test days: raw quantiles, conformal (margins from validation), Holt-Winters 95%
            vcells = Y.notna().to_numpy() & validation.mask(Y.index)[:, None]
            y, yv = Y.to_numpy()[base], Y.to_numpy()[vcells]
            bands = {}
            for level, (lo, hi) in ((80, (0.1, 0.9)), (95, (0.025, 0.975))):
                qlo, qhi = p_test[f"q{lo}"].to_numpy(), p_test[f"q{hi}"].to_numpy()
                vlo, vhi = p_val[f"q{lo}"].to_numpy()[vcells], p_val[f"q{hi}"].to_numpy()[vcells]
                assert not (np.isnan(vlo).any() or np.isnan(vhi).any() or np.isnan(qlo[base]).any())
                margin = conformal_margin(yv, vlo, vhi, level / 100)
                clo, chi = np.maximum(qlo - margin, 0), qhi + margin
                bands[level] = (clo, chi)
                for model, (a, b) in (("lgbm_quantile", (qlo, qhi)), ("lgbm_conformal", (clo, chi))):
                    ivals.append({"target": target, "k": k, "model": model, "level": level,
                                  "margin": margin if model == "lgbm_conformal" else 0.0, "n": int(base.sum()),
                                  **interval_scores(y, a[base], b[base]),
                                  "pinball": np.mean([pinball(y, a[base], lo), pinball(y, b[base], hi)])})
            hcells = base & ok["holt_winters"]
            yh, hlo, hhi = Y.to_numpy()[hcells], hw[k]["lower"].to_numpy()[hcells], hw[k]["upper"].to_numpy()[hcells]
            ivals.append({"target": target, "k": k, "model": "holt_winters", "level": 95, "margin": 0.0,
                          "n": int(hcells.sum()), **interval_scores(yh, hlo, hhi),
                          "pinball": np.mean([pinball(yh, hlo, 0.025), pinball(yh, hhi, 0.975)])})

            # Bootstrap over target days: MAE and Poisson deviance skill
            yb = Y.to_numpy()[base]
            days = np.broadcast_to(Y.index.to_numpy()[:, None], base.shape)[base]
            okb = {m: ok[m][base] for m in preds}
            comparisons = [(m, DAILY_REF) for m in preds if m != DAILY_REF] + [
                (m, daily_bt.REFERENCE) for m in ("lgbm", "poisson_glm", "holt_winters", DAILY_REF)]
            for metric, loss_fn in (("mae", lambda p: np.abs(p - yb)),
                                    ("poisson_dev", lambda p: poisson_deviance_cells(yb, p))):
                loss = {m: np.nan_to_num(loss_fn(np.nan_to_num(preds[m].to_numpy()[base]))) for m in preds}
                boot += bootstrap_rows(loss, days, okb, comparisons, target=target, k=k, metric=metric)

            day_idx, unit_idx = np.nonzero(base)
            series.append(pd.DataFrame({
                "target": target, "k": k, "day": Y.index[day_idx], "unit": Y.columns[unit_idx],
                "oblast": [unit_oblast[u] for u in Y.columns[unit_idx]],
                "y": Y.to_numpy()[day_idx, unit_idx],
                **{m: preds[m].to_numpy()[day_idx, unit_idx] for m in ("lgbm", DAILY_REF, "holt_winters")},
                "lo80": bands[80][0][day_idx, unit_idx], "hi80": bands[80][1][day_idx, unit_idx],
                "lo95": bands[95][0][day_idx, unit_idx], "hi95": bands[95][1][day_idx, unit_idx],
            }))
            print(f"  {target} k={k}: n={int(base.sum())} (common {int(common.sum())})", flush=True)
    return {"scores": pd.DataFrame(scores), "intervals": pd.DataFrame(ivals), "bootstrap": boot,
            "series": pd.concat(series, ignore_index=True), "window": [str(test.start.date()), str(test.end.date())]}


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-run although the test was already scored")
    ap.add_argument("--rehearse", type=Path, metavar="DIR",
                    help="use the validation window instead of the test and write to DIR")
    args = ap.parse_args()
    if args.rehearse and args.rehearse.resolve() == RESULTS_DIR.resolve():
        raise SystemExit("a rehearsal must not write to reports/results")
    out_dir = args.rehearse or RESULTS_DIR
    previous = None
    if not args.rehearse:
        if MARKER.exists() and not args.force:
            raise SystemExit(f"the final test was already opened ({MARKER}); it is scored once (PLAN.md 4.1)")
        dirty = uncommitted_code()
        if dirty:
            raise SystemExit("commit the code first, so the recorded commit is the code that runs:\n"
                             + "\n".join(dirty))
        previous = json.loads(MARKER.read_text(encoding="utf-8")) if MARKER.exists() else None

    intervals, coverage = load()
    frozen = load_frozen_test()
    config_path, config_daily_path = RESULTS_DIR / "model_config.json", RESULTS_DIR / "model_config_daily.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_daily = json.loads(config_daily_path.read_text(encoding="utf-8"))
    text_ok = check_inputs(coverage, frozen, config, config_daily)
    outages = test_period_outages(intervals, frozen)
    if outages:
        raise SystemExit(f"feed outage(s) in the test window: {outages}")

    full_covered = build_panel(intervals, coverage).covered
    if args.rehearse:
        _, validation, _ = make_splits(full_covered, 1, frozen=frozen)
        frozen = frozen | {"test_start": str(validation.start), "test_end": str(validation.end)}
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"REHEARSAL on the validation window {validation.start} .. {validation.end} -> {out_dir}")

    marker = {
        "status": "opened",
        "rehearsal": bool(args.rehearse),
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "data_cutoff": coverage["data_cutoff"],
        "snapshots": frozen["snapshots"],
        "test_window_utc": [frozen["test_start"], frozen["test_end"]],
        "sha256": {"model_config.json": sha256(config_path), "model_config_daily.json": sha256(config_daily_path)},
        "text_ablation": text_ok,
        "coverage_windows_sha256": windows_fingerprint(coverage),
        "test_period_outages": outages,
        "reruns": [],
    }
    if text_ok:
        marker["sha256"] |= {"telegram_snapshot": sha256(TELEGRAM_DIR / "kpszsu" / config["text"]["snapshot"]),
                             "src/features/text.py": sha256(ROOT / "src" / "features" / "text.py"),
                             "src/data/telegram.py": sha256(ROOT / "src" / "data" / "telegram.py")}
    if previous:
        archive = RESULTS_DIR / "archive" / previous["scored_at"].replace(":", "")
        archive.mkdir(parents=True, exist_ok=True)
        for f in RESULTS_DIR.glob("final_test*"):
            shutil.move(str(f), archive / f.name)
        marker["reruns"] = previous.get("reruns", []) + [{"scored_at": previous["scored_at"],
                                                          "status": previous.get("status"),
                                                          "archived_to": str(archive.relative_to(ROOT))}]
        print(f"re-run: earlier outputs archived to {archive}")
    marker_path = out_dir / MARKER.name
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")  # opened, before anything is scored

    text = load_text(config) if text_ok else None
    hourly = [evaluate_hourly(full_covered, intervals, coverage, frozen, h, config, text) for h in HORIZONS]
    daily = evaluate_daily(intervals, coverage, frozen, config_daily)

    def save_csv(frame, name):
        frame.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.4f")

    save_csv(pd.concat([o["scores"] for o in hourly], ignore_index=True), "final_test")
    save_csv(pd.concat([o["reliability"] for o in hourly], ignore_index=True), "final_test_reliability")
    save_csv(pd.concat([o["errors"] for o in hourly], ignore_index=True), "final_test_errors")
    save_csv(pd.DataFrame([r for o in hourly for r in o["bootstrap"]] + daily["bootstrap"]), "final_test_bootstrap")
    save_csv(daily["scores"], "final_test_daily")
    save_csv(daily["intervals"], "final_test_intervals_daily")
    s = pd.concat([o["series"] for o in hourly], ignore_index=True)
    s.astype({c: "float32" for c in ("y", "lgbm", REFERENCE)}).to_parquet(
        out_dir / "final_test_hourly.parquet", index=False)
    d = daily["series"]
    d.astype({c: "float32" for c in d.columns if d[c].dtype == "float64"}).to_parquet(
        out_dir / "final_test_daily.parquet", index=False)

    marker |= {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origins_by_h": {str(h): o["window"] for h, o in zip(HORIZONS, hourly)},
        "scored_cells_by_h": {str(h): o["scored_cells"] for h, o in zip(HORIZONS, hourly)},
        "test_days_kyiv": daily["window"],
    }
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")

    res = pd.concat([o["scores"] for o in hourly], ignore_index=True)
    cols = ["h", "model", "n", "n_days", "base_rate", "brier", "bss", "log_loss", "roc_auc", "roc_auc_within",
            "pr_auc"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(res[cols].to_string(index=False))
        ds = daily["scores"]
        for (target, cells), g in ds.groupby(["target", "cells"]):
            print(f"\n{target} / {cells}: MAE relative to {g['ref'].iloc[0]} by k")
            print(g.pivot(index="model", columns="k", values="rel_mae").to_string())
        print(daily["intervals"].pivot_table(index=["target", "model", "level"], columns="k",
                                             values="coverage").to_string())
    print(f"\nsaved {out_dir}/final_test*")


if __name__ == "__main__":
    main()
