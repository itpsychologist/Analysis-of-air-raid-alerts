"""build_report.py on a tiny synthetic results set (PLAN.md 6.3): valid, offline HTML."""

import json
import re
from html.parser import HTMLParser

import numpy as np
import pandas as pd

from scripts.build_report import build_html
from src.viz.results import load_results

A_MODELS = ["seasonal_naive", "recent_rate_24h", "unit_rate_28d", "hour_of_day_rate_28d",
            "logit", "lgbm", "logit_text", "lgbm_text"]
B_MODELS = ["seasonal_naive", "last_day", "mean_7d", "mean_28d", "poisson_glm", "lgbm", "holt_winters"]
OBLASTS = ["Харківська область", "Львівська область"]
UNITS = {"Харківський район": OBLASTS[0], "Львівський район": OBLASTS[1]}


def write_fixture(d):
    rng = np.random.default_rng(0)
    r = lambda: float(rng.uniform(0.05, 0.9))  # noqa: E731
    pd.DataFrame([{"h": h, "split": "test", "model": m, "n": 1000, "base_rate": r(), "brier": r(), "log_loss": r(),
                   "roc_auc": r(), "pr_auc": r(), "bss": r() - 0.5, "roc_auc_within": r(), "n_days": 28,
                   "first_origin": "2026-09-13 10:00:00+00:00"}
                  for h in (1, 3, 6) for m in A_MODELS]).to_csv(d / "final_test.csv", index=False)
    pd.DataFrame([{"h": h, "split": s, "model": m, "bss": r()} for h in (1, 3, 6) for m in A_MODELS
                  for s in ("folds", "validation")]).to_csv(d / "backtest.csv", index=False)
    pd.DataFrame([{"bin": b, "mean_forecast": b / 10 + 0.05, "observed": r(), "n": 50, "h": h, "model": m}
                  for h in (1, 3, 6) for m in ("unit_rate_28d", "lgbm", "logit") for b in range(10)]
                 ).to_csv(d / "final_test_reliability.csv", index=False)
    pd.DataFrame([{"group": g, "n": 10, "base_rate": r(), "brier": r(), "brier_ref": r(), "bss": r() - 0.5,
                   "h": h, "model": m, "by": by} for h in (1, 3, 6) for m in ("lgbm", "logit")
                  for by, groups in (("oblast", OBLASTS), ("kyiv_hour", range(24)), ("period", ["day", "night"]))
                  for g in groups]).to_csv(d / "final_test_errors.csv", index=False)
    boot = [{"target": "A", "h": h, "metric": "brier", "model": m, "vs": vs, "block_days": bd, "skill": 0.1,
             "ci_low": 0.05, "ci_high": 0.15, "share_not_better": 0.0, "n_blocks": 28}
            for h in (1, 3, 6) for bd in (1, 3)
            for m, vs in [(m, "unit_rate_28d") for m in A_MODELS if m != "unit_rate_28d"] + [("lgbm_text", "lgbm")]]
    boot += [{"target": t, "k": k, "metric": met, "model": m, "vs": "mean_28d", "block_days": 1, "skill": -0.02,
              "ci_low": -0.05, "ci_high": 0.01, "share_not_better": 0.8, "n_blocks": 27}
             for t in ("count", "duration") for k in range(1, 8) for met in ("mae", "poisson_dev")
             for m in ("lgbm", "holt_winters")]
    pd.DataFrame(boot).to_csv(d / "final_test_bootstrap.csv", index=False)
    pd.DataFrame([{"target": t, "k": k, "cells": c, "model": m, "ref": ref, "n": 100, "mean_y": 1.0, "mae": r(),
                   "rmse": r(), "bias": r() - 0.5, "poisson_dev": r(), "rel_mae": r() + 0.5, "n_days": 27}
                  for t in ("count", "duration") for k in range(1, 8) for m in B_MODELS
                  for c, ref in (("all_days", "mean_28d"), ("common", "seasonal_naive"))]
                 ).to_csv(d / "final_test_daily.csv", index=False)
    pd.DataFrame([{"target": t, "k": k, "cells": "folds", "model": m, "mase": r()} for t in ("count", "duration")
                  for k in range(1, 8) for m in B_MODELS]).to_csv(d / "backtest_daily.csv", index=False)
    pd.DataFrame([{"target": t, "k": k, "model": m, "level": lv, "margin": 0.1, "n": 100, "coverage": r(),
                   "width": r(), "pinball": r()} for t in ("count", "duration") for k in range(1, 8)
                  for m, lv in (("lgbm_quantile", 80), ("lgbm_quantile", 95), ("lgbm_conformal", 80),
                                ("lgbm_conformal", 95), ("holt_winters", 95))]
                 ).to_csv(d / "final_test_intervals_daily.csv", index=False)
    origins = pd.date_range("2026-09-13 10:00", periods=48, freq="h", tz="UTC")
    pd.DataFrame([{"origin": o, "oblast": ob, "y": r(), "lgbm": r(), "unit_rate_28d": r(), "h": h}
                  for h in (1, 3, 6) for o in origins for ob in OBLASTS + ["Україна"]]
                 ).to_parquet(d / "final_test_hourly.parquet", index=False)
    days = pd.date_range("2026-09-14", periods=5, freq="D")
    pd.DataFrame([{"target": t, "k": k, "day": day, "unit": u, "oblast": ob, "y": 2.0, "lgbm": r(), "mean_28d": r(),
                   "holt_winters": r(), "lo80": 0.5, "hi80": 3.0, "lo95": 0.0, "hi95": 4.0}
                  for t in ("count", "duration") for k in range(1, 8) for day in days for u, ob in UNITS.items()]
                 ).to_parquet(d / "final_test_daily.parquet", index=False)
    (d / "final_test.json").write_text(json.dumps({
        "status": "complete", "scored_at": "2026-10-11T18:00:00+00:00", "git_commit": "0123456789abcdef",
        "data_cutoff": "2026-10-11T10:06:46+00:00", "snapshots": {"official_data_uk": "a.csv.gz", "alerts_in_ua": "b"},
        "test_window_utc": ["2026-09-13T10:00:00+00:00", "2026-10-11T10:00:00+00:00"],
        "test_days_kyiv": ["2026-09-14", "2026-10-11"]}))


class _Checker(HTMLParser):
    VOID = {"meta", "br", "img", "input", "link", "hr"}

    def __init__(self):
        super().__init__()
        self.stack, self.errors, self.scripts = [], [], []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.scripts.append(dict(attrs))
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}> (open: {self.stack[-3:]})")
        else:
            self.stack.pop()


def test_report_is_valid_self_contained_html(tmp_path):
    write_fixture(tmp_path)
    res = load_results(tmp_path)
    assert res is not None and res.overview == {}  # overview tables are optional
    page = build_html(res, "<script>/* plotly.js inlined here */</script>", "11 Oct 2026")

    checker = _Checker()
    checker.feed(page)
    assert not checker.errors and checker.stack == []
    assert page.startswith("<!doctype html>") and "<title>" in page
    # Offline: no external scripts or stylesheets
    assert all("src" not in s for s in checker.scripts)
    assert not re.search(r"<(script|link)[^>]+(src|href)=[\"']?https?:", page)
    # Every chart made it in, and the disclaimer is there
    assert page.count('class="chart"') >= 12
    assert "does not replace" in page and "official" in page


def test_load_results_is_none_before_the_test_is_scored(tmp_path):
    assert load_results(tmp_path) is None
