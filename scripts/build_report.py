"""CLI: results -> reports/index.html, a self-contained static report (PLAN.md, section 6).

Reads only the committed results in reports/results/ (the final-test run of
scripts/run_final_test.py and the backtests); it never calls the API or fits a model.
The data-overview tables are recomputed from data/processed when it exists and saved
next to the results, so the report can be rebuilt from a fresh clone without data.

Plotly.js is inlined once, so the page works offline (~5 MB); `--cdn` loads it from
a CDN instead (small file, needs internet).

Usage:
    .venv/bin/python scripts/build_report.py [--cdn] [--out reports/index.html]
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.viz import charts  # noqa: E402
from src.viz.results import RESULTS_DIR, Results, compute_overview, load_results, save_overview  # noqa: E402

OUT = ROOT / "reports" / "index.html"
PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.32.0/plotly.min.js"
REFERENCE = "unit_rate_28d"
A_MODELS = ["lgbm", "logit", "lgbm_text", "logit_text", "unit_rate_28d", "hour_of_day_rate_28d",
            "recent_rate_24h", "seasonal_naive"]
B_MODELS = ["lgbm", "poisson_glm", "holt_winters", "mean_28d", "mean_7d", "last_day", "seasonal_naive"]
GITHUB = "https://github.com/itpsychologist/Analysis-of-air-raid-alerts"


# ---------------------------------------------------------------------------
# Tables and text
# ---------------------------------------------------------------------------

def fmt(x, digits=3, signed=False):
    if pd.isna(x):
        return "–"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def table(df: pd.DataFrame, best: dict[str, str] | None = None, cls: str = "") -> str:
    """HTML table; `best` maps a column to "min"/"max" and bolds the best value per column."""
    best = best or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    marks = {}
    for col, how in best.items():
        vals = pd.to_numeric(df[col].astype(str).str.replace("+", "", regex=False).str.split(" ").str[0],
                             errors="coerce")
        if vals.notna().any():
            marks[col] = vals.idxmin() if how == "min" else vals.idxmax()
    rows = []
    for i, r in df.iterrows():
        cells = "".join(
            f"<td{' class=best' if marks.get(c) == i else ''}>{html.escape(str(v))}</td>" for c, v in r.items())
        rows.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{head}</tr></thead>' \
           f'<tbody>{"".join(rows)}</tbody></table></div>'


def table_a(res: Results, h: int) -> str:
    s = res.scores[res.scores["h"] == h].set_index("model")
    bt = res.backtest[res.backtest["h"] == h].pivot(index="model", columns="split", values="bss")
    boot = res.bootstrap[(res.bootstrap["target"] == "A") & (res.bootstrap["h"] == h)
                         & (res.bootstrap["vs"] == REFERENCE) & (res.bootstrap["block_days"] == 1)].set_index("model")
    rows = []
    for m in [m for m in A_MODELS if m in s.index]:
        ci = f" [{fmt(boot.at[m, 'ci_low'], signed=True)}, {fmt(boot.at[m, 'ci_high'], signed=True)}]" \
            if m in boot.index else ""
        rows.append({
            "Model": charts.model_label(m),
            "Cells": f"{int(s.at[m, 'n']):,}",
            "Brier": fmt(s.at[m, "brier"], 4),
            f"BSS vs raion rate [95% CI]": fmt(s.at[m, "bss"], signed=True) + ci,
            "ROC-AUC": fmt(s.at[m, "roc_auc"]),
            "AUC within raion": fmt(s.at[m, "roc_auc_within"]),
            "PR-AUC": fmt(s.at[m, "pr_auc"]),
            "Log loss": fmt(s.at[m, "log_loss"]),
            "BSS folds / validation": f"{fmt(bt.at[m, 'folds'], signed=True)} / "
                                      f"{fmt(bt.at[m, 'validation'], signed=True)}" if m in bt.index else "–",
        })
    return table(pd.DataFrame(rows), best={"Brier": "min", "BSS vs raion rate [95% CI]": "max", "ROC-AUC": "max",
                                           "AUC within raion": "max", "PR-AUC": "max", "Log loss": "min"})


def table_b(res: Results, target: str) -> str:
    d = res.daily[(res.daily["target"] == target) & (res.daily["cells"] == "common")]
    rows = []
    for m in [m for m in B_MODELS if m in set(d["model"])]:
        g = d[d["model"] == m].set_index("k")
        rows.append({"Model": charts.model_label(m),
                     **{f"MASE k={k}": fmt(g.at[k, "rel_mae"]) for k in (1, 3, 7)},
                     "MAE k=1": fmt(g.at[1, "mae"], 2), "Poisson dev. k=1": fmt(g.at[1, "poisson_dev"]),
                     "Bias k=1": fmt(g.at[1, "bias"], 2, signed=True)})
    df = pd.DataFrame(rows)
    return table(df, best={c: "min" for c in df.columns if c.startswith(("MASE", "MAE", "Poisson"))})


def skill(res: Results, model: str, h: int, vs: str = REFERENCE) -> pd.Series | None:
    b = res.bootstrap[(res.bootstrap["target"] == "A") & (res.bootstrap["h"] == h)
                      & (res.bootstrap["model"] == model) & (res.bootstrap["vs"] == vs)
                      & (res.bootstrap["block_days"] == 1)]
    return None if b.empty else b.iloc[0]


def daily_skill(res: Results, target: str, k: int, model: str, vs: str, metric: str) -> pd.Series | None:
    b = res.bootstrap[(res.bootstrap["target"] == target) & (res.bootstrap["k"] == k)
                      & (res.bootstrap["model"] == model) & (res.bootstrap["vs"] == vs)
                      & (res.bootstrap["metric"] == metric) & (res.bootstrap["block_days"] == 1)]
    return None if b.empty else b.iloc[0]


def ci_text(b: pd.Series | None) -> str:
    if b is None:
        return "–"
    verdict = "significant" if b["ci_low"] > 0 else ("significantly worse" if b["ci_high"] < 0 else "not significant")
    return f"{b['skill']:+.3f} (95% CI {b['ci_low']:+.3f} … {b['ci_high']:+.3f}, {verdict})"


def kpis(res: Results) -> str:
    s = res.scores.set_index(["h", "model"])
    cards = []
    for h in (1, 3, 6):
        b = skill(res, "lgbm", h)
        cards.append(
            f'<div class="kpi"><div class="kpi-label">Alert within {h} h · LightGBM</div>'
            f'<div class="kpi-value">{b["skill"]:+.3f}</div>'
            f'<div class="kpi-sub">Brier skill vs raion rate<br>95% CI {b["ci_low"]:+.3f} … {b["ci_high"]:+.3f}'
            f'<br>AUC within raion {s.at[(h, "lgbm"), "roc_auc_within"]:.2f} '
            f'(baseline {s.at[(h, REFERENCE), "roc_auc_within"]:.2f})</div></div>')
    d = res.daily[(res.daily["target"] == "count") & (res.daily["cells"] == "common")].set_index(["model", "k"])
    cards.append(
        f'<div class="kpi"><div class="kpi-label">Daily alert count · 1 day ahead</div>'
        f'<div class="kpi-value">{d.at[("lgbm", 1), "rel_mae"]:.2f}</div>'
        f'<div class="kpi-sub">MASE of LightGBM (1 = same day last week)<br>'
        f'28-day mean {d.at[("mean_28d", 1), "rel_mae"]:.2f} · Holt-Winters {d.at[("holt_winters", 1), "rel_mae"]:.2f}'
        f'</div></div>')
    return f'<div class="kpis">{"".join(cards)}</div>'


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#ffffff;--fg:#1b1f24;--muted:#5b6470;--card:#f5f6f8;--border:#e2e5ea;--accent:#d93636;
--best:rgba(217,54,54,.10);--warn-bg:#fff6e5;--warn-border:#f0b44c}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0e1117;--fg:#e6e8eb;--muted:#9aa3ad;
--card:#161b25;--border:#262d3a;--accent:#ff5c5c;--best:rgba(255,92,92,.16);--warn-bg:#2a2210;--warn-border:#a07020}}
:root[data-theme="dark"]{--bg:#0e1117;--fg:#e6e8eb;--muted:#9aa3ad;--card:#161b25;--border:#262d3a;
--accent:#ff5c5c;--best:rgba(255,92,92,.16);--warn-bg:#2a2210;--warn-border:#a07020}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 Inter,system-ui,-apple-system,"Segoe UI",
Roboto,sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 64px}
header.top{border-bottom:3px solid var(--accent);padding-bottom:16px;margin-bottom:8px}
h1{font-size:1.9rem;line-height:1.2;margin:0 0 8px}
h2{font-size:1.35rem;margin:44px 0 8px;padding-top:8px;border-top:1px solid var(--border)}
h3{font-size:1.05rem;margin:24px 0 6px}
p,li{max-width:76ch}
.meta{color:var(--muted);font-size:.9rem;display:flex;flex-wrap:wrap;gap:4px 20px}
.meta b{color:var(--fg);font-weight:600}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:.9rem;margin:12px 0 0}
nav a,a{color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.kpi-label{font-size:.82rem;color:var(--muted)}
.kpi-value{font-size:1.8rem;font-weight:700;color:var(--accent);font-variant-numeric:tabular-nums}
.kpi-sub{font-size:.8rem;color:var(--muted);line-height:1.4}
.chart{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:8px;margin:14px 0;
min-width:0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.table-wrap{overflow-x:auto;margin:10px 0;border:1px solid var(--border);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{background:var(--card);font-weight:600}
td.best{background:var(--best);font-weight:700}
.note{background:var(--warn-bg);border-left:4px solid var(--warn-border);padding:10px 14px;border-radius:0 8px 8px 0;
margin:16px 0}
.muted{color:var(--muted);font-size:.88rem}
button.theme{position:fixed;top:12px;right:12px;background:var(--card);color:var(--fg);border:1px solid
var(--border);border-radius:8px;padding:6px 10px;cursor:pointer;font:inherit;font-size:.85rem;z-index:10}
footer{margin-top:48px;color:var(--muted);font-size:.85rem;border-top:1px solid var(--border);padding-top:12px}
"""

THEME_JS = """
(function(){
  var root=document.documentElement, btn=document.getElementById('theme-btn');
  var LIGHT=%(light)s, DARK=%(dark)s;
  function isDark(){var t=root.getAttribute('data-theme');
    return t ? t==='dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;}
  function apply(){
    var dark=isDark(), grid=dark?'rgba(255,255,255,0.08)':'rgba(0,0,0,0.08)';
    btn.textContent=dark?'Light theme':'Dark theme';
    document.querySelectorAll('.js-plotly-plot').forEach(function(gd){
      var upd={template:dark?DARK:LIGHT};
      Object.keys(gd.layout).forEach(function(k){if(/^[xy]axis\\d*$/.test(k)){
        upd[k+'.gridcolor']=grid; upd[k+'.zerolinecolor']=grid;}});
      Plotly.relayout(gd,upd);
    });
  }
  btn.addEventListener('click',function(){root.setAttribute('data-theme',isDark()?'light':'dark');apply();});
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',apply);
  window.addEventListener('load',apply);
})();
"""


def fig_html(fig, div_id: str) -> str:
    fig.update_layout(autosize=True)
    inner = fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                        config={"displaylogo": False, "responsive": True})
    return f'<div class="chart">{inner}</div>'


def build_html(res: Results, plotly_js: str, generated_at: str) -> str:
    import plotly.io as pio

    m, ov = res.marker, res.overview
    t0, t1 = (pd.Timestamp(x) for x in m["test_window_utc"])
    cutoff = pd.Timestamp(m["data_cutoff"])
    days = m["test_days_kyiv"]
    F = lambda fig, i: fig_html(fig, i)  # noqa: E731
    th = "light"
    text_note = ""
    tb = skill(res, "lgbm_text", 1, vs="lgbm")
    if tb is not None:
        text_note = (f"Adding the Air Force channel's posts to LightGBM changes Brier by "
                     f"{ci_text(tb)} relative to LightGBM without text at h = 1 h")
        for h in (3, 6):
            b = skill(res, "lgbm_text", h, vs="lgbm")
            text_note += f"; {b['skill']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] at h = {h} h"
        text_note += " (positive = text helps)."

    hw = daily_skill(res, "count", 1, "holt_winters", "mean_28d", "mae")
    lg = daily_skill(res, "count", 1, "lgbm", "mean_28d", "mae")
    lgd = daily_skill(res, "count", 1, "lgbm", "mean_28d", "poisson_dev")
    iv = res.intervals[(res.intervals["target"] == "count") & (res.intervals["k"] == 1)].set_index(["model", "level"])

    overview = ""
    if ov:
        overview = f"""
<h2 id="data">1. Data</h2>
<p>Alert intervals per raion (118 raions + Kyiv City) from the
<a href="https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset">official sirens dataset</a> (2022 → 06 Sep 2026)
and the <a href="https://alerts.in.ua">alerts.in.ua</a> API (12 Sep → 11 Oct 2026), stitched by time and never mixed.
Oblast-level alerts (the norm until December 2025) are spread to every raion of the oblast;
Luhansk oblast (permanent alert) is excluded.</p>
{F(charts.make_history_chart(ov["daily"], th), "history")}
<div class="grid2">
{F(charts.make_hour_dow_heatmap(ov["heatmap"], th, "Alert starts by hour (Kyiv) and weekday, last 365 days"), "heat")}
{F(charts.make_oblast_share_chart(ov["oblast"], th, "Share of time under alert, last 365 days"), "oblast-share")}
</div>"""

    body = f"""
<header class="top">
<h1>Forecasting Ukrainian air-raid alerts: final test</h1>
<p>How well can the start of an air-raid alert in a raion be predicted 1–6 hours ahead, and
the daily number of alerts 1–7 days ahead, using only information available at the time of the forecast?</p>
<div class="meta">
<span>Data cutoff <b>{cutoff:%d %b %Y %H:%M} UTC</b></span>
<span>Final test <b>{t0:%d %b} – {t1:%d %b %Y}</b> (forecast origins, UTC; Kyiv days {days[0]} … {days[1]}, end exclusive)</span>
<span>Scored once on <b>{pd.Timestamp(m["scored_at"]):%d %b %Y}</b></span>
<span>Report generated {generated_at}</span>
</div>
<nav><a href="#data">Data</a><a href="#method">Method</a><a href="#results">Results</a>
<a href="#fact">Forecast vs actual</a><a href="#calibration">Calibration</a><a href="#oblasts">By oblast</a>
<a href="#daily">Daily forecasts</a><a href="#limits">Limitations</a></nav>
</header>
<div class="note"><b>Disclaimer.</b> This is an analytical forecast from public data. It does not replace
official alert signals. Always follow the official air-raid warning.</div>
{kpis(res)}
{overview}
<h2 id="method">2. Method: forecasts without leakage</h2>
<p>A forecast is made at an hourly origin <i>t</i> for every raion. Every feature uses only events that started
before <i>t</i>; an alert still running at <i>t</i> counts as "active now" with its duration cut at <i>t</i>; the label of
an earlier origin <i>t′</i> is used only once it is known (<i>t′ + h ≤ t</i>). Splits are chronological, with an
embargo of <i>h</i> hours between training and test. The final test (the last 28 days of the snapshot)
was frozen when the evaluation framework was built. The training window, tree settings and the
number of trees were chosen on the validation period and frozen in <code>model_config.json</code>. The test was
then scored exactly once. Every feature has a "perturb the future" test: rewriting all events after <i>t</i> must
not change the features at <i>t</i>.</p>
{F(charts.make_split_chart(ov["splits"], th), "splits") if ov else ""}
<ul>
<li><b>Target A</b>: probability that an alert starts in the raion within the next <i>h</i> = 1, 3, 6 hours.
Models: logistic regression and LightGBM on raion history, same-oblast, neighbouring-oblast and country activity,
the Kyiv calendar and past label rates. Baselines: raion rate over 28 days (the reference for the Brier skill score),
same-hour rate, last-24-hour rate, same hour last week.</li>
<li><b>Target B</b>: number of alerts and hours under alert per raion and Kyiv day, 1–7 days ahead.
Models: Poisson GLM, LightGBM (Poisson), and the dashboard's Holt-Winters refit at every origin. Baselines:
28/7-day means, last day, same day last week (MASE denominator). The LightGBM quantile intervals are widened by
split-conformal margins fit on the validation period.</li>
<li><b>Significance</b>: 95% intervals from a bootstrap that resamples whole Kyiv days (hours within a day are
strongly dependent).</li>
</ul>

<h2 id="results">3. Results: will an alert start within <i>h</i> hours?</h2>
<p>Bold marks the best value in each column. "AUC within raion" averages the ROC-AUC computed separately for each
raion. It measures whether a model gets the <i>timing</i> right. The pooled ROC-AUC is dominated by the
frontline-vs-west difference, which any per-raion rate already captures. The last column repeats the
backtest numbers (6 weekly folds / validation). Validation is optimistic because it was used for selection.</p>
{F(charts.make_skill_chart(res.bootstrap, REFERENCE, th), "skill")}
{"".join(f"<h3>h = {h} h</h3>" + table_a(res, h) for h in (1, 3, 6))}
<p>{text_note} The text tagging rules were frozen before the test (sha256 of <code>text.py</code> in
<code>model_config.json</code>). The rules were first drafted on posts from inside the test window (texts only, no alert
labels), so any optimism would favour the text models.</p>

<h2 id="fact">4. Forecast vs actual on the test period</h2>
<p>Hourly: the share of an oblast's raions where an alert started within the next 3 hours, against the mean
forecast probability over those raions. Use the menu to pick an oblast.</p>
{F(charts.make_hourly_fact_chart(res.hourly_series, 3, th), "fact-hourly")}
<p>Daily: alerts per day in a raion, forecast one day ahead, with the conformal 80% and 95% bands.</p>
{F(charts.make_daily_fact_chart(res.daily_series, "count", 1, th), "fact-daily")}

<h2 id="calibration">5. Calibration</h2>
<p>A calibrated forecast lies on the diagonal: of all cases forecast at 30%, about 30% see an alert start.
Marker size grows with the number of cases.</p>
<div class="grid3">{"".join(F(charts.make_reliability_chart(res.reliability, h, th), f"rel{h}") for h in (1, 3, 6))}</div>

<h2 id="oblasts">6. Where the model helps</h2>
<div class="grid2">
{F(charts.make_oblast_skill_chart(res.errors, 1, "lgbm", th), "oblast-skill")}
{F(charts.make_hour_skill_chart(res.errors, "lgbm", th), "hour-skill")}
</div>

<h2 id="daily">7. Daily alert count and duration, 1–7 days ahead</h2>
<p>MASE below 1 means better than "same day last week". Scored on the cells where every model, including
Holt-Winters, has a forecast. Holt-Winters needs 3 covered days before the origin, so the first test days are
missing. At one day ahead, LightGBM vs the 28-day mean: MAE skill {ci_text(lg)}, Poisson deviance skill
{ci_text(lgd)}; Holt-Winters vs the 28-day mean: MAE skill {ci_text(hw)}.</p>
<div class="grid2">
{F(charts.make_mase_chart(res.daily, "count", "rel_mae", "common", th), "mase-count")}
{F(charts.make_mase_chart(res.daily, "duration", "rel_mae", "common", th), "mase-duration")}
</div>
<h3>Alert count</h3>{table_b(res, "count")}
<h3>Hours under alert</h3>{table_b(res, "duration")}
<div class="grid2">
{F(charts.make_coverage_chart(res.intervals, "count", th), "cov-count")}
{F(charts.make_coverage_chart(res.intervals, "duration", th), "cov-duration")}
</div>
<p class="muted">Alert count, k = 1: 80% interval coverage {iv.at[("lgbm_quantile", 80), "coverage"]:.2f} raw →
{iv.at[("lgbm_conformal", 80), "coverage"]:.2f} conformal; 95% interval {iv.at[("lgbm_quantile", 95), "coverage"]:.2f} →
{iv.at[("lgbm_conformal", 95), "coverage"]:.2f}; Holt-Winters 95% interval {iv.at[("holt_winters", 95), "coverage"]:.2f}.</p>

<h2 id="limits">8. Limitations</h2>
<ul>
<li><b>One test month.</b> 28 days of one season. Attack intensity changes month to month (see the folds column),
so the test is one draw, not a guarantee.</li>
<li><b>Regime change.</b> Tactics and the alert system itself change. Raion-level alerts became the norm in
December 2025. Training windows longer than 180 days made the models worse.</li>
<li><b>Data gap.</b> The official dataset ends on 06 Sep 2026 and the API history starts on 12 Sep, so features with
7–28-day windows are partly missing on the first test days.</li>
<li><b>Rare raions.</b> In western oblasts with almost no alerts, the shared model can be worse than the raion's own
rate.</li>
<li><b>Mass attacks</b> are hard to foresee from alert history alone. Public channel posts did not add skill at hourly
resolution.</li>
<li><b>Not an alert service.</b> Forecasts are analytical. Follow official signals.</li>
</ul>
<footer>Code, tests and the full method: <a href="{GITHUB}">{GITHUB.split("//")[1]}</a> · Git commit of the test run
<code>{m["git_commit"][:12]}</code> · Data snapshots {", ".join(m["snapshots"].values())}.</footer>
"""
    light = pio.templates["plotly_white"].to_plotly_json()
    dark = pio.templates["plotly_dark"].to_plotly_json()
    theme_js = THEME_JS % {"light": json.dumps(light), "dark": json.dumps(dark)}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Air-Raid Alert Forecasts</title>
<meta name="description" content="Final-test evaluation of hourly and daily air-raid alert forecasts per raion.">
<style>{CSS}</style>
{plotly_js}
</head><body><button class="theme" id="theme-btn" type="button">Dark theme</button><main>{body}</main>
<script>{theme_js}</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--cdn", action="store_true", help="load plotly.js from a CDN instead of inlining it")
    args = ap.parse_args()

    if (ROOT / "data" / "processed" / "raion_intervals.parquet").exists() and args.results == RESULTS_DIR:
        from src.data.build import load

        save_overview(compute_overview(*load()), args.results)
    res = load_results(args.results)
    if res is None:
        raise SystemExit(f"no final-test results in {args.results}; run scripts/run_final_test.py first")
    if args.cdn:
        plotly_js = f'<script src="{PLOTLY_CDN}"></script>'
    else:
        from plotly.offline import get_plotlyjs

        plotly_js = f"<script>{get_plotlyjs()}</script>"
    page = build_html(res, plotly_js, datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
