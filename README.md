# 🇺🇦 Air Raid Alarms Analytics Dashboard

A **time-series analysis and forecasting dashboard** for Ukrainian air-raid alerts, built with Streamlit, Plotly, and statsmodels.

> **Data source:** [alerts.in.ua](https://alerts.in.ua) — volunteer-maintained real-time alert map  
> **API docs:** [devs.alerts.in.ua](https://devs.alerts.in.ua)

> ⚠️ **Disclaimer.** The forecasts are analytical and made from public data. They **do not replace official
> air-raid alert signals**. Always follow the official warning.

The repository has two parts:

1. **The dashboard** (`app.py`): live and 30-day alert analytics with a short-term forecast.
2. **The forecasting study** (`src/`, `scripts/`, `tests/`): hourly and daily forecasts per raion, evaluated
   without data leakage on the most recent 4 weeks. The results are in a self-contained static report:
   **[itpsychologist.github.io/Analysis-of-air-raid-alerts](https://itpsychologist.github.io/Analysis-of-air-raid-alerts/)**
   (source: [`reports/index.html`](reports/index.html); it also works offline when downloaded).

---

## 🔬 Forecasting study

### Problem

At an hourly forecast origin *t*, for each of **119 units** (118 raions + Kyiv City):

| Target | Question | Horizon | Metrics |
|---|---|---|---|
| **A** (main) | Probability that an alert **starts** in the raion within the next *h* hours | h = 1, 3, 6 h | Brier, Brier skill (BSS) vs the raion's 28-day rate, ROC-AUC **within raion**, PR-AUC, log loss, calibration |
| **B** | Number of alerts and hours under alert per raion and Kyiv day | 1–7 days | MAE, MASE, Poisson deviance, interval coverage |

The study works at raion level, not oblast level. Since December 2025, alerts are mostly declared per raion.
At oblast level the frontline oblasts are under alert 95–100% of the time, so forecasting them is trivial.

### Data (cutoff 2026-10-11 10:06 UTC)

| Period | Source |
|---|---|
| 2022-03-15 → 2026-09-06 | [Official sirens dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), pinned to commit `bf0c323c78e3` |
| 2026-09-06 → 09-12 | no data (gap between sources) |
| 2026-09-12 → 10-11 | alerts.in.ua API, `month_ago` per oblast |
| 2025-10-31 → 2026-10-11 | Air Force posts (t.me/s/kpszsu preview), used for the text ablation |

The sources are stitched by time and never mixed. Oblast-level alerts are spread to every raion of the oblast.
Luhansk oblast (permanent alert) is excluded.

The official feed has an **outage from 29 Aug to 2 Sep 2026**: 105 h without a single alert start anywhere.
The longest real national lull since 2022 is 30 h. Any stretch over 48 h without a start is masked as missing
data (`src/data/build.py`), so it is never counted as "no alerts".

### Method: no leakage

- Features at *t* use only events that started before *t*. An alert still running at *t* is cut at *t*. The
  label of an earlier origin *t′* is used only once it is known (*t′ + h ≤ t*). Calendar features use Kyiv time.
- Splits are chronological only: 6 weekly rolling folds, then a 4-week validation period (1–29 Aug 2026),
  then the **final test, 13 Sep – 11 Oct 2026**. The test window was frozen in
  [`config/evaluation.json`](config/evaluation.json). There is an embargo of *h* hours between training and
  scored periods.
- The training window, tree size and number of trees were selected on validation only and frozen in
  `reports/results/model_config*.json`.
- The test was scored **once** (`scripts/run_final_test.py`). The run checks the frozen configs, the
  data-build fingerprint and that the code is committed. It writes a marker before scoring and refuses to
  run again.
- Every feature has a "perturb the future" test: rewriting all events after *t* must leave the features at
  *t* unchanged (`tests/test_features.py`, `test_leakage.py`, `test_daily.py`, `test_text.py`).
- A separate read-only reviewer audited the features, splits and the final-test script before the test was
  opened.
- Significance: 95% intervals from a bootstrap that resamples whole Kyiv days (1-day and 3-day blocks
  give the same picture).

### Results on the final test (13 Sep – 11 Oct 2026)

**Target A: will an alert start in the raion within *h* hours?** About 80,000 raion-hours per horizon.

| h | Base rate | LightGBM BSS vs raion rate [95% CI] | Logistic reg. BSS | AUC within raion: LightGBM / raion rate | PR-AUC: LightGBM / raion rate | LightGBM BSS in backtest (folds / validation) |
|---|---|---|---|---|---|---|
| 1 h | 0.16 | **+0.113** [+0.098, +0.126] | +0.096 | 0.68 / 0.46 | 0.49 / 0.32 | +0.103 / +0.100 |
| 3 h | 0.37 | **+0.070** [+0.054, +0.086] | +0.048 | 0.64 / 0.46 | 0.74 / 0.67 | +0.076 / +0.065 |
| 6 h | 0.52 | **+0.042** [+0.022, +0.061] | +0.038 | 0.63 / 0.43 | 0.87 / 0.84 | +0.030 / +0.042 |

- **The models beat every baseline at every horizon**, and the test confirms the backtest. No baseline
  (same-hour rate, last-24-hour rate, same hour last week) beats the raion's 28-day rate.
- **Timing skill:** within a raion, the 28-day rate cannot rank hours at all (AUC ≈ 0.45). LightGBM reaches
  0.63–0.68 by using alerts in neighbouring oblasts, the country-wide state and the raion's recent activity.
- LightGBM beats logistic regression at 1 h and 3 h (+0.018 and +0.023 Brier skill, significant). At 6 h they
  are equal.
- The forecasts are close to calibrated: the mean gap between forecast and observed frequency is 1–2
  percentage points. At h = 1, high probabilities are slightly under-forecast (forecast 0.55 → observed
  0.61): the test month was more intense than the training period. Platt or isotonic recalibration did not
  help in the backtest.
- **Where it helps:** the most in Zaporizhzhia and Donetsk (BSS ≈ +0.33…+0.35 at h = 1), then Kyiv oblast,
  Kharkiv, Poltava and Chernihiv. It does **not** help in Zakarpattia, Lviv and Ivano-Frankivsk (base rate
  < 1%, BSS −0.03…−0.04). There, the shared model slightly over-forecasts compared with the raion's own rate.
- **Text did not help.** Adding the Air Force channel posts changes LightGBM's Brier by −0.001 [−0.007, +0.004]
  at 1 h, −0.006 [−0.020, +0.006] at 3 h and 0.000 at 6 h. The posts mostly appear when the alert is declared,
  and the neighbouring-oblast alerts already carry that signal at hourly resolution.

**Target B: daily alerts per raion, 1–7 days ahead.** 27 Kyiv days × 119 raions.

- **The 28-day mean wins.** LightGBM's MAE is 5% worse at k = 1 (significant) and 11% worse at k = 7. The
  Poisson GLM ties the mean at k = 1 and is worse after that.
- **The dashboard's Holt-Winters** is 31% worse than the 28-day mean at k = 1 and 2.1× worse at k = 7. Its
  "95%" interval covers only 54–66% of the outcomes.
- LightGBM quantile intervals widened by split-conformal margins (fit on validation) improve coverage, but they
  still cover less than nominal on the test: 80% intervals cover 70–76% and 95% intervals 82–92% (alert
  count). The test month did not behave like the validation month.

Full tables, charts, forecast-vs-actual by oblast and raion, and calibration are in
[`reports/index.html`](reports/index.html) and in the dashboard's **🧪 Backtest** tab.
The raw numbers are in `reports/results/final_test*.csv`.

### Limitations

- **One test month.** Attack intensity changes month to month.
- **Regime change.** The alert system and attack tactics change. The training window is a tuned setting.
- **Gaps.** "Same hour last week" exists only from 19 Sep because of the data gap. The outage rule is national,
  so an outage in a single oblast would not be detected.
- **Mass attacks** are hard to foresee from alert history alone.
- **Not an alert service.**

### Reproduce

```bash
uv venv --python python3.12 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python scripts/fetch_data.py --api          # raw snapshots (API part ~13 min, rate limit)
.venv/bin/python scripts/fetch_telegram.py --since 2025-11-01   # optional, text ablation (~1.5 h)
.venv/bin/python scripts/build_dataset.py             # data/raw -> data/processed
.venv/bin/python scripts/run_backtest.py              # target A: model selection + folds/validation (~15 min)
.venv/bin/python scripts/run_backtest_daily.py        # target B (~10 min)
.venv/bin/python scripts/run_final_test.py --rehearse /tmp/rehearsal   # safe dry run on validation
.venv/bin/python scripts/build_report.py              # reports/results -> reports/index.html
.venv/bin/python -m pytest                            # 111 tests, incl. leakage tests
```

The whole project was built with an AI assistant (Claude Code); the conversation log is in
[`AI_CONVERSATION_LOG.md`](AI_CONVERSATION_LOG.md) (`scripts/build_conversation_log.py` regenerates it).

A new data snapshot does not move the frozen test window. `run_final_test.py` refuses to run once
`reports/results/final_test.json` exists.

---

## 📸 Features

| Feature | Details |
|---|---|
| **Live Active Alerts** | Real-time alert count fetched every 5 minutes |
| **30-Day Historical Data** | Per-oblast history via the regional history endpoint |
| **KPI Metrics Panel** | Raion alerts, avg/max duration, affected oblasts, raions under alert now; delta vs the previous window when it is covered by the history |
| **Daily Trend Chart** | Raion alerts per day: bar + 7-day MA + OLS trendline |
| **Diurnal Heatmap** | Hour of Day × Day of Week of raion-alert starts (Kyiv time) |
| **Alert Type Breakdown** | Horizontal bar by type with avg duration |
| **Vulnerability Ranking** | Per raion, so large oblasts are not favoured: 60% alerts per raion + 40% share of time under alert |
| **7-Day Forecast** | Holt-Winters (primary) or Ridge Regression (fallback) with 95% CI ribbon |
| **Risk Gauge** | Risk from the forecast per raion (comparable across oblasts of different size) |
| **Multi-Oblast Risk Map** | Side-by-side forecast scores for all selected regions |
| **🧪 Backtest tab** | Final-test results of the forecasting study: skill with CIs, calibration, forecast vs actual (no token needed) |

> **What is counted.** Since December 2025 alerts are declared mostly per raion, not per oblast. All dashboard
> figures therefore count **raion alerts**: an oblast-wide alert counts once for each raion of the oblast, a
> raion alert for its raion, and overlapping announcements for one raion count once (`etl.raion_alerts`).
> Hromada- and city-level alerts are not counted, and Luhansk oblast (under permanent alert since 2022) is excluded.

---

## 🚀 Quick Start (Local)

### 1. Clone the repo

```bash
git clone https://github.com/your-username/Analysis-of-air-raid-alerts.git
cd Analysis-of-air-raid-alerts
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API token

Request a free token at [alerts.in.ua/api-request](https://alerts.in.ua/api-request), then:

```toml
# .streamlit/secrets.toml  (already gitignored — safe to add your token)
ALERTS_API_TOKEN = "your_token_here"
```

Or via environment variable:
```bash
export ALERTS_API_TOKEN="your_token_here"
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Streamlit Cloud Deployment

1. Push this repo to GitHub (`.streamlit/secrets.toml` is gitignored ✅).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point to `app.py` as the main file.
4. In **Advanced Settings → Secrets**, paste:
   ```toml
   ALERTS_API_TOKEN = "your_token_here"
   ```
5. Deploy. The app will auto-install `requirements.txt`.

You can open online result of my project for link:
https://analysis-of-air-raid-alerts-kpetxtqhtsiwrspcccpfpt.streamlit.app/
---

## 📐 Project Architecture

```
Analysis-of-air-raid-alerts/
├── app.py              # Streamlit UI (tabs: history, forecast, backtest)
├── etl.py              # API acquisition, cleaning & feature engineering (Kyiv calendar)
├── forecasting.py      # Dashboard forecast (Holt-Winters + Ridge fallback)
├── src/
│   ├── data/           # snapshots: official dataset, API, Telegram; stitching, outage mask
│   ├── features/       # hourly raion panel, as-of features, daily target B, text tags
│   ├── models/         # baselines, logistic regression / LightGBM / Poisson GLM, calibration, conformal
│   ├── evaluation/     # chronological splits with embargo, metrics, day-block bootstrap
│   └── viz/            # Plotly charts shared by app.py and the report; results loader
├── scripts/            # fetch_*, build_dataset, run_backtest*, run_final_test, build_report
├── tests/              # leakage ("perturb the future") and unit tests
├── config/evaluation.json  # frozen final-test window
├── reports/
│   ├── index.html      # self-contained static report (offline)
│   └── results/        # backtest + final-test tables, model configs
├── PLAN.md             # implementation plan (Ukrainian)
└── .streamlit/         # theme; secrets.toml is gitignored
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `etl.py` | `fetch_and_process_data()` → API calls, timestamp parsing, duration calc, feature engineering, rolling aggregations |
| `forecasting.py` | `train_and_forecast()` → model selection, Holt-Winters / Ridge fit, CI approximation |
| `app.py` | Streamlit layout, sidebar controls, KPI row, tab routing |
| `src/viz/charts.py` | All Plotly figures, shared by the dashboard and `reports/index.html` |

---

## ⚙️ API Rate Limits

The alerts.in.ua API enforces the following limits:

| Endpoint | Limit |
|---|---|
| `/v1/alerts/active.json` | 8–12 req/min |
| `/v1/regions/{uid}/alerts/month_ago.json` | **2 req/min** |

The app handles this automatically:
- **Active alerts:** Cached for 5 minutes (`@st.cache_data(ttl=300)`).
- **History:** Fetched once per session with a 31-second inter-request delay. Loading 24 oblasts (Luhansk, under permanent alert, is skipped) takes ~12 minutes on first load. Subsequent loads use the cache.
- **Toggle:** Disable "Fetch 30-day history" in the sidebar for instant load (active alerts only).

---

## 🤖 Forecasting Methodology

| Parameter | Value |
|---|---|
| Primary model | Holt-Winters Exponential Smoothing |
| Trend component | Additive |
| Seasonal component | Additive, period = 7 (weekly) |
| Fallback model | Ridge Regression (when < 14 data points) |
| Fallback features | Linear trend, sin/cos weekly + monthly encodings |
| Confidence intervals | 95% CI: ± 1.96 × std(in-sample residuals) |
| Forecast horizon | 3–14 days (configurable in sidebar) |

> On the final test (per raion, daily alert count), this Holt-Winters forecast was **worse than a plain 28-day
> mean** (MAE +31% at 1 day ahead), and its 95% interval covered only 54–66% of outcomes. See the forecasting
> study above.

---

## 🔐 Security

- **Zero hardcoded credentials** — token is always resolved from `st.secrets` or `os.getenv`.
- `.streamlit/secrets.toml` is in `.gitignore` and will never be committed.
- Rate-limit headers respected to avoid token blacklisting.

---

## 📄 License

MIT. Data is © alerts.in.ua volunteers. Use responsibly.
