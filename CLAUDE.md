# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Streamlit dashboard that analyses and forecasts Ukrainian air-raid alerts from the alerts.in.ua API (docs: https://devs.alerts.in.ua). Deployed on Streamlit Cloud with `app.py` as the entry point. The forecasting project (see `PLAN.md`) adds a data pipeline in `src/`, CLIs in `scripts/` and tests in `tests/`.

## Commands

Use the project venv (Python 3.12; the pins in `requirements.txt` don't build on 3.14, and `pyarrow` must stay <18 for numpy 1.26):

```bash
uv venv --python /opt/homebrew/bin/python3.12 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/streamlit run app.py                  # dashboard, http://localhost:8501
.venv/bin/python -m pytest                      # all tests (pytest.ini sets pythonpath)
.venv/bin/python -m pytest tests/test_merge.py::test_open_alert_is_closed_at_cutoff_not_later   # single test
.venv/bin/python scripts/fetch_data.py [--api]  # new raw snapshots; --api takes ~13 min (rate limit)
.venv/bin/python scripts/fetch_telegram.py --since 2025-11-01  # Air Force channel posts from t.me/s preview (~1.5 h, resumable)
.venv/bin/python scripts/build_dataset.py       # data/raw -> data/processed
.venv/bin/python scripts/run_backtest.py [--no-text]  # target A: select + fit models (+ *_text ablation if a Telegram snapshot exists), score vs baselines on folds + validation -> reports/results/ (~15 min)
.venv/bin/python scripts/run_backtest_daily.py  # target B (daily count / hours per raion, k = 1..7 days), same protocol (~10 min)
```

The app requires `ALERTS_API_TOKEN`, resolved in `get_api_token()` (app.py) from `st.secrets` first, then the environment variable. Locally, put it in `.streamlit/secrets.toml` (gitignored).

## Architecture

Three modules form a one-way pipeline: `etl.py` → `forecasting.py` → `app.py`.

- **`etl.py`**: `fetch_and_process_data()` is the single ETL entry point. It fetches `/v1/alerts/active.json`, then (optionally) loops over every oblast in `OBLAST_UID_MAP` calling the per-region `month_ago` history endpoint. It normalizes the records (`_normalize_alert_record`), dedupes on `(id, started_at)`, converts to UTC-aware timestamps and adds features (`_engineer_features`). It returns `(df, status)`, where `status` holds `api_ok`, `active_count`, `last_updated`, `error` and `history_failures`. Auth/network failures raise `RuntimeError`, which `app.py` catches and shows to the user. A failed history fetch for one oblast is logged in `history_failures` and skipped. The module also has the aggregation helpers the UI uses: `compute_daily_counts`, `compute_heatmap_data` and `compute_vulnerability_score` (60% frequency + 40% duration).
- **Oblast naming**: the API returns Ukrainian oblast names. `OBLAST_UID_MAP` maps them to API UIDs, `OBLAST_EN_MAP` maps them to English display labels, and `ALERT_TYPE_EN_MAP` translates alert types. Any new region or alert type has to be added to these maps.
- **`forecasting.py`**: `train_and_forecast(series, horizon)` takes a contiguous daily count series with a UTC DatetimeIndex (built by `prepare_series_for_region`, which fills gaps with zeros) and returns a DataFrame with the columns `date, forecast, ci_lower, ci_upper, model`. How the model is chosen:
  - a constant series gets a flat forecast,
  - a series with ≥14 points uses additive Holt-Winters with weekly seasonality,
  - on any Holt-Winters failure, or with <14 points, it falls back to Ridge regression on trend plus sin/cos weekly and monthly features.

  The CI is ±1.96·std of the in-sample residuals. `statsmodels` and the `etl` helpers are imported inside functions, not at module top level.
- **`app.py`**: the Streamlit UI. It contains the custom CSS, colour constants (dark theme matching `.streamlit/config.toml`), Plotly chart builders (`make_*_chart`), and the renderers `render_sidebar`, `render_kpi_row`, `render_history_tab` and `render_forecast_tab`, all orchestrated by `main()`.
  - `load_data` wraps the ETL in `@st.cache_data(ttl=300)`.
  - `get_forecast` wraps forecasting in `@st.cache_resource` and takes hashable tuples of values and dates instead of a Series.
  - `main()` reads `st.session_state["fetch_history"]` *before* the sidebar renders, so the sidebar toggle takes effect on the next rerun.

## Data pipeline (`src/data/`, forecasting project)

`data/` is gitignored and reproducible from the scripts. Raw snapshots are immutable (never overwrite; new download = new file):
- `load_dataset.py`: GitHub dataset `Vadimkin/ukrainian-air-raid-sirens-dataset`, pinned to a commit → `data/raw/github/<dataset>/<fetched_at>_<sha>.csv.gz`. The official CSV contains a fully duplicated block (dropped in `normalize_official`) and currently ends 2026-09-07.
- `fetch_api.py`: alerts.in.ua `month_ago` per oblast → `data/raw/alerts_in_ua/<fetched_at>/<uid>.json` + `manifest.json`.
- `merge.py`: one event schema for all sources; harmonizes oblast/raion names (apostrophes, 2024 raion renames); `expand_to_raions` maps oblast-level alerts onto every raion of the oblast; `union_intervals` merges overlaps per unit.
- `build.py`: stitches sources **by time, never mixed**: official → gap → API. Writes `data/processed/raion_intervals.parquet` and `coverage.json` (windows, gap, provenance, data cutoff). The forecast unit is the raion (118 + Kyiv City); hromada-level alerts and Luhansk oblast (permanent alert) are excluded.

## Evaluation framework (`src/features`, `src/evaluation`, `src/models`)

- `features/panel.py`: wide hourly matrices (UTC hours × units): `starts`, `active_minutes`, `covered`, `active_at` (under alert at the instant t). Target A `labels(panel, h)`: 1 if a new alert starts in [t, t+h), NaN if the window touches an uncovered hour. The label of origin t is only known at t + h — anything used as input at t may only read labels with t' + h <= t.
- `evaluation/splits.py`: final test = last 28 days (scored **once**, at the end of the project; `run_backtest.py` must not score it), validation = 28 days before the data gap, 6 weekly rolling folds; embargo = h (`Split.train_end`).
- `features/asof.py`: features as of origin t (row t reads only hours < t, `active_at[t]`, labels with t' + h <= t via the baselines, and the Kyiv calendar). Unit history, same-oblast / neighbouring-oblast (`features/geo.py`, hand-written adjacency) / country aggregates. `stack` turns wide features into the long (origin, unit) design matrix.
- `features/text.py` (stage 4): Air Force posts (`data/telegram.py`, `data/raw/telegram/kpszsu/*.jsonl.gz`) tagged by regex (threat categories, oblast/city mentions incl. oblique cases, launched counts from the "противник атакував" sentence). Hourly by UTC publication hour; row t sees posts with date_utc < t. Edited posts are kept but their content is NaN (dropping them would depend on edits after t). The tagging rules are frozen for the final test (sha256 in `model_config.json`) — don't edit `text.py` before the holdout run. Text did not improve target A.
- `models/tabular.py`: logistic regression and LightGBM, fit per split on `train_rows` (origins < `train_end(h)` within a window of days); Platt/isotonic calibrators. `run_backtest.py` selects the training window and tree settings on validation only (so validation scores of models are optimistic; folds reuse those later-chosen settings too, so only the final test is unbiased) and writes `model_config.json` for the final-test run.
- `features/daily.py`: target B on Kyiv-local days (23/25-hour DST days handled; a day is covered only if all its hours are). Origin = Kyiv midnight of day d; `for_horizon(F, daily, k)` puts features of origin D-k+1 on target day D; `DaySplit.train_end(k)`: training target days <= split start - k. `make_daily_splits` uses the same frozen test window (whole Kyiv days only).
- `models/baselines.py`: reference forecasts on the label matrix (target A) and daily baselines incl. the dashboard's Holt-Winters refit per origin (target B). Shifts are in rows, so they require a regular hourly index; never group by Kyiv local hour (DST makes "previous same hour" 1 h away).
- `evaluation/metrics.py`: Brier/BSS, log loss, ROC-AUC, PR-AUC and `within_unit_auc` — pooled AUC is dominated by frontline-vs-west differences; within-unit AUC measures timing skill.
- `tests/test_leakage.py`, `tests/test_features.py`, `tests/test_daily.py` and `tests/test_text.py` hold the future-perturbation tests; any new feature or model needs one (`test_features.py` rewrites raw intervals after t and checks every feature at t).

Open alerts are closed at the data cutoff (`open_until`), never at "now", to avoid leakage.

## API rate limits (important when touching ETL or caching)

- Active alerts: ~8–12 req/min.
- Per-region history: **2 req/min**. `fetch_and_process_data` sleeps 31 s between oblast requests, so a full history load (~25 oblasts) takes ~13 minutes. Never remove the delay or parallelise these calls, or the token risks being blacklisted.
- Turning off "Fetch 30-day history" in the sidebar loads only active alerts, which is instant. Use this while iterating on the UI.

## Project tooling (`.claude/`)

- `PLAN.md` is the implementation plan (in Ukrainian) for the forecasting project; section 3 defines the no-leakage rules.
- Subagent `leakage-auditor`: read-only review for temporal data leakage. Run it after changing features, splits, evaluation or text features, and before the final holdout evaluation.
- A PreToolUse hook (`.claude/hooks/block_secret_commit.py`) blocks `git commit` when the commit would include `.streamlit/secrets.toml`, `.env`, or the token value.
