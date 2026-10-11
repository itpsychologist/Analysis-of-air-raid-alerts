# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Streamlit dashboard that analyses and forecasts Ukrainian air-raid alerts from the alerts.in.ua API (docs: https://devs.alerts.in.ua). Deployed on Streamlit Cloud with `app.py` as the entry point. There are no tests, linters, or build steps.

## Commands

```bash
pip install -r requirements.txt   # pinned versions (streamlit 1.35, pandas 2.2, statsmodels 0.14, scikit-learn 1.5)
streamlit run app.py              # http://localhost:8501
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

## API rate limits (important when touching ETL or caching)

- Active alerts: ~8–12 req/min.
- Per-region history: **2 req/min**. `fetch_and_process_data` sleeps 31 s between oblast requests, so a full history load (~25 oblasts) takes ~13 minutes. Never remove the delay or parallelise these calls, or the token risks being blacklisted.
- Turning off "Fetch 30-day history" in the sidebar loads only active alerts, which is instant. Use this while iterating on the UI.

## Project tooling (`.claude/`)

- `PLAN.md` is the implementation plan (in Ukrainian) for the forecasting project; section 3 defines the no-leakage rules.
- Subagent `leakage-auditor`: read-only review for temporal data leakage. Run it after changing features, splits, evaluation or text features, and before the final holdout evaluation.
- A PreToolUse hook (`.claude/hooks/block_secret_commit.py`) blocks `git commit` when the commit would include `.streamlit/secrets.toml`, `.env`, or the token value.
