# 🇺🇦 Air Raid Alarms Analytics Dashboard

A **time-series analysis and forecasting dashboard** for Ukrainian air-raid alerts, built with Streamlit, Plotly, and statsmodels.

> **Data source:** [alerts.in.ua](https://alerts.in.ua) — volunteer-maintained real-time alert map  
> **API docs:** [devs.alerts.in.ua](https://devs.alerts.in.ua)

---

## 📸 Features

| Feature | Details |
|---|---|
| **Live Active Alerts** | Real-time alert count fetched every 5 minutes |
| **30-Day Historical Data** | Per-oblast history via the regional history endpoint |
| **KPI Metrics Panel** | Total alerts, avg/max duration, affected oblasts, delta vs previous window |
| **Daily Trend Chart** | Bar + 7-day MA + OLS trendline |
| **Diurnal Heatmap** | Hour of Day × Day of Week alert frequency |
| **Alert Type Breakdown** | Horizontal bar by type with avg duration |
| **Vulnerability Ranking** | Weighted score: 60% frequency + 40% duration |
| **7-Day Forecast** | Holt-Winters (primary) or Ridge Regression (fallback) with 95% CI ribbon |
| **Risk Gauge** | Per-region risk score visualised as a gauge chart |
| **Multi-Oblast Risk Map** | Side-by-side forecast scores for all selected regions |

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
├── app.py              # Main Streamlit UI — all visualization logic
├── etl.py              # Data acquisition, cleaning & feature engineering
├── forecasting.py      # Time-series forecasting (Holt-Winters + Ridge fallback)
├── requirements.txt    # Pinned Python dependencies
├── .streamlit/
│   ├── config.toml     # Dark-mode theme & server settings
│   └── secrets.toml    # 🔒 API token (gitignored — never commit)
└── .gitignore
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `etl.py` | `fetch_and_process_data()` → API calls, timestamp parsing, duration calc, feature engineering, rolling aggregations |
| `forecasting.py` | `train_and_forecast()` → model selection, Holt-Winters / Ridge fit, CI approximation |
| `app.py` | Streamlit layout, Plotly chart builders, sidebar controls, KPI row, tab routing |

---

## ⚙️ API Rate Limits

The alerts.in.ua API enforces the following limits:

| Endpoint | Limit |
|---|---|
| `/v1/alerts/active.json` | 8–12 req/min |
| `/v1/regions/{uid}/alerts/month_ago.json` | **2 req/min** |

The app handles this automatically:
- **Active alerts:** Cached for 5 minutes (`@st.cache_data(ttl=300)`).
- **History:** Fetched once per session with a 31-second inter-request delay. Loading all 25 oblasts takes ~13 minutes on first load. Subsequent loads use the cache.
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

---

## 🔐 Security

- **Zero hardcoded credentials** — token is always resolved from `st.secrets` or `os.getenv`.
- `.streamlit/secrets.toml` is in `.gitignore` and will never be committed.
- Rate-limit headers respected to avoid token blacklisting.

---

## 📄 License

MIT. Data is © alerts.in.ua volunteers. Use responsibly.
