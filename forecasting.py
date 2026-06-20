"""
forecasting.py — Time-Series Forecasting Module
================================================
Lightweight, compilation-free forecasting for daily alert counts per region.

Model strategy:
  Primary   → Holt-Winters Exponential Smoothing (additive trend + weekly seasonality)
  Fallback  → Ridge Regression with cyclical (sin/cos) and linear trend features
              (used when series has < 14 data points for Holt-Winters)

95% confidence intervals are approximated via residual standard deviation
on in-sample fit: CI = forecast ± 1.96 × std(residuals).
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Holt-Winters primary model
# ---------------------------------------------------------------------------

def _forecast_holtwinters(
    series: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    """
    Fit Holt-Winters Exponential Smoothing and produce a forecast.

    Args:
        series:  Daily alert-count Series with DatetimeIndex (UTC).
        horizon: Number of days to forecast forward.

    Returns:
        DataFrame with columns: date, forecast, ci_lower, ci_upper.

    Raises:
        Exception: Re-raises fitting exceptions so caller can fall back.
    """
    # Import here to avoid top-level import cost when module is loaded
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            series.values,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, use_brute=False)

    # In-sample residual std for CI computation
    residuals = series.values - fit.fittedvalues
    residual_std = float(np.std(residuals))

    raw_forecast = fit.forecast(horizon)
    # Clip negative forecasts — alert counts cannot be below zero
    forecast_values = np.clip(raw_forecast, a_min=0.0, a_max=None)

    last_date = series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
        tz="UTC",
    )

    z_95 = 1.96
    ci_half = z_95 * residual_std
    return pd.DataFrame({
        "date": future_dates,
        "forecast": np.round(forecast_values, 2),
        "ci_lower": np.clip(np.round(forecast_values - ci_half, 2), 0, None),
        "ci_upper": np.round(forecast_values + ci_half, 2),
        "model": "Holt-Winters",
    })


# ---------------------------------------------------------------------------
# Ridge regression fallback model
# ---------------------------------------------------------------------------

def _build_regression_features(n: int, start_offset: int = 0) -> np.ndarray:
    """
    Build feature matrix for the regression fallback model.

    Features per day t:
        - Linear trend index (t)
        - sin(2π·t / 7), cos(2π·t / 7)  — weekly cycle
        - sin(2π·t / 30), cos(2π·t / 30) — monthly cycle

    Args:
        n:            Number of time steps to generate.
        start_offset: Starting index (0 for training, len(series) for forecast).

    Returns:
        Feature matrix of shape (n, 5).
    """
    t = np.arange(start_offset, start_offset + n, dtype=float)
    features = np.column_stack([
        t,
        np.sin(2 * np.pi * t / 7),
        np.cos(2 * np.pi * t / 7),
        np.sin(2 * np.pi * t / 30),
        np.cos(2 * np.pi * t / 30),
    ])
    return features


def _forecast_ridge(
    series: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    """
    Fit a Ridge regression model with cyclical features as fallback.

    Args:
        series:  Daily alert-count Series with DatetimeIndex.
        horizon: Number of forecast days.

    Returns:
        DataFrame with columns: date, forecast, ci_lower, ci_upper, model.
    """
    y = series.values.astype(float)
    n_train = len(y)

    X_train = _build_regression_features(n_train, start_offset=0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y)

    residual_std = float(np.std(y - model.predict(X_train_scaled)))

    X_future = _build_regression_features(horizon, start_offset=n_train)
    X_future_scaled = scaler.transform(X_future)
    raw_forecast = model.predict(X_future_scaled)
    forecast_values = np.clip(raw_forecast, 0, None)

    last_date = series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
        tz="UTC",
    )

    z_95 = 1.96
    ci_half = z_95 * residual_std
    return pd.DataFrame({
        "date": future_dates,
        "forecast": np.round(forecast_values, 2),
        "ci_lower": np.clip(np.round(forecast_values - ci_half, 2), 0, None),
        "ci_upper": np.round(forecast_values + ci_half, 2),
        "model": "Ridge Regression",
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_series_for_region(
    daily_df: pd.DataFrame,
    region_name: Optional[str] = None,
) -> pd.Series:
    """
    Extract and prepare a clean daily count time-series from the daily aggregation.

    Fills gaps (zero-count days) so the series is contiguous.

    Args:
        daily_df:    Output of etl.compute_daily_counts() — must have 'date' and 'count' columns.
        region_name: Optional label for logging context.

    Returns:
        pd.Series with DatetimeIndex (UTC) and int64 alert counts, sorted ascending.
    """
    if daily_df.empty:
        logger.warning("Empty daily DataFrame for region=%s", region_name)
        return pd.Series([], dtype=float, name="count")

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date")

    # Ensure contiguous daily index (fill any gaps with 0)
    full_range = pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="D", tz="UTC")
    series = df.set_index("date")["count"].reindex(full_range, fill_value=0).astype(float)
    series.name = "count"
    return series


def train_and_forecast(
    series: pd.Series,
    horizon: int = 7,
) -> pd.DataFrame:
    """
    Train a forecasting model and produce predictions with 95% confidence intervals.

    Model selection:
        ≥ 14 data points → Holt-Winters Exponential Smoothing (primary)
        < 14 data points → Ridge Regression with cyclical features (fallback)

    Args:
        series:  Daily alert-count pd.Series with DatetimeIndex (UTC-aware),
                 as produced by prepare_series_for_region().
        horizon: Number of days to forecast. Defaults to 7.

    Returns:
        pd.DataFrame with columns:
            - date (DatetimeIndex, UTC)
            - forecast (float): Predicted alert count
            - ci_lower (float): Lower 95% confidence bound (≥ 0)
            - ci_upper (float): Upper 95% confidence bound
            - model (str):  Model name used for the forecast

    Raises:
        ValueError: If series is empty.
    """
    if series.empty:
        raise ValueError("Cannot forecast an empty series.")

    n = len(series)
    logger.info("Forecasting %d days ahead from %d historical data points.", horizon, n)

    # Try primary model first; fall back gracefully
    if n >= 14:
        try:
            return _forecast_holtwinters(series, horizon)
        except Exception as exc:
            logger.warning(
                "Holt-Winters failed (%s). Falling back to Ridge Regression.", exc
            )

    return _forecast_ridge(series, horizon)


def compute_national_forecast(
    df: pd.DataFrame,
    horizon: int = 7,
) -> pd.DataFrame:
    """
    Compute a national-level forecast by aggregating all-oblast daily counts.

    Args:
        df:      Processed alerts DataFrame from etl.fetch_and_process_data().
        horizon: Forecast horizon in days.

    Returns:
        Forecast DataFrame (same schema as train_and_forecast output).
    """
    from etl import compute_daily_counts

    daily = compute_daily_counts(df)
    series = prepare_series_for_region(daily, region_name="Ukraine (national)")
    if series.empty:
        return pd.DataFrame(columns=["date", "forecast", "ci_lower", "ci_upper", "model"])
    return train_and_forecast(series, horizon)


def compute_per_oblast_forecast(
    df: pd.DataFrame,
    horizon: int = 7,
) -> dict[str, pd.DataFrame]:
    """
    Compute 7-day forecasts for each oblast independently.

    Skips oblasts with fewer than 3 data points (insufficient history).

    Args:
        df:      Processed alerts DataFrame.
        horizon: Forecast horizon in days.

    Returns:
        Dict mapping oblast English name → forecast DataFrame.
    """
    from etl import compute_daily_counts, OBLAST_EN_MAP

    results: dict[str, pd.DataFrame] = {}
    oblast_names = list(OBLAST_EN_MAP.values())

    for oblast_en in oblast_names:
        daily = compute_daily_counts(df, oblast_filter=[oblast_en])
        series = prepare_series_for_region(daily, region_name=oblast_en)
        if len(series) < 3:
            logger.debug("Skipping forecast for %s — insufficient data (%d points).", oblast_en, len(series))
            continue
        try:
            forecast_df = train_and_forecast(series, horizon)
            results[oblast_en] = forecast_df
        except Exception as exc:
            logger.warning("Forecast failed for %s: %s", oblast_en, exc)
    return results


def compute_risk_forecast_score(
    forecast_df: pd.DataFrame,
) -> float:
    """
    Derive a single risk score for the upcoming forecast window.

    Score = mean forecast / (mean forecast + 1) scaled to [0, 1].
    Higher score → higher predicted alert frequency.

    Args:
        forecast_df: Output of train_and_forecast().

    Returns:
        Float in [0, 1].
    """
    if forecast_df.empty:
        return 0.0
    mean_fc = forecast_df["forecast"].mean()
    return float(mean_fc / (mean_fc + 1)) if mean_fc >= 0 else 0.0
