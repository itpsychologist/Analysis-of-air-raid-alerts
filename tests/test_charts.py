"""Dashboard charts (src/viz/charts.py)."""

import pandas as pd

from src.viz.charts import make_trend_chart


def test_trend_chart_drops_the_partial_first_day_and_keeps_today_out_of_the_average():
    daily = pd.DataFrame({
        "date": pd.date_range("2026-10-05", periods=5, freq="D"),
        "count": [10, 400, 500, 600, 50],  # first day starts mid-day, today is still running
        "is_complete": [False, True, True, True, False],
    })
    fig = make_trend_chart(daily, ["Kharkiv"])
    traces = {t.name: t for t in fig.data}
    assert list(traces["Daily Alerts"].y) == [400, 500, 600]
    assert list(traces["Today (so far)"].y) == [50]
    assert list(traces["7-Day Avg"].y) == [400, 450, 500]  # complete days only
    assert "OLS Trend" not in traces  # fewer than 4 complete days
