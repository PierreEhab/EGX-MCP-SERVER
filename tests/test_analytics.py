import math

import pandas as pd
import pytest

from egx_mcp.analytics import (
    history_rows,
    key_levels,
    relative_performance,
    stock_summary,
    summarize,
)
from egx_mcp.server import _sample_history


def frame(values: list[float | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=len(values), freq="D"),
            "value": values,
        }
    )


def test_summary_calculates_changes_range_and_volatility() -> None:
    result = summarize(frame([100, 110, 99]))

    assert result["as_of"] == "2026-01-03"
    assert result["latest_value"] == 99.0
    assert result["previous_close"] == 110.0
    assert result["daily_change"] == -11.0
    assert result["daily_change_pct"] == -10.0
    assert result["period_high"] == 110.0
    assert result["period_low"] == 99.0
    assert math.isclose(result["period_return_pct"], -1.0)
    assert math.isclose(
        result["annualized_volatility_pct"],
        pd.Series([0.1, -0.1]).std(ddof=1) * math.sqrt(252) * 100,
    )
    assert result["date_range"] == {
        "start": "2026-01-01",
        "end": "2026-01-03",
    }


def test_moving_averages_are_only_reported_when_enough_values_exist() -> None:
    result = summarize(frame(list(range(1, 51))))

    assert result["sma_20"] == 40.5
    assert result["sma_50"] == 25.5

    short_result = summarize(frame(list(range(1, 20))))
    assert short_result["sma_20"] is None
    assert short_result["sma_50"] is None


def test_short_and_empty_series_return_null_analytics_without_crashing() -> None:
    one_value = summarize(frame([12]))
    assert one_value["latest_value"] == 12.0
    assert one_value["previous_close"] is None
    assert one_value["daily_change"] is None
    assert one_value["daily_change_pct"] is None
    assert one_value["period_return_pct"] == 0.0
    assert one_value["annualized_volatility_pct"] is None

    two_values = summarize(frame([12, 12]))
    assert two_values["annualized_volatility_pct"] is None

    empty = summarize(frame([]))
    assert empty["as_of"] is None
    assert empty["latest_value"] is None
    assert empty["period_return_pct"] is None
    assert empty["date_range"] == {"start": None, "end": None}


def test_missing_trailing_values_are_ignored_consistently() -> None:
    data = frame([100, None, 110])
    data.loc[len(data)] = [pd.Timestamp("2026-01-04"), None]

    result = summarize(data)
    assert result["latest_value"] == 110.0
    assert result["previous_close"] == 100.0
    assert math.isclose(result["period_return_pct"], 10.0)
    assert result["as_of"] == result["date_range"]["end"] == "2026-01-03"
    assert history_rows(data) == [
        {"date": "2026-01-01", "value": 100.0},
        {"date": "2026-01-03", "value": 110.0},
    ]


def test_zero_previous_close_does_not_create_infinite_percentages() -> None:
    result = summarize(frame([0, 10]))

    assert result["daily_change"] == 10.0
    assert result["daily_change_pct"] is None
    assert result["period_return_pct"] is None


def test_non_finite_values_are_omitted_consistently() -> None:
    data = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "value": [float("inf"), 10.0],
        }
    )

    assert history_rows(data) == [{"date": "2026-01-02", "value": 10.0}]
    assert summarize(data)["latest_value"] == 10.0


def test_history_omits_rows_with_invalid_dates_or_values() -> None:
    data = pd.DataFrame(
        {
            "datetime": ["2026-01-01", "not a date", "2026-01-03"],
            "value": [100.0, 200.0, float("nan")],
        }
    )

    assert history_rows(data) == [{"date": "2026-01-01", "value": 100.0}]


def test_drawdown_and_five_observation_return() -> None:
    result = summarize(frame([100, 120, 90, 100, 110, 115]))

    assert math.isclose(result["drawdown_from_period_high_pct"], -4.1666666667)
    assert math.isclose(result["return_5d_pct"], 15.0)


def test_weekly_sampling_uses_thursday_week_ends() -> None:
    dates = pd.to_datetime(
        [
            "2026-10-04",
            "2026-10-05",
            "2026-10-07",
            "2026-10-08",
            "2026-10-11",
            "2026-10-12",
            "2026-10-15",
        ]
    )
    data = pd.DataFrame({"datetime": dates, "value": [100, 101, 102, 103, 104, 105, 106]})

    sampled = _sample_history(data, "weekly", None)

    assert sampled["datetime"].dt.date.tolist() == [
        pd.Timestamp("2026-10-08").date(),
        pd.Timestamp("2026-10-15").date(),
    ]
    assert sampled["value"].tolist() == [103.0, 106.0]


def test_drawdown_and_five_observation_return_for_short_series() -> None:
    one = summarize(frame([100]))
    assert one["drawdown_from_period_high_pct"] == 0.0
    assert one["return_5d_pct"] is None
    zero = summarize(frame([0]))
    assert zero["drawdown_from_period_high_pct"] == 0.0

    four = summarize(frame([100, 110, 105, 108]))
    assert math.isclose(four["drawdown_from_period_high_pct"], -100 * (1 - 108 / 110))
    assert four["return_5d_pct"] is None


def stock_bars(
    closes: list[float],
    start: str = "2024-01-01",
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [value + 2 for value in closes],
            "low": [value - 2 for value in closes],
            "close": closes,
            "adj_close": [value * 0.9 for value in closes],
            "volume": [1000 + index for index in range(len(closes))],
        }
    )


def test_stock_summary_computes_price_total_return_and_risk_metrics() -> None:
    bars = stock_bars([100, 110, 90, 120])
    dividends = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-04"]),
            "amount": [1.2],
        }
    )

    result = stock_summary(bars, dividends, period_days=30)

    assert result["as_of"] == "2024-01-04"
    assert result["latest_close"] == 120.0
    assert result["daily_change"] == 30.0
    assert result["daily_change_pct"] == pytest.approx(33.3333333333)
    assert result["period_high"] == 122.0
    assert result["period_low"] == 88.0
    assert result["sma_20"] is None
    assert result["price_return_pct"] == 20.0
    assert result["total_return_pct"] == 20.0
    assert result["max_drawdown_pct"] == pytest.approx(-18.1818181818)
    assert result["drawdown_from_period_high_pct"] == 0.0
    assert result["dividend_yield_ttm_pct"] == 1.0
    assert result["average_volume_20d"] is None
    assert result["returns"]["1m"]["price_return_pct"] is None


def test_stock_drawdowns_use_closing_prices_not_intraday_highs() -> None:
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "high": [200.0, 120.0, 150.0],
            "low": [90.0, 70.0, 80.0],
            "close": [100.0, 80.0, 90.0],
            "adj_close": [100.0, 80.0, 90.0],
        }
    )

    result = stock_summary(bars, pd.DataFrame(), period_days=30)

    assert result["drawdown_from_period_high_pct"] == pytest.approx(-10.0)
    assert result["max_drawdown_pct"] == pytest.approx(-20.0)


def test_stock_summary_longer_sma_and_average_volume_require_enough_rows() -> None:
    values = list(range(1, 201))
    result = stock_summary(stock_bars(values, start="2024-01-01"), pd.DataFrame(), 900)

    assert result["sma_20"] == 190.5
    assert result["sma_50"] == 175.5
    assert result["sma_200"] == 100.5
    assert result["average_volume_20d"] == pytest.approx(1189.5)
    assert result["average_volume_50d"] == pytest.approx(1174.5)
    assert result["week52_high"] is not None
    assert result["week52_low"] is not None


def test_stock_summary_returns_nulls_for_empty_and_short_inputs() -> None:
    empty = stock_summary(pd.DataFrame(), pd.DataFrame(), 365)
    assert empty["as_of"] is None
    assert empty["latest_close"] is None
    assert empty["price_return_pct"] is None
    assert empty["annualized_volatility_pct"] is None
    assert empty["dividend_yield_ttm_pct"] is None

    one = stock_summary(stock_bars([10]), pd.DataFrame(), 365)
    assert one["latest_close"] == 10.0
    assert one["price_return_pct"] == 0.0
    assert one["annualized_volatility_pct"] is None
    assert one["sma_20"] is None


def test_stock_summary_uses_adj_close_for_total_return_and_ttm_dividends() -> None:
    bars = stock_bars([100, 110, 120])
    bars.loc[:, "adj_close"] = [90, 95, 108]
    dividends = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-12-31", "2024-01-02"]),
            "amount": [99, 2],
        }
    )

    result = stock_summary(bars, dividends, period_days=30)

    assert result["price_return_pct"] == 20.0
    assert result["total_return_pct"] == 20.0
    assert result["dividend_yield_ttm_pct"] == pytest.approx(1.6666666667)


def test_stock_summary_accepts_up_to_five_calendar_days_of_window_start_slack() -> None:
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-10-26", "2026-10-22"]),
            "close": [100.0, 120.0],
            "adj_close": [95.0, 114.0],
        }
    )

    result = stock_summary(bars, pd.DataFrame(), period_days=365)

    assert result["returns"]["1y"]["price_return_pct"] == pytest.approx(20.0)
    assert result["returns"]["1y"]["total_return_pct"] == pytest.approx(20.0)
    assert result["returns"]["1y"]["start_date"] == "2025-10-26"


def test_relative_performance_aligns_common_dates_and_requires_30_observations() -> None:
    dates = pd.bdate_range("2024-01-01", periods=40)
    stock = pd.Series([2 * value for value in range(200, 240)], index=dates)
    index = pd.Series(range(200, 240), index=dates)

    result = relative_performance(stock, index)

    assert result["observations"] == 40
    assert result["correlation"] == pytest.approx(1.0)
    assert result["beta"] == pytest.approx(1.0)
    assert result["excess_return_pct"] == pytest.approx(0.0)

    short = relative_performance(stock.iloc[:20], index.iloc[:20])
    assert short["correlation"] is None
    assert short["beta"] is None
    assert short["excess_return_pct"] is None
    assert "30" in short["message"]


def test_relative_performance_accepts_date_value_frames_and_handles_constant_index() -> None:
    dates = pd.bdate_range("2024-01-01", periods=35)
    stock = pd.DataFrame({"date": dates, "close": range(10, 45)})
    index = pd.DataFrame({"date": dates, "value": [100.0] * len(dates)})

    result = relative_performance(stock, index)

    assert result["observations"] == 35
    assert result["beta"] is None
    assert result["correlation"] is None
    assert result["excess_return_pct"] == pytest.approx(340.0)


def test_key_levels_returns_week_range_sma_and_60_session_swings() -> None:
    bars = stock_bars(list(range(1, 81)), start="2024-01-01")

    result = key_levels(bars)

    assert result["week52_high"] == 82.0
    assert result["week52_low"] == -1.0
    assert result["sma_20"] == 70.5
    assert result["sma_50"] == 55.5
    assert result["sma_200"] is None
    assert result["swing_high_60d"] == 82.0
    assert result["swing_low_60d"] == 19.0


def test_stock_analytics_drop_invalid_dates_and_non_finite_closes() -> None:
    bars = pd.DataFrame(
        {
            "date": ["2024-01-01", "bad", "2024-01-03"],
            "close": [10, 20, float("inf")],
            "adj_close": [9, 18, 27],
        }
    )

    result = stock_summary(bars, pd.DataFrame(), period_days=30)

    assert result["as_of"] == "2024-01-01"
    assert result["latest_close"] == 10.0
