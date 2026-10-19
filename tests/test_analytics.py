import math

import pandas as pd

from egx_mcp.analytics import history_rows, summarize
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
