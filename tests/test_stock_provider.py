from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import logging

from egx_mcp import stock_provider
from egx_mcp.stock_provider import (
    EmptyStockDataError,
    StockDataSourceError,
    YahooProvider,
)


ENTRY = {
    "symbol": "TEST",
    "yahoo_ticker": "TEST.CA",
}


def test_yfinance_logger_is_silenced_at_import_level() -> None:
    assert logging.getLogger("yfinance").level == logging.CRITICAL


def test_history_normalizes_dates_numeric_values_and_invalid_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cairo = ZoneInfo("Africa/Cairo")
    today = datetime.now(cairo).date()
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp(today - pd.Timedelta(days=3), tz=cairo),
            pd.Timestamp(today - pd.Timedelta(days=2), tz=cairo),
            pd.Timestamp(today - pd.Timedelta(days=1), tz=cairo),
        ]
    )
    source = pd.DataFrame(
        {
            "Open": ["10", "bad", "12"],
            "High": ["11", "11", "13"],
            "Low": ["9", "9", "11"],
            "Close": ["10.5", "NaN", "inf"],
            "Adj Close": ["10", "NaN", "inf"],
            "Volume": ["100", "200", "300"],
        },
        index=dates,
    )
    monkeypatch.setattr(
        stock_provider.yf,
        "Ticker",
        lambda ticker: type("TickerStub", (), {"history": lambda self, **kwargs: source})(),
    )
    monkeypatch.setattr(stock_provider, "_minimum_interval", lambda: 0)

    result = YahooProvider().get_history(ENTRY, period_days=90)

    assert result.columns.tolist() == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert len(result) == 1
    assert result.iloc[0]["date"] == today - pd.Timedelta(days=3)
    assert result.iloc[0]["close"] == 10.5


def test_history_prices_and_dividend_amounts_are_rounded_to_four_decimals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    today = datetime.now(ZoneInfo("Africa/Cairo")).date()
    history = pd.DataFrame(
        {
            "Open": [125.1500015258789],
            "High": [126.1500015258789],
            "Low": [124.1500015258789],
            "Close": [125.1500015258789],
            "Adj Close": [124.987654321],
            "Volume": [100],
        },
        index=pd.DatetimeIndex([pd.Timestamp(today, tz="Africa/Cairo")]),
    )
    dividends = pd.Series(
        [0.123456789],
        index=pd.DatetimeIndex([pd.Timestamp(today, tz="Africa/Cairo")]),
    )

    monkeypatch.setattr(
        stock_provider.yf,
        "Ticker",
        lambda ticker: type(
            "TickerStub",
            (),
            {
                "history": lambda self, **kwargs: history,
                "dividends": dividends,
            },
        )(),
    )
    monkeypatch.setattr(stock_provider, "_minimum_interval", lambda: 0)

    provider = YahooProvider()
    bars = provider.get_history(ENTRY, period_days=3)
    payments = provider.get_dividends(ENTRY, years=1)

    assert bars.iloc[0]["open"] == 125.15
    assert bars.iloc[0]["high"] == 126.15
    assert bars.iloc[0]["low"] == 124.15
    assert bars.iloc[0]["close"] == 125.15
    assert bars.iloc[0]["adj_close"] == 124.9877
    assert payments.iloc[0]["amount"] == 0.1235


def test_yahoo_call_counter_counts_actual_wrapped_calls() -> None:
    before = stock_provider.yahoo_call_count()
    assert stock_provider._yahoo_call("TEST", "test", lambda: "ok") == "ok"
    assert stock_provider.yahoo_call_count() == before + 1


def test_yahoo_key_error_becomes_typed_data_source_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stock_provider.yf,
        "Ticker",
        lambda ticker: type(
            "TickerStub",
            (),
            {"history": lambda self, **kwargs: (_ for _ in ()).throw(KeyError("tradingPeriods"))},
        )(),
    )
    monkeypatch.setattr(stock_provider, "_minimum_interval", lambda: 0)

    with pytest.raises(StockDataSourceError) as error:
        YahooProvider().get_history(ENTRY, period_days=30)

    assert error.value.code == "DATA_SOURCE_ERROR"
    assert "tradingPeriods" not in str(error.value)


def test_empty_yahoo_history_raises_empty_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stock_provider.yf,
        "Ticker",
        lambda ticker: type(
            "TickerStub",
            (),
            {"history": lambda self, **kwargs: pd.DataFrame()},
        )(),
    )
    monkeypatch.setattr(stock_provider, "_minimum_interval", lambda: 0)

    with pytest.raises(EmptyStockDataError) as error:
        YahooProvider().get_history(ENTRY, period_days=30)

    assert error.value.code == "EMPTY_DATA"


def test_dividends_and_info_are_normalized_and_missing_values_stay_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cairo = ZoneInfo("Africa/Cairo")
    today = datetime.now(cairo).date()
    dividend_series = pd.Series(
        [0.5, float("nan")],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp(today - pd.Timedelta(days=10), tz=cairo),
                pd.Timestamp(today - pd.Timedelta(days=5), tz=cairo),
            ]
        ),
    )

    class TickerStub:
        dividends = dividend_series
        info = {
            "longName": "Test Company",
            "marketCap": 1000,
            "trailingPE": None,
            "fiftyTwoWeekHigh": 20,
            "currency": "EGP",
        }

    monkeypatch.setattr(stock_provider.yf, "Ticker", lambda ticker: TickerStub())
    monkeypatch.setattr(stock_provider, "_minimum_interval", lambda: 0)
    provider = YahooProvider()

    dividends = provider.get_dividends(ENTRY, years=1)
    info = provider.get_info_light(ENTRY)

    assert dividends.to_dict("records") == [
        {"date": today - pd.Timedelta(days=10), "amount": 0.5}
    ]
    assert info == {
        "name": "Test Company",
        "market_cap": 1000.0,
        "trailing_pe": None,
        "week52_high": 20.0,
        "week52_low": None,
        "currency": "EGP",
    }
