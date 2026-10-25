"""Yahoo Finance adapter for daily EGX company data."""

from __future__ import annotations

import logging
import math
import os
import random
import sys
import threading
import time
import warnings
from datetime import date, datetime, timedelta
from contextlib import redirect_stdout
from typing import Any, Protocol

import pandas as pd
import yfinance as yf

from .egx_client import _cairo_timezone

warnings.filterwarnings("ignore", module="yfinance")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

SOURCE = "Yahoo Finance via yfinance (delayed)"
_logger = logging.getLogger(__name__)
_YAHOO_LOCK = threading.Lock()
_LAST_YAHOO_CALL = 0.0
_YAHOO_CALL_COUNT = 0
_MAX_ATTEMPTS = 3
_BAR_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)


class StockDataError(Exception):
    """Base class for typed errors raised by a stock data provider."""

    code = "DATA_SOURCE_ERROR"


class StockSymbolNotFoundError(StockDataError):
    """No EGX company in the symbol catalog matches the request."""

    code = "SYMBOL_NOT_FOUND"


class EmptyStockDataError(StockDataError):
    """Yahoo returned no usable rows for the requested stock data."""

    code = "EMPTY_DATA"


class StockDataSourceError(StockDataError):
    """Yahoo failed or returned data in an unsupported format."""

    code = "DATA_SOURCE_ERROR"


class StockDataProvider(Protocol):
    """Interface implemented by sources of individual EGX company data."""

    def get_history(self, entry: dict[str, Any], period_days: int) -> pd.DataFrame:
        """Return normalized daily OHLCV bars for a catalog company."""
        ...

    def get_dividends(self, entry: dict[str, Any], years: int) -> pd.DataFrame:
        """Return dividend payment dates and amounts for a catalog company."""
        ...

    def get_info_light(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Return the small set of Yahoo-provided company information fields."""
        ...

def _minimum_interval() -> float:
    configured = os.getenv("EGX_YAHOO_MIN_INTERVAL_SECONDS", "0.5")
    try:
        interval = float(configured)
    except ValueError as exc:
        raise ValueError(
            "EGX_YAHOO_MIN_INTERVAL_SECONDS must be a non-negative number."
        ) from exc
    if not math.isfinite(interval) or interval < 0:
        raise ValueError(
            "EGX_YAHOO_MIN_INTERVAL_SECONDS must be a finite non-negative number."
        )
    return interval


def _is_rate_limit_error(exc: Exception) -> bool:
    rate_limit_type = getattr(yf.exceptions, "YFRateLimitError", ())
    return (
        (bool(rate_limit_type) and isinstance(exc, rate_limit_type))
        or "429" in str(exc)
        or "rate limit" in str(exc).casefold()
    )


def yahoo_call_count() -> int:
    """Return the number of actual calls made through the Yahoo wrapper."""
    with _YAHOO_LOCK:
        return _YAHOO_CALL_COUNT


def _yahoo_call(symbol: str, operation: str, call: Any) -> Any:
    """Serialize, throttle, and safely invoke one Yahoo operation."""
    global _LAST_YAHOO_CALL, _YAHOO_CALL_COUNT
    with _YAHOO_LOCK:
        for attempt in range(_MAX_ATTEMPTS):
            wait = _minimum_interval() - (time.monotonic() - _LAST_YAHOO_CALL)
            if _LAST_YAHOO_CALL and wait > 0:
                time.sleep(wait)
            _LAST_YAHOO_CALL = time.monotonic()
            try:
                with redirect_stdout(sys.stderr):
                    _YAHOO_CALL_COUNT += 1
                    return call()
            except Exception as exc:
                _logger.exception(
                    "Yahoo %s request failed for %s (attempt %d/%d)",
                    operation,
                    symbol,
                    attempt + 1,
                    _MAX_ATTEMPTS,
                )
                if _is_rate_limit_error(exc) and attempt + 1 < _MAX_ATTEMPTS:
                    time.sleep((2**attempt) + random.uniform(0, 0.25))
                    continue
                raise StockDataSourceError(
                    f"Yahoo Finance could not provide {operation} for {symbol}."
                ) from None
    raise StockDataSourceError(
        f"Yahoo Finance could not provide {operation} for {symbol}."
    )


def _calendar_date(value: Any) -> date | None:
    """Convert an index value to a Cairo calendar date."""
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(_cairo_timezone())
    return timestamp.date()


def _finite_numeric(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class YahooProvider:
    """Fetch and normalize delayed daily stock data from Yahoo Finance.

    Calls are serialized and rate-limit responses are retried at most three
    times. History is daily only; Yahoo's intraday endpoints are not used.
    """

    def get_history(self, entry: dict[str, Any], period_days: int) -> pd.DataFrame:
        """Return daily ``date/open/high/low/close/adj_close/volume`` rows.

        ``period_days`` is a calendar-day lookback. The source's split-adjusted
        ``Close`` and dividend-adjusted ``Adj Close`` are both retained.
        """
        if isinstance(period_days, bool) or not isinstance(period_days, int):
            raise ValueError("period_days must be an integer.")
        if period_days < 1:
            raise ValueError("period_days must be positive.")
        ticker = entry["yahoo_ticker"]
        today = datetime.now(_cairo_timezone()).date()
        start = today - timedelta(days=period_days)
        end = today + timedelta(days=1)
        history = _yahoo_call(
            entry["symbol"],
            "history",
            lambda: yf.Ticker(ticker).history(
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                auto_adjust=False,
            ),
        )
        try:
            normalized = _normalize_history(history)
        except EmptyStockDataError:
            raise
        except Exception:
            _logger.exception("Unable to normalize Yahoo history for %s", ticker)
            raise StockDataSourceError(
                f"Yahoo Finance returned unsupported history data for {entry['symbol']}."
            ) from None
        if normalized.empty:
            raise EmptyStockDataError(
                f"No Yahoo daily history is available for {entry['symbol']}."
            )
        cutoff = today - timedelta(days=period_days)
        normalized = normalized.loc[normalized["date"] >= cutoff].reset_index(drop=True)
        if normalized.empty:
            raise EmptyStockDataError(
                f"No Yahoo daily history is available for {entry['symbol']}."
            )
        return normalized

    def get_dividends(self, entry: dict[str, Any], years: int) -> pd.DataFrame:
        """Return full dividend history as sorted ``date`` and ``amount`` rows.

        ``years`` is validated for protocol compatibility; history is not
        truncated here because callers may persist it and filter locally.
        """
        if isinstance(years, bool) or not isinstance(years, int) or years < 1:
            raise ValueError("years must be a positive integer.")
        ticker = entry["yahoo_ticker"]
        dividends = _yahoo_call(
            entry["symbol"],
            "dividends",
            lambda: yf.Ticker(ticker).dividends,
        )
        try:
            normalized = _normalize_dividends(dividends)
        except Exception:
            _logger.exception("Unable to normalize Yahoo dividends for %s", ticker)
            raise StockDataSourceError(
                f"Yahoo Finance returned unsupported dividend data for {entry['symbol']}."
            ) from None
        return normalized

    def get_info_light(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Return available company info and fundamentals for the 24-hour cache."""
        ticker = entry["yahoo_ticker"]
        info = _yahoo_call(
            entry["symbol"],
            "company information",
            lambda: yf.Ticker(ticker).info,
        )
        if not isinstance(info, dict):
            _logger.error("Yahoo returned unsupported company information for %s", ticker)
            raise StockDataSourceError(
                f"Yahoo Finance returned unsupported company information for {entry['symbol']}."
            )
        return {
            "name": info.get("longName"),
            "market_cap": _finite_numeric(info.get("marketCap")),
            "trailing_pe": _finite_numeric(info.get("trailingPE")),
            "forward_pe": _finite_numeric(info.get("forwardPE")),
            "price_to_book": _finite_numeric(info.get("priceToBook")),
            "return_on_equity": _finite_numeric(info.get("returnOnEquity")),
            "profit_margins": _finite_numeric(info.get("profitMargins")),
            "operating_margins": _finite_numeric(info.get("operatingMargins")),
            "dividend_yield": _finite_numeric(info.get("dividendYield")),
            "payout_ratio": _finite_numeric(info.get("payoutRatio")),
            "total_debt": _finite_numeric(info.get("totalDebt")),
            "total_cash": _finite_numeric(info.get("totalCash")),
            "week52_high": _finite_numeric(info.get("fiftyTwoWeekHigh")),
            "week52_low": _finite_numeric(info.get("fiftyTwoWeekLow")),
            "currency": info.get("currency"),
            "financial_currency": info.get("financialCurrency"),
        }


def _normalize_history(history: Any) -> pd.DataFrame:
    """Coerce Yahoo daily history into the stable OHLCV schema."""
    if not isinstance(history, pd.DataFrame):
        raise StockDataSourceError("Yahoo Finance returned an unsupported history format.")
    if history.empty:
        raise EmptyStockDataError("Yahoo Finance returned no daily history.")

    frame = history.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    lookup = {str(column).casefold(): column for column in frame.columns}
    if "close" not in lookup:
        raise StockDataSourceError("Yahoo Finance history has no Close column.")
    dates = [_calendar_date(value) for value in frame.index]
    result = pd.DataFrame({"date": dates})
    for column, yahoo_name in (
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("adj_close", "adj close"),
        ("volume", "volume"),
    ):
        source_column = lookup.get(yahoo_name)
        if source_column is None and column == "adj_close":
            source_column = lookup.get("adj_close")
        if source_column is None:
            result[column] = float("nan")
        else:
            result[column] = pd.to_numeric(frame[source_column], errors="coerce").to_numpy()

    for price_column in ("open", "high", "low", "close", "adj_close"):
        result[price_column] = result[price_column].round(4)
    result = result.dropna(subset=["date", "close"])
    result = result.loc[result["close"].map(lambda value: math.isfinite(float(value)))]
    result = result.sort_values("date").reset_index(drop=True)
    if result.empty:
        raise EmptyStockDataError("Yahoo Finance returned no usable daily history.")
    return result.loc[:, list(_BAR_COLUMNS)]


def _normalize_dividends(dividends: Any) -> pd.DataFrame:
    """Normalize a Yahoo dividend Series or date/amount frame."""
    if isinstance(dividends, pd.Series):
        dates, amounts = dividends.index, dividends.to_numpy()
    elif isinstance(dividends, pd.DataFrame):
        if dividends.empty:
            return pd.DataFrame(columns=["date", "amount"])
        lookup = {str(column).casefold(): column for column in dividends.columns}
        amount_column = lookup.get("amount", lookup.get("dividends"))
        if amount_column is None:
            if len(dividends.columns) != 1:
                raise ValueError("Dividend data is missing an amount column.")
            amount_column = dividends.columns[0]
        dates, amounts = dividends.index, dividends[amount_column].to_numpy()
    else:
        raise ValueError("Yahoo dividend response must be a Series or DataFrame.")
    result = pd.DataFrame(
        {
            "date": [_calendar_date(value) for value in dates],
            "amount": pd.to_numeric(amounts, errors="coerce"),
        }
    ).dropna(subset=["date", "amount"])
    result = result.loc[result["amount"].map(lambda value: math.isfinite(float(value)))]
    result["amount"] = result["amount"].round(4)
    return result.sort_values("date").reset_index(drop=True)
