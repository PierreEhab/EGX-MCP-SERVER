"""FastMCP server exposing Egyptian Exchange index data and analytics."""

from __future__ import annotations

import argparse
import asyncio
from functools import wraps
import logging
import os
import threading
from datetime import date
from typing import Any, Literal, Sequence

import pandas as pd
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings

from . import analytics, egx_client
from .constants import INDEX_DESCRIPTIONS
from .http_app import _is_loopback
from .stock_cache import CachedStockProvider
from .stock_provider import (
    EmptyStockDataError,
    SOURCE as STOCK_SOURCE,
    StockDataError,
    YahooProvider,
)
from .symbols import (
    SymbolNotFoundError,
    SymbolUnavailableError,
    list_symbols as catalog_list_symbols,
    require_yahoo_available,
    resolve_symbol,
    search as catalog_search,
    sectors as catalog_sectors,
)

SOURCE = egx_client.SOURCE
_logger = logging.getLogger(__name__)
_mcp: FastMCP | None = None
_mcp_lock = threading.Lock()
_stock_provider: CachedStockProvider | None = None
_stock_provider_lock = threading.Lock()


class InvalidStockPeriodError(ValueError):
    """The requested stock lookback is not an integer."""

    code = "INVALID_PERIOD"


class AmbiguousStockSymbolError(ValueError):
    """A company name matches multiple non-variant catalog entries."""

    def __init__(self, candidates: list[str]) -> None:
        self.candidates = candidates
        super().__init__(f"Ambiguous company name; candidates: {', '.join(candidates)}")


def _get_stock_provider() -> CachedStockProvider:
    """Lazily construct the process-wide Yahoo provider and disk cache."""
    global _stock_provider
    if _stock_provider is None:
        with _stock_provider_lock:
            if _stock_provider is None:
                _stock_provider = CachedStockProvider(YahooProvider())
    return _stock_provider


def _clamp_stock_period(period_days: int) -> int:
    if isinstance(period_days, bool) or not isinstance(period_days, int):
        raise InvalidStockPeriodError("period_days must be an integer.")
    return max(1, min(9000, period_days))


def _stock_error(
    code: str,
    message: str,
    symbol: str | list[str] | None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "as_of": None,
        "source": STOCK_SOURCE,
        **extra,
        "error": {"code": code, "message": message},
    }


def _stock_exception_error(
    exc: Exception,
    symbol: str | list[str] | None,
    **extra: Any,
) -> dict[str, Any]:
    if isinstance(exc, SymbolNotFoundError):
        code = "SYMBOL_NOT_FOUND"
        extra.setdefault("suggestions", exc.suggestions)
    elif isinstance(exc, AmbiguousStockSymbolError):
        code = "AMBIGUOUS_SYMBOL"
        extra.setdefault("candidates", exc.candidates)
    elif isinstance(exc, SymbolUnavailableError):
        code = "SYMBOL_UNAVAILABLE"
    elif isinstance(exc, EmptyStockDataError):
        code = "EMPTY_DATA"
    elif isinstance(exc, InvalidStockPeriodError):
        code = exc.code
    elif isinstance(exc, StockDataError):
        code = exc.code
    else:
        code = "DATA_SOURCE_ERROR"
    return _stock_error(code, str(exc), symbol, **extra)


def _maybe_raise_stock_error(payload: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("EGX_RAISE_TOOL_ERRORS", "").casefold() == "true":
        error = payload.get("error")
        if error:
            raise ToolError(error["message"])
    return payload


def _unexpected_stock_error(
    tool_name: str,
    symbol: str | list[str] | None,
    exc: Exception,
) -> dict[str, Any]:
    _logger.exception("Unexpected failure in stock tool %s", tool_name)
    payload = _stock_error(
        "DATA_SOURCE_ERROR",
        "An unexpected stock data error occurred.",
        symbol,
    )
    return _maybe_raise_stock_error(payload)


def _guard_stock_tool(function):
    """Map unexpected stock-tool failures to safe, structured errors."""
    @wraps(function)
    async def guarded(*args, **kwargs):
        symbol = kwargs.get("symbol")
        if symbol is None and args:
            symbol = args[0] if isinstance(args[0], (str, list)) else None
        try:
            return await function(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            return _unexpected_stock_error(function.__name__, symbol, exc)

    return guarded


def _resolve_stock(text: str) -> dict[str, Any]:
    resolved = resolve_symbol(text)
    if isinstance(resolved, list):
        raise AmbiguousStockSymbolError([entry["symbol"] for entry in resolved])
    require_yahoo_available(resolved)
    return resolved


def _stale_metadata(as_of: str | None) -> dict[str, Any]:
    stale = False
    if as_of:
        cairo_today = pd.Timestamp.now(tz=egx_client._cairo_timezone()).date()
        stale = (cairo_today - date.fromisoformat(as_of)).days > 30
    return {
        "stale": stale,
        "warning": (
            "Yahoo data is stale (last available bar is more than 30 days old)."
            if stale
            else None
        ),
    }


def _catalog_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    return {"catalog_checked_on": entry.get("catalog_checked_on")}


def _catalog_row(entry: dict[str, Any], verbose: bool) -> dict[str, Any]:
    metadata = _catalog_metadata(entry)
    if verbose:
        return {**entry, **metadata}
    fields = ("symbol", "name_en", "sector", "sector_source", "yahoo_available")
    row = {field: entry[field] for field in fields}
    row.update(metadata)
    if "unavailable" in entry:
        row["unavailable"] = entry["unavailable"]
    return row


def _stock_rows(
    frame: pd.DataFrame,
    frequency: Literal["daily", "weekly", "monthly"],
    max_rows: int | None,
) -> list[dict[str, Any]]:
    if max_rows is not None and (
        isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1
    ):
        raise ValueError("max_rows must be a positive integer or None.")
    observations = frame.copy()
    observations["date"] = pd.to_datetime(observations["date"], errors="coerce")
    observations = observations.dropna(subset=["date"]).sort_values("date")
    if frequency != "daily" and not observations.empty:
        periods = observations["date"].dt.to_period(
            "W-THU" if frequency == "weekly" else "M"
        )
        observations = (
            observations.assign(_period=periods)
            .groupby("_period", sort=True)
            .tail(1)
            .drop(columns="_period")
        )
    observations = observations.reset_index(drop=True)
    if max_rows is not None:
        observations = observations.tail(max_rows).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for row in observations.to_dict("records"):
        rows.append(
            {
                key: (
                    pd.Timestamp(value).date().isoformat()
                    if key == "date" and pd.notna(value)
                    else (float(value) if pd.notna(value) else None)
                    if key != "date"
                    else None
                )
                for key, value in row.items()
            }
        )
    return rows


def _load_stock_sync(
    symbol: str,
    period_days: int,
) -> tuple[dict[str, Any], int, pd.DataFrame]:
    entry = _resolve_stock(symbol)
    period = _clamp_stock_period(period_days)
    bars = _get_stock_provider().get_history(entry, period)
    return entry, period, bars


async def _load_stock(
    symbol: str,
    period_days: int,
) -> tuple[dict[str, Any], int, pd.DataFrame] | dict[str, Any]:
    try:
        return await asyncio.to_thread(_load_stock_sync, symbol, period_days)
    except (
        SymbolNotFoundError,
        SymbolUnavailableError,
        AmbiguousStockSymbolError,
        StockDataError,
        InvalidStockPeriodError,
    ) as exc:
        return _stock_exception_error(exc, symbol)
    except ValueError as exc:
        return _stock_error("INVALID_ARGUMENT", str(exc), symbol)
    except Exception as exc:
        return _unexpected_stock_error("_load_stock", symbol, exc)


@_guard_stock_tool
async def list_stocks(
    sector: str | None = None,
    limit: int = 50,
    include_unavailable: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """List EGX stocks, filtered by catalog sector and available Yahoo data.

    Unavailable listings are hidden unless ``include_unavailable`` is true.
    Set ``verbose`` to include the full catalog record. Inferred sectors are
    proposals, not confirmed classifications; ``sector_source`` is included.
    """
    if not isinstance(verbose, bool):
        payload = _stock_error(
            "INVALID_ARGUMENT", "verbose must be a boolean.", None
        )
        return _maybe_raise_stock_error(payload)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        payload = _stock_error(
            "INVALID_ARGUMENT", "limit must be a positive integer.", None
        )
        return _maybe_raise_stock_error(payload)
    if sector is not None and sector.casefold() not in {
        value.casefold() for value in catalog_sectors()
    }:
        payload = _stock_error("INVALID_ARGUMENT", "Unknown sector.", None)
        return _maybe_raise_stock_error(payload)
    try:
        entries = await asyncio.to_thread(
            catalog_list_symbols,
            sector,
            False,
            True,
            include_unavailable,
        )
    except (TypeError, ValueError) as exc:
        payload = _stock_error("INVALID_ARGUMENT", str(exc), None)
        return _maybe_raise_stock_error(payload)
    stocks = [_catalog_row(entry, verbose) for entry in entries[:limit]]
    return {
        "symbol": None,
        "as_of": None,
        "source": "EGX symbol catalog",
        "stocks": stocks,
        "observations": len(stocks),
        "sectors": catalog_sectors(),
    }


@_guard_stock_tool
async def find_stock(query: str, verbose: bool = False) -> dict[str, Any]:
    """Search available EGX companies; ``verbose`` returns full catalog rows."""
    if not isinstance(verbose, bool):
        payload = _stock_error(
            "INVALID_ARGUMENT", "verbose must be a boolean.", query
        )
        return _maybe_raise_stock_error(payload)
    try:
        matches = await asyncio.to_thread(catalog_search, query, 50)
    except (TypeError, ValueError) as exc:
        payload = _stock_error("INVALID_ARGUMENT", str(exc), query)
        return _maybe_raise_stock_error(payload)
    stocks = [
        _catalog_row(entry, verbose)
        for entry in matches
        if entry["yahoo_available"] is not False
    ]
    return {
        "symbol": [entry["symbol"] for entry in stocks],
        "as_of": None,
        "source": "EGX symbol catalog",
        "stocks": stocks,
        "observations": len(stocks),
    }


@_guard_stock_tool
async def get_stock_history(
    symbol: str,
    period_days: int = 90,
    frequency: Literal["daily", "weekly", "monthly"] = "daily",
    max_rows: int | None = None,
    adjusted: bool = False,
) -> dict[str, Any]:
    """Return daily or downsampled Yahoo stock OHLCV rows.

    ``period_days`` is clamped to 1-9000 calendar days. Weekly bars use
    Thursday-based periods. ``Close`` is split-adjusted; when ``adjusted`` is
    true the row's ``close`` is ``Adj Close`` (dividend-adjusted).
    """
    if frequency not in ("daily", "weekly", "monthly"):
        payload = _stock_error(
            "INVALID_ARGUMENT",
            "frequency must be 'daily', 'weekly', or 'monthly'.",
            symbol,
        )
        return _maybe_raise_stock_error(payload)
    if not isinstance(adjusted, bool):
        payload = _stock_error("INVALID_ARGUMENT", "adjusted must be a boolean.", symbol)
        return _maybe_raise_stock_error(payload)
    loaded = await _load_stock(symbol, period_days)
    if isinstance(loaded, dict):
        return _maybe_raise_stock_error(loaded)
    entry, period, bars = loaded
    frame = bars.copy()
    if adjusted:
        frame["close"] = frame["adj_close"]
    try:
        rows = await asyncio.to_thread(_stock_rows, frame, frequency, max_rows)
    except ValueError as exc:
        payload = _stock_error("INVALID_ARGUMENT", str(exc), entry["symbol"])
        return _maybe_raise_stock_error(payload)
    return _maybe_raise_stock_error(
        {
            "symbol": entry["symbol"],
            "as_of": rows[-1]["date"] if rows else None,
            "source": STOCK_SOURCE,
            "rows": rows,
            "observations": len(rows),
            "period_days_applied": period,
            "frequency": frequency,
            "price_basis": "Adj Close (dividend-adjusted)"
            if adjusted
            else "Close (split-adjusted)",
            **_stale_metadata(rows[-1]["date"] if rows else None),
        }
    )


@_guard_stock_tool
async def get_stock_summary(
    symbol: str,
    period_days: int = 365,
) -> dict[str, Any]:
    """Summarize recent stock price, total return, and risk metrics."""
    loaded = await _load_stock(symbol, period_days)
    if isinstance(loaded, dict):
        return _maybe_raise_stock_error(loaded)
    entry, period, bars = loaded
    try:
        dividends = await asyncio.to_thread(
            _get_stock_provider().get_dividends, entry, 1
        )
    except StockDataError as exc:
        dividends = pd.DataFrame(columns=["date", "amount"])
        quality = f"Dividend history unavailable: {exc.code}."
    else:
        quality = None
    summary = await asyncio.to_thread(
        analytics.stock_summary,
        bars,
        dividends,
        period,
    )
    return _maybe_raise_stock_error(
        {
            "symbol": entry["symbol"],
            **summary,
            "source": STOCK_SOURCE,
            "period_days_applied": period,
            "observations": summary["observations"],
            "price_basis": "Close (split-adjusted); Adj Close for total return",
            "data_quality": quality,
            **_stale_metadata(summary["as_of"]),
        }
    )


@_guard_stock_tool
async def get_stock_fundamentals(symbol: str) -> dict[str, Any]:
    """Return best-effort catalog sector, Yahoo fundamentals, and computed metrics."""
    try:
        entry = await asyncio.to_thread(_resolve_stock, symbol)
    except (
        SymbolNotFoundError,
        SymbolUnavailableError,
        AmbiguousStockSymbolError,
    ) as exc:
        return _maybe_raise_stock_error(_stock_exception_error(exc, symbol))
    provider = _get_stock_provider()
    quality_notes: list[str] = []
    try:
        info = await asyncio.to_thread(provider.get_info_light, entry)
    except StockDataError as exc:
        info = {
            "name": None,
            "market_cap": None,
            "trailing_pe": None,
            "week52_high": None,
            "week52_low": None,
            "currency": None,
        }
        quality_notes.append(f"Yahoo company information unavailable ({exc.code}).")
    try:
        bars = await asyncio.to_thread(provider.get_history, entry, 365)
        summary = await asyncio.to_thread(
            analytics.stock_summary,
            bars,
            pd.DataFrame(columns=["date", "amount"]),
            365,
        )
        average_volume = summary["average_volume_20d"]
        metric_as_of = summary["as_of"]
        metric_observations = summary["observations"]
    except StockDataError as exc:
        bars = None
        average_volume = None
        metric_as_of = None
        metric_observations = 0
        quality_notes.append(f"Computed history metrics unavailable ({exc.code}).")
    try:
        dividend_rows = await asyncio.to_thread(provider.get_dividends, entry, 1)
        if bars is None:
            dividend_yield = None
        else:
            dividend_summary = await asyncio.to_thread(
                analytics.stock_summary,
                bars,
                dividend_rows,
                365,
            )
            dividend_yield = dividend_summary["dividend_yield_ttm_pct"]
    except StockDataError as exc:
        dividend_yield = None
        quality_notes.append(f"Dividend history unavailable ({exc.code}).")
    unavailable_fields = [
        key
        for key, value in info.items()
        if value is None
    ]
    if dividend_yield is None:
        unavailable_fields.append("dividend_yield_ttm")
    if average_volume is None:
        unavailable_fields.append("avg_volume_20d")
    if unavailable_fields:
        quality_notes.append(
            "Unavailable fields: " + ", ".join(dict.fromkeys(unavailable_fields)) + "."
        )
    return _maybe_raise_stock_error(
        {
            "symbol": entry["symbol"],
            "as_of": metric_as_of,
            "source": STOCK_SOURCE,
            "name": info["name"] or entry["name_en"],
            "sector": entry["sector"],
            "sector_source": entry["sector_source"],
            "market_cap": info["market_cap"],
            "trailing_pe": info["trailing_pe"],
            "week52_high": info["week52_high"],
            "week52_low": info["week52_low"],
            "currency": info["currency"],
            "dividend_yield_ttm": dividend_yield,
            "avg_volume_20d": average_volume,
            "data_quality": "; ".join(quality_notes) if quality_notes else None,
            "observations": metric_observations,
            "period_days_applied": 365,
            **_stale_metadata(metric_as_of),
        }
    )


@_guard_stock_tool
async def get_dividends(symbol: str, years: int = 5) -> dict[str, Any]:
    """Return Yahoo cash dividends filtered to 1-50 requested years."""
    if isinstance(years, bool) or not isinstance(years, int):
        payload = _stock_error(
            "INVALID_ARGUMENT", "years must be an integer.", symbol
        )
        return _maybe_raise_stock_error(payload)
    years_applied = max(1, min(50, years))
    try:
        entry = await asyncio.to_thread(_resolve_stock, symbol)
        dividends = await asyncio.to_thread(
            _get_stock_provider().get_dividends,
            entry,
            years_applied,
        )
    except (
        SymbolNotFoundError,
        SymbolUnavailableError,
        AmbiguousStockSymbolError,
        StockDataError,
    ) as exc:
        return _maybe_raise_stock_error(_stock_exception_error(exc, symbol))
    rows = [
        {"date": pd.Timestamp(row.date).date().isoformat(), "amount": float(row.amount)}
        for row in dividends.itertuples(index=False)
    ]
    return _maybe_raise_stock_error(
        {
            "symbol": entry["symbol"],
            "as_of": rows[-1]["date"] if rows else None,
            "source": STOCK_SOURCE,
            "rows": rows,
            "observations": len(rows),
            "years": years_applied,
            "years_applied": years_applied,
        }
    )


@_guard_stock_tool
async def compare_stocks(
    symbols: list[str],
    period_days: int = 365,
) -> dict[str, Any]:
    """Rank up to ten unique EGX companies by adjusted total return.

    Results retain per-symbol errors so one failed source request does not
    discard other comparisons.
    """
    if not isinstance(symbols, list) or not symbols:
        return _maybe_raise_stock_error(
            _stock_error("INVALID_ARGUMENT", "symbols must be a non-empty list.", [])
        )
    if any(not isinstance(symbol, str) for symbol in symbols):
        return _maybe_raise_stock_error(
            _stock_error("INVALID_ARGUMENT", "every symbol must be a string.", symbols)
        )
    unique: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        key = symbol.strip().casefold()
        if key not in seen:
            seen.add(key)
            unique.append(symbol)
    if len(unique) > 10:
        return _maybe_raise_stock_error(
            _stock_error(
                "INVALID_ARGUMENT",
                "At most 10 unique symbols may be compared.",
                unique,
            )
        )
    try:
        period = _clamp_stock_period(period_days)
    except InvalidStockPeriodError as exc:
        return _maybe_raise_stock_error(_stock_exception_error(exc, unique))
    results = await asyncio.gather(*(_load_stock(symbol, period) for symbol in unique))
    rows: list[dict[str, Any]] = []
    for requested, result in zip(unique, results):
        if isinstance(result, dict):
            rows.append(
                {
                    "symbol": requested,
                    "period_days_applied": period,
                    "observations": 0,
                    "total_return_pct": None,
                    "drawdown_from_period_high_pct": None,
                    "annualized_volatility_pct": None,
                    "as_of": None,
                    "stale": None,
                    "warning": None,
                    "error": result["error"],
                }
            )
            continue
        entry, applied_period, bars = result
        summary = await asyncio.to_thread(
            analytics.stock_summary,
            bars,
            pd.DataFrame(columns=["date", "amount"]),
            applied_period,
        )
        rows.append(
            {
                "symbol": entry["symbol"],
                "period_days_applied": applied_period,
                "observations": summary["observations"],
                "total_return_pct": summary["total_return_pct"],
                "drawdown_from_period_high_pct": summary[
                    "drawdown_from_period_high_pct"
                ],
                "annualized_volatility_pct": summary["annualized_volatility_pct"],
                "as_of": summary["as_of"],
                **_stale_metadata(summary["as_of"]),
            }
        )
    rows.sort(
        key=lambda row: (
            row["total_return_pct"] is None,
            -(row["total_return_pct"] or 0.0),
        )
    )
    if (
        os.getenv("EGX_RAISE_TOOL_ERRORS", "").casefold() == "true"
        and rows
        and all("error" in row for row in rows)
    ):
        raise ToolError(rows[0]["error"]["message"])
    return {
        "symbol": [row["symbol"] for row in rows],
        "as_of": max((row["as_of"] for row in rows if row["as_of"]), default=None),
        "source": STOCK_SOURCE,
        "comparisons": rows,
        "period_days_applied": period,
        "observations": sum(row["observations"] for row in rows),
    }


@_guard_stock_tool
async def stock_vs_index(
    symbol: str,
    index: str = "EGX30",
    period_days: int = 365,
) -> dict[str, Any]:
    """Compare a stock with an EGX index using common-date daily returns."""
    loaded_stock, loaded_index = await asyncio.gather(
        _load_stock(symbol, period_days),
        _load(index, period_days),
    )
    if isinstance(loaded_stock, dict):
        return _maybe_raise_stock_error(loaded_stock)
    if isinstance(loaded_index, dict):
        error = loaded_index["error"]
        payload = _stock_error(error["code"], error["message"], symbol, index=index)
        return _maybe_raise_stock_error(payload)
    entry, period, bars = loaded_stock
    canonical_index, _, index_frame = loaded_index
    stock_prices = bars.loc[:, ["date", "close", "adj_close"]].copy()
    stock_prices["date"] = pd.to_datetime(stock_prices["date"], errors="coerce").dt.date
    index_prices = index_frame.loc[:, ["datetime", "value"]].rename(
        columns={"datetime": "date", "value": "index_close"}
    )
    index_prices["date"] = pd.to_datetime(
        index_prices["date"], errors="coerce"
    ).dt.date
    common = stock_prices.merge(index_prices, on="date", how="inner").sort_values("date")
    common = common.dropna(subset=["close", "index_close"])

    def return_pct(column: str) -> float | None:
        values = pd.to_numeric(common[column], errors="coerce").dropna()
        if values.empty or values.iloc[0] == 0:
            return None
        return float((values.iloc[-1] / values.iloc[0] - 1) * 100)

    stock_price_return = return_pct("close")
    index_price_return = return_pct("index_close")
    stock_total_return = return_pct("adj_close")
    excess_return = (
        stock_price_return - index_price_return
        if stock_price_return is not None and index_price_return is not None
        else None
    )
    stock_relative = await asyncio.to_thread(
        analytics.relative_performance,
        common.loc[:, ["date", "close"]],
        common.loc[:, ["date", "index_close"]].rename(
            columns={"index_close": "close"}
        ),
    )
    summary = await asyncio.to_thread(
        analytics.stock_summary,
        bars,
        pd.DataFrame(columns=["date", "amount"]),
        period,
    )
    return _maybe_raise_stock_error(
        {
            "symbol": entry["symbol"],
            "index": canonical_index,
            "as_of": (
                common["date"].iloc[-1].isoformat() if not common.empty else None
            ),
            "source": f"{STOCK_SOURCE}; {SOURCE}",
            "relative_performance": stock_relative,
            "stock_price_return_pct": stock_price_return,
            "index_price_return_pct": index_price_return,
            "excess_return_pct": excess_return,
            "stock_total_return_pct": stock_total_return,
            "index_total_return_pct": None,
            "stock_return_pct": stock_price_return,
            "index_return_pct": index_price_return,
            "period_days_applied": period,
            "observations": len(common),
            "price_basis": (
                "Stock price returns use Close (split-adjusted); stock total "
                "return uses Adj Close. Index price return uses EGX daily values."
            ),
            "index_total_return_note": (
                "Index total-return history was not requested; unavailable."
            ),
            **_stale_metadata(summary["as_of"]),
        }
    )


@_guard_stock_tool
async def analysis_snapshot(symbol: str) -> dict[str, Any]:
    """Return a compact summary, fundamentals, index-relative view, and key levels."""
    loaded, index_result = await asyncio.gather(
        _load_stock(symbol, 365),
        _load("EGX30", 365),
    )
    if isinstance(loaded, dict):
        return _maybe_raise_stock_error(loaded)
    if isinstance(index_result, dict):
        error = index_result["error"]
        return _maybe_raise_stock_error(
            _stock_error(error["code"], error["message"], symbol, index="EGX30")
        )
    entry, period, bars = loaded
    provider = _get_stock_provider()
    info_result, dividends_result = await asyncio.gather(
        asyncio.to_thread(provider.get_info_light, entry),
        asyncio.to_thread(provider.get_dividends, entry, 1),
        return_exceptions=True,
    )
    quality_notes: list[str] = []
    if isinstance(info_result, Exception):
        if not isinstance(info_result, StockDataError):
            raise info_result
        quality_notes.append(
            f"Yahoo company information unavailable ({info_result.code})."
        )
        info = {
            "name": None,
            "market_cap": None,
            "trailing_pe": None,
            "week52_high": None,
            "week52_low": None,
            "currency": None,
        }
    else:
        info = info_result
    if isinstance(dividends_result, Exception):
        if not isinstance(dividends_result, StockDataError):
            raise dividends_result
        quality_notes.append(
            f"Dividend history unavailable ({dividends_result.code})."
        )
        dividends = pd.DataFrame(columns=["date", "amount"])
    else:
        dividends = dividends_result
    summary = await asyncio.to_thread(analytics.stock_summary, bars, dividends, period)
    unavailable_fields = [
        field for field, value in info.items() if value is None
    ]
    if summary["dividend_yield_ttm_pct"] is None:
        unavailable_fields.append("dividend_yield_ttm")
    if summary["average_volume_20d"] is None:
        unavailable_fields.append("avg_volume_20d")
    if unavailable_fields:
        quality_notes.append(
            "Unavailable fields: " + ", ".join(dict.fromkeys(unavailable_fields)) + "."
        )
    fundamentals = {
        "sector": entry["sector"],
        "sector_source": entry["sector_source"],
        "market_cap": info["market_cap"],
        "trailing_pe": info["trailing_pe"],
        "dividend_yield_ttm": summary["dividend_yield_ttm_pct"],
        "avg_volume_20d": summary["average_volume_20d"],
    }
    canonical_index, _, index_bars = index_result
    relative_performance = await asyncio.to_thread(
        analytics.relative_performance,
        bars.loc[:, ["date", "close"]],
        index_bars.loc[:, ["datetime", "value"]].rename(
            columns={"datetime": "date", "value": "close"}
        ),
    )
    levels = await asyncio.to_thread(analytics.key_levels, bars)
    dividend_rows = [
        {
            "date": pd.Timestamp(row.date).date().isoformat(),
            "amount": float(row.amount),
        }
        for row in dividends.itertuples(index=False)
    ]
    return {
        "symbol": entry["symbol"],
        "as_of": summary["as_of"],
        "source": STOCK_SOURCE,
        "summary": {
            key: summary.get(key)
            for key in (
                "latest_close",
                "daily_change_pct",
                "price_return_pct",
                "total_return_pct",
                "annualized_volatility_pct",
                "max_drawdown_pct",
            )
        },
        "relative_performance": relative_performance,
        "fundamentals": fundamentals,
        "dividends": {
            "observations": len(dividend_rows),
            "rows": dividend_rows[-5:],
        },
        "key_levels": levels,
        "data_quality": [
            value
            for value in (
                *quality_notes,
                (
                    "Yahoo data is stale (last available bar is more than "
                    "30 days old)."
                    if _stale_metadata(summary["as_of"])["stale"]
                    else None
                ),
            )
            if value
        ],
    }


def research_company(symbol: str) -> str:
    """Prompt a structured EGX company research report using stock tools."""
    return f"""Research the EGX-listed company {symbol}.

First call analysis_snapshot for {symbol}, get_stock_fundamentals for {symbol},
and stock_vs_index for {symbol} against EGX30. Use the returned data and dates;
do not infer facts that the tools do not provide. Then write a structured
report with these sections:
1. Trend: summarize price direction, returns, moving averages, and key levels.
2. Risk: discuss volatility, drawdown, liquidity caveats, and relative
   performance against EGX30.
3. Valuation caveats: report available fundamentals and explain any missing
   valuation inputs; do not treat sector estimates as confirmed.
4. Missing or stale data: identify unavailable fields, source errors, and
   stale observations explicitly, without guessing.

This is analysis, not financial advice. Yahoo data is delayed. Close is
split-adjusted. EGX has daily price limits and thin liquidity for many names.
Stale or missing data must be reported, not guessed."""


def build_trade_plan(symbol: str, horizon: str, risk_tolerance: str) -> str:
    """Prompt scenario planning based on EGX tools and explicit risk caveats."""
    return f"""Prepare an educational scenario plan for EGX company {symbol}.
The user's stated horizon is {horizon}; stated risk tolerance is
{risk_tolerance}.

First call analysis_snapshot for {symbol}, get_stock_fundamentals for {symbol},
get_stock_history for an appropriate period and frequency, and
stock_vs_index for {symbol} against EGX30. Ground every price level in tool
output and cite its as_of date. Then draft:
1. Bull, base, and bear scenarios, with evidence and conditions that would
   support each scenario.
2. Relevant support, resistance, and other key levels; distinguish observed
   levels from interpretation.
3. Position-sizing considerations tied to the stated risk tolerance, without
   prescribing a personal allocation.
4. Invalidation conditions that would weaken or invalidate each scenario.
5. Data limitations, including any missing fields, source errors, or stale
   observations; do not guess unavailable values.

This is analysis, not financial advice. Yahoo data is delayed. Close is
split-adjusted. EGX has daily price limits and thin liquidity for many names.
Stale or missing data must be reported, not guessed."""


def _transport_security(host: str) -> TransportSecuritySettings | None:
    configured_hosts = os.getenv("MCP_ALLOWED_HOSTS")
    token = os.getenv("MCP_AUTH_TOKEN")
    if configured_hosts is not None:
        allowed_hosts = [
            entry.strip() for entry in configured_hosts.split(",") if entry.strip()
        ]
        if not allowed_hosts:
            raise ValueError("MCP_ALLOWED_HOSTS must contain at least one host.")
        allowed_origins: list[str] = []
        for allowed_host in allowed_hosts:
            exact_host = (
                allowed_host[:-2]
                if allowed_host.endswith(":*")
                else allowed_host
            )
            for scheme in ("http", "https"):
                allowed_origins.append(f"{scheme}://{exact_host}")
                if allowed_host.endswith(":*"):
                    allowed_origins.append(f"{scheme}://{allowed_host}")
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts
            + [
                entry[:-2]
                for entry in allowed_hosts
                if entry.endswith(":*")
            ],
            allowed_origins=allowed_origins,
        )
    if token and not _is_loopback(host):
        _logger.warning(
            "MCP_AUTH_TOKEN is set for a non-loopback bind; disabling DNS-rebinding protection because bearer auth is enabled and MCP_ALLOWED_HOSTS is unset."
        )
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
            allowed_hosts=[],
            allowed_origins=[],
        )
    if _is_loopback(host):
        return None
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            f"{host}:*",
            "localhost:*",
            "localhost",
            "127.0.0.1:*",
            "127.0.0.1",
            "[::1]:*",
            "[::1]",
        ],
        allowed_origins=[
            "http://localhost:*",
            "http://localhost",
            "https://localhost:*",
            "https://localhost",
            "http://127.0.0.1:*",
            "http://127.0.0.1",
            "https://127.0.0.1:*",
            "https://127.0.0.1",
            "http://[::1]:*",
            "http://[::1]",
            "https://[::1]:*",
            "https://[::1]",
        ],
    )


def _new_mcp(host: str, port: int) -> FastMCP:
    settings = _transport_security(host)
    if settings is not None:
        _logger.info(
            "HTTP allowed hosts: %s",
            settings.allowed_hosts or "(disabled)",
        )
    return FastMCP(
        "egx-mcp-server",
        instructions=(
            "Retrieve Egyptian Exchange index data and individual stocks "
            "through Yahoo Finance (delayed daily bars), use the analysis tools, "
            "and follow the research prompts for structured company research "
            "or scenario planning. Results are analysis, not financial advice."
        ),
        host=host,
        port=port,
        transport_security=settings,
    )


def _error(
    error_code: str,
    message: str,
    index: str | list[str] | None = None,
) -> dict[str, Any]:
    return {
        "index": index,
        "as_of": None,
        "source": SOURCE,
        "error": {"code": error_code, "message": message},
    }


def _return_error(
    error_code: str,
    message: str,
    index: str | list[str] | None = None,
) -> dict[str, Any]:
    return _error(error_code, message, index)


def _maybe_raise_tool_error(payload: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("EGX_RAISE_TOOL_ERRORS", "").casefold() == "true":
        error = payload.get("error")
        if error:
            raise ToolError(error["message"])
    return payload


def _error_for(
    exc: egx_client.EGXClientError,
    index: str | list[str] | None,
) -> dict[str, Any]:
    if isinstance(exc, egx_client.InvalidIndexError):
        code = "INVALID_INDEX"
    elif isinstance(exc, egx_client.EmptyDataError):
        code = "EMPTY_DATA"
    elif isinstance(exc, egx_client.EGXNetworkError):
        code = "NETWORK_ERROR"
    elif isinstance(exc, egx_client.InvalidPeriodError):
        code = "INVALID_PERIOD"
    else:
        code = "DATA_SOURCE_ERROR"
    return _error(code, str(exc), index)


def _load_sync(index: str, period_days: int) -> tuple[str, int, pd.DataFrame]:
    canonical = egx_client.resolve_index(index)
    period = egx_client.clamp_period_days(period_days)
    frame = egx_client.fetch_index_data(canonical, period)
    return canonical, period, frame


async def _load(
    index: str,
    period_days: int,
) -> tuple[str, int, pd.DataFrame] | dict[str, Any]:
    """Resolve and load an index off the event loop, returning safe tool errors."""
    try:
        return await asyncio.to_thread(_load_sync, index, period_days)
    except egx_client.EGXClientError as exc:
        return _error_for(exc, index)


def _sample_history(
    frame: pd.DataFrame,
    frequency: Literal["daily", "weekly", "monthly"],
    max_rows: int | None,
) -> pd.DataFrame:
    if max_rows is not None and (
        isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1
    ):
        raise ValueError("max_rows must be a positive integer or None.")
    observations = frame.copy(deep=False).sort_values("datetime")
    if frequency != "daily":
        key = "W-THU" if frequency == "weekly" else "M"
        periods = observations["datetime"].dt.to_period(key)
        observations = (
            observations.assign(_period=periods)
            .groupby("_period", sort=True)
            .tail(1)
            .drop(columns="_period")
        )
    observations = observations.reset_index(drop=True)
    if max_rows is not None:
        observations = observations.tail(max_rows).reset_index(drop=True)
    return observations


def list_indices() -> dict[str, Any]:
    """List the index codes supported by the installed EGX data package.

    Returns canonical index codes and a one-line description for each. Use
    these codes with the history, summary, and comparison tools.
    """
    codes = egx_client.supported_indices()
    indices = [
        {
            "code": code,
            "description": INDEX_DESCRIPTIONS.get(
                code, f"{code} index published by the Egyptian Exchange."
            ),
        }
        for code in codes
    ]
    return {"index": None, "as_of": None, "source": SOURCE, "indices": indices}


async def get_index_history(
    index: str,
    period_days: int = 30,
    frequency: Literal["daily", "weekly", "monthly"] = "daily",
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Return daily or downsampled historical EGX index values as ISO-date rows.

    ``index`` is matched case-insensitively and ``period_days`` is clamped to
    1-1825 calendar days. ``frequency`` keeps the last observation per day,
    week, or month; choose weekly/monthly for long periods to reduce payload
    size. ``max_rows`` optionally keeps only the most recent positive number of
    rows. The source provides ``datetime`` and ``value``, not OHLC fields.
    """
    if frequency not in ("daily", "weekly", "monthly"):
        payload = _return_error(
            "INVALID_ARGUMENT",
            "frequency must be 'daily', 'weekly', or 'monthly'.",
            index,
        )
        return _maybe_raise_tool_error(payload)
    loaded = await _load(index, period_days)
    if isinstance(loaded, dict):
        return _maybe_raise_tool_error(loaded)
    canonical, period, frame = loaded
    try:
        sampled = _sample_history(frame, frequency, max_rows)
    except ValueError as exc:
        payload = _return_error("INVALID_ARGUMENT", str(exc), canonical)
        return _maybe_raise_tool_error(payload)
    rows = analytics.history_rows(sampled)
    as_of = rows[-1]["date"] if rows else None
    payload = {
        "index": canonical,
        "as_of": as_of,
        "source": SOURCE,
        "rows": rows,
        "period_days_applied": period,
        "observations": len(rows),
    }
    return _maybe_raise_tool_error(payload)


async def get_index_summary(index: str, period_days: int = 90) -> dict[str, Any]:
    """Calculate recent performance and risk statistics for one EGX index.

    Returns latest value, previous close, daily move, period high/low and
    return, 20/50-observation simple moving averages when enough data exists,
    annualized volatility, drawdown from the period high, return over the last
    five sessions when available, and the valid observation date range.
    ``period_days`` is a calendar-day lookback clamped to 1-1825.
    """
    loaded = await _load(index, period_days)
    if isinstance(loaded, dict):
        return _maybe_raise_tool_error(loaded)
    canonical, period, frame = loaded
    summary = analytics.summarize(frame)
    payload = {
        "index": canonical,
        **summary,
        "source": SOURCE,
        "period_days_applied": period,
        "observations": len(frame),
    }
    return _maybe_raise_tool_error(payload)


async def compare_indices(
    indices: list[str],
    period_days: int = 90,
) -> dict[str, Any]:
    """Compare period return, volatility, and drawdown for EGX indices.

    Index codes are case-insensitive; duplicates are removed while preserving
    input order. The response includes each index's applied calendar-day
    period and observation count, per-index errors without discarding
    successful comparisons, and the best performer by period return.
    """
    if not isinstance(indices, list) or not indices:
        payload = _return_error(
            "INVALID_ARGUMENT",
            "indices must be a non-empty list.",
            [],
        )
        return _maybe_raise_tool_error(payload)
    if any(not isinstance(index, str) for index in indices):
        payload = _return_error(
            "INVALID_ARGUMENT",
            "every index code must be a string.",
            indices,
        )
        return _maybe_raise_tool_error(payload)

    unique_indices: list[str] = []
    seen: set[str] = set()
    for index in indices:
        key = index.strip().casefold()
        if key not in seen:
            seen.add(key)
            unique_indices.append(index)

    try:
        period = egx_client.clamp_period_days(period_days)
    except egx_client.InvalidPeriodError as exc:
        payload = _error_for(exc, unique_indices)
        return _maybe_raise_tool_error(payload)
    if len(unique_indices) > len(egx_client.supported_indices()):
        payload = _return_error(
            "INVALID_ARGUMENT",
            "The number of unique requested indices exceeds the supported index count.",
            unique_indices,
        )
        return _maybe_raise_tool_error(payload)

    loaded_results = await asyncio.gather(
        *(_load(index, period) for index in unique_indices)
    )
    comparisons: list[dict[str, Any]] = []
    for requested_index, loaded in zip(unique_indices, loaded_results):
        if isinstance(loaded, dict):
            comparisons.append(
                {
                    "index": requested_index,
                    "period_days_applied": period,
                    "observations": 0,
                    "period_return_pct": None,
                    "annualized_volatility_pct": None,
                    "drawdown_from_period_high_pct": None,
                    "as_of": None,
                    "error": loaded["error"],
                }
            )
            continue

        canonical, applied_period, frame = loaded
        summary = analytics.summarize(frame)
        comparisons.append(
            {
                "index": canonical,
                "period_days_applied": applied_period,
                "observations": len(frame),
                "period_return_pct": summary["period_return_pct"],
                "annualized_volatility_pct": summary[
                    "annualized_volatility_pct"
                ],
                "drawdown_from_period_high_pct": summary[
                    "drawdown_from_period_high_pct"
                ],
                "as_of": summary["as_of"],
            }
        )

    candidates = [
        row
        for row in comparisons
        if row.get("error") is None and row["period_return_pct"] is not None
    ]
    best = (
        max(candidates, key=lambda row: row["period_return_pct"])["index"]
        if candidates
        else None
    )
    successful_dates = [
        row["as_of"] for row in comparisons if row.get("error") is None and row["as_of"]
    ]
    payload = {
        "index": [row["index"] for row in comparisons],
        "as_of": max(successful_dates) if successful_dates else None,
        "source": SOURCE,
        "comparisons": comparisons,
        "best_performer": best,
    }
    if (
        os.getenv("EGX_RAISE_TOOL_ERRORS", "").casefold() == "true"
        and comparisons
        and all("error" in row for row in comparisons)
    ):
        first_error = next(
            (row.get("error", {"message": "Unknown error"}) for row in comparisons if row.get("error")),
            {"message": "All requested indices failed."},
        )
        raise ToolError(first_error["message"])
    return payload


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def create_mcp_server(host: str, port: int) -> FastMCP:
    """Create an independently configured MCP server with all EGX tools."""
    server = _new_mcp(host, port)
    for tool in (
        list_indices,
        get_index_history,
        get_index_summary,
        compare_indices,
        list_stocks,
        find_stock,
        get_stock_history,
        get_stock_summary,
        get_stock_fundamentals,
        get_dividends,
        compare_stocks,
        stock_vs_index,
        analysis_snapshot,
    ):
        server.tool()(tool)
    for prompt in (research_company, build_trade_plan):
        server.prompt()(prompt)
    return server


def get_mcp_server(host: str | None = None, port: int | None = None) -> FastMCP:
    """Return the process's lazily constructed default MCP server."""
    global _mcp
    if _mcp is None:
        with _mcp_lock:
            if _mcp is None:
                configured_host = host or os.getenv("MCP_HOST", "127.0.0.1")
                configured_port = port or _port(os.getenv("MCP_PORT", "8000"))
                _mcp = create_mcp_server(configured_host, configured_port)
    return _mcp


def __getattr__(name: str) -> Any:
    if name == "mcp":
        return get_mcp_server()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(argv: Sequence[str] | None = None) -> None:
    """Start the MCP server over stdio or streamable HTTP."""
    egx_client.configure_logging()
    default_transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if default_transport == "streamable-http":
        default_transport = "http"
    parser = argparse.ArgumentParser(
        description=(
            "Serve EGX index data and individual stock data (Yahoo, delayed "
            "daily bars), analysis tools, and research prompts over MCP. "
            "Results are analysis, not financial advice."
        )
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "http", "streamable-http"),
        default=default_transport,
        help="MCP transport (default: stdio, or MCP_TRANSPORT).",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MCP_HOST", "127.0.0.1"),
        help="HTTP bind host (default: MCP_HOST or 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=_port(os.getenv("MCP_PORT", "8000")),
        help="HTTP bind port (default: MCP_PORT or 8000).",
    )
    args = parser.parse_args(argv)
    transport = "http" if args.transport == "streamable-http" else args.transport
    if transport == "stdio":
        server = get_mcp_server(args.host, args.port)
        server.run(transport="stdio")
        return
    from .http_app import create_app

    token = os.getenv("MCP_AUTH_TOKEN")
    http_server = get_mcp_server(args.host, args.port)
    app = create_app({"": http_server}, auth_token=token)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
