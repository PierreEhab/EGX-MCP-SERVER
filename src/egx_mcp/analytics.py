"""Analytics for EGX index history frames."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _json_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return round(result, 12)


def iso_date(value: Any) -> str | None:
    """Convert a date-like value to an ISO calendar date, or ``None`` if missing."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sorted rows with usable dates and numeric index values."""
    if frame.empty or "datetime" not in frame.columns or "value" not in frame.columns:
        return pd.DataFrame(columns=["datetime", "value"])
    observations = frame.loc[:, ["datetime", "value"]].copy()
    observations["datetime"] = pd.to_datetime(observations["datetime"], errors="coerce")
    observations["value"] = pd.to_numeric(observations["value"], errors="coerce")
    observations = observations.dropna(subset=["datetime", "value"])
    observations = observations.loc[observations["value"].map(math.isfinite)]
    return observations.sort_values("datetime").reset_index(drop=True)


def history_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize source observations as JSON-safe date/value rows."""
    rows = _observations(frame)
    output: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        output.append(
            {
                "date": iso_date(row.datetime),
                "value": _json_float(pd.to_numeric(row.value, errors="coerce")),
            }
        )
    return output


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate period performance, moving averages, and annualized volatility."""
    observations = _observations(frame)
    values = observations["value"]
    returns = (
        values.pct_change(fill_method=None)
        .replace([float("inf"), float("-inf")], pd.NA)
        .dropna()
    )

    latest = _json_float(values.iloc[-1]) if not values.empty else None
    previous = _json_float(values.iloc[-2]) if len(values) >= 2 else None
    daily_change = (
        _json_float(latest - previous)
        if latest is not None and previous is not None
        else None
    )
    daily_change_pct = (
        _json_float(daily_change / previous * 100)
        if daily_change is not None and previous != 0
        else None
    )
    first = _json_float(values.iloc[0]) if not values.empty else None
    period_return = (
        _json_float((latest / first - 1) * 100)
        if latest is not None and first not in (None, 0)
        else None
    )
    volatility = (
        _json_float(returns.std(ddof=1) * (252**0.5) * 100)
        if len(returns) >= 2
        else None
    )
    period_peak = float(values.max()) if not values.empty else None
    drawdown = None
    if latest is not None and period_peak is not None:
        drawdown = (
            _json_float((latest / period_peak - 1) * 100)
            if period_peak != 0
            else (0.0 if latest == 0 else None)
        )
    return_5d = None
    if len(values) >= 6:
        comparison_value = values.iloc[-6]
        if comparison_value not in (None, 0) and math.isfinite(float(comparison_value)):
            return_5d = _json_float((latest / float(comparison_value) - 1) * 100)

    return {
        "as_of": iso_date(observations["datetime"].iloc[-1])
        if not observations.empty
        else None,
        "latest_value": latest,
        "previous_close": previous,
        "daily_change": daily_change,
        "daily_change_pct": daily_change_pct,
        "period_high": _json_float(values.max()) if not values.empty else None,
        "period_low": _json_float(values.min()) if not values.empty else None,
        "period_return_pct": period_return,
        "sma_20": _json_float(values.tail(20).mean()) if len(values) >= 20 else None,
        "sma_50": _json_float(values.tail(50).mean()) if len(values) >= 50 else None,
        "annualized_volatility_pct": volatility,
        "drawdown_from_period_high_pct": drawdown,
        "return_5d_pct": return_5d,
        "date_range": {
            "start": iso_date(observations["datetime"].iloc[0])
            if not observations.empty
            else None,
            "end": iso_date(observations["datetime"].iloc[-1])
            if not observations.empty
            else None,
        },
    }


def _stock_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, valid daily stock bars with numeric optional fields."""
    columns = (
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    )
    if frame.empty or "date" not in frame.columns or "close" not in frame.columns:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame({"date": pd.to_datetime(frame["date"], errors="coerce")})
    for column in columns[1:]:
        if column in frame.columns:
            result[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            result[column] = float("nan")
    result = result.dropna(subset=["date", "close"])
    result = result.loc[result["close"].map(lambda value: math.isfinite(float(value)))]
    for column in columns[1:]:
        result[column] = result[column].replace([float("inf"), float("-inf")], pd.NA)
    return result.sort_values("date").reset_index(drop=True).loc[:, list(columns)]


def _return_pct(values: pd.Series) -> float | None:
    if len(values) < 1:
        return None
    first = float(values.iloc[0])
    last = float(values.iloc[-1])
    if first == 0 or not math.isfinite(first) or not math.isfinite(last):
        return None
    return _json_float((last / first - 1) * 100)


def _stock_return_pair(frame: pd.DataFrame) -> dict[str, float | None]:
    return {
        "price_return_pct": _return_pct(frame["close"]),
        "total_return_pct": (
            _return_pct(frame["adj_close"].dropna())
            if frame["adj_close"].notna().all()
            else None
        ),
    }


def _window_returns(
    observations: pd.DataFrame,
    latest_date: pd.Timestamp,
) -> dict[str, dict[str, float | None]]:
    windows = {"1m": 30, "3m": 90, "6m": 180, "1y": 365}
    output: dict[str, dict[str, float | None]] = {}
    for label, days in windows.items():
        cutoff = latest_date - pd.Timedelta(days=days)
        window = observations.loc[observations["date"] >= cutoff]
        is_covered = (
            not observations.empty
            and observations["date"].iloc[0] <= cutoff + pd.Timedelta(days=5)
        )
        output[label] = {
            **(
                _stock_return_pair(window)
                if is_covered and not window.empty
                else {"price_return_pct": None, "total_return_pct": None}
            ),
            "start_date": iso_date(window["date"].iloc[0]) if not window.empty else None,
        }
    return output


def stock_summary(
    bars: pd.DataFrame,
    dividends: pd.DataFrame,
    period_days: int,
) -> dict[str, Any]:
    """Calculate price, adjusted-total-return, risk, volume, and dividend metrics.

    ``Close`` is split-adjusted and is used for price return and risk. ``Adj
    Close`` is dividend-adjusted and is used for total return. Fixed-window
    returns are reported only when the observations cover the full calendar
    window, allowing up to five calendar days of start-date slack. Both
    ``max_drawdown_pct`` and ``drawdown_from_period_high_pct`` use closing
    prices; reported period high/low and 52-week high/low use intraday highs
    and lows when available.
    """
    if isinstance(period_days, bool) or not isinstance(period_days, int) or period_days < 1:
        raise ValueError("period_days must be a positive integer.")
    observations = _stock_observations(bars)
    close = observations["close"]
    latest_date = observations["date"].iloc[-1] if not observations.empty else None
    if latest_date is not None:
        cutoff = latest_date - pd.Timedelta(days=period_days)
        period = observations.loc[observations["date"] >= cutoff].reset_index(drop=True)
    else:
        period = observations

    period_close = period["close"]
    latest = _json_float(period_close.iloc[-1]) if not period.empty else None
    previous = _json_float(period_close.iloc[-2]) if len(period_close) >= 2 else None
    daily_change = (
        _json_float(latest - previous)
        if latest is not None and previous is not None
        else None
    )
    daily_change_pct = (
        _json_float(daily_change / previous * 100)
        if daily_change is not None and previous not in (None, 0)
        else None
    )
    period_peak = (
        _json_float(period["high"].max())
        if not period.empty and period["high"].notna().any()
        else (_json_float(period_close.max()) if not period.empty else None)
    )
    period_low = (
        _json_float(period["low"].min())
        if not period.empty and period["low"].notna().any()
        else (_json_float(period_close.min()) if not period.empty else None)
    )
    drawdown_from_high = (
        _json_float((latest / period_close.max() - 1) * 100)
        if latest is not None and not period_close.empty and period_close.max() != 0
        else (0.0 if latest == 0 and not period_close.empty else None)
    )
    close_returns = period_close.pct_change(fill_method=None).dropna()
    close_returns = close_returns.loc[
        close_returns.map(lambda value: math.isfinite(float(value)))
    ]
    annualized_volatility = (
        _json_float(close_returns.std(ddof=1) * math.sqrt(252) * 100)
        if len(close_returns) >= 2
        else None
    )
    drawdown_series = period_close / period_close.cummax() - 1
    max_drawdown = (
        _json_float(drawdown_series.min() * 100) if not drawdown_series.empty else None
    )

    week_cutoff = latest_date - pd.Timedelta(days=365) if latest_date is not None else None
    week = (
        observations.loc[observations["date"] >= week_cutoff]
        if week_cutoff is not None
        else observations
    )
    week_high = (
        _json_float(week["high"].max())
        if not week.empty and week["high"].notna().any()
        else (_json_float(week["close"].max()) if not week.empty else None)
    )
    week_low = (
        _json_float(week["low"].min())
        if not week.empty and week["low"].notna().any()
        else (_json_float(week["close"].min()) if not week.empty else None)
    )
    dividend_yield = None
    if latest is not None and latest != 0 and not dividends.empty:
        if "date" in dividends.columns and "amount" in dividends.columns:
            payments = dividends.loc[:, ["date", "amount"]].copy()
            payments["date"] = pd.to_datetime(payments["date"], errors="coerce")
            payments["amount"] = pd.to_numeric(payments["amount"], errors="coerce")
            payments = payments.dropna(subset=["date", "amount"])
            trailing_start = latest_date - pd.Timedelta(days=365)
            payments = payments.loc[
                (payments["date"] >= trailing_start)
                & (payments["date"] <= latest_date)
            ]
            dividend_sum = payments["amount"].sum()
            if math.isfinite(float(dividend_sum)) and dividend_sum != 0:
                dividend_yield = _json_float(float(dividend_sum) / latest * 100)

    price_returns = _stock_return_pair(period)
    return {
        "as_of": iso_date(latest_date),
        "latest_close": latest,
        "previous_close": previous,
        "daily_change": daily_change,
        "daily_change_pct": daily_change_pct,
        "period_high": period_peak,
        "period_low": period_low,
        "sma_20": _json_float(close.tail(20).mean()) if len(close) >= 20 else None,
        "sma_50": _json_float(close.tail(50).mean()) if len(close) >= 50 else None,
        "sma_200": _json_float(close.tail(200).mean()) if len(close) >= 200 else None,
        "annualized_volatility_pct": annualized_volatility,
        "max_drawdown_pct": max_drawdown,
        "drawdown_from_period_high_pct": drawdown_from_high,
        "price_return_pct": price_returns["price_return_pct"],
        "total_return_pct": price_returns["total_return_pct"],
        "returns": _window_returns(observations, latest_date)
        if latest_date is not None
        else {
            label: {
                "price_return_pct": None,
                "total_return_pct": None,
                "start_date": None,
            }
            for label in ("1m", "3m", "6m", "1y")
        },
        "week52_high": week_high,
        "week52_low": week_low,
        "average_volume_20d": (
            _json_float(observations["volume"].tail(20).mean())
            if len(observations) >= 20 and observations["volume"].tail(20).notna().all()
            else None
        ),
        "average_volume_50d": (
            _json_float(observations["volume"].tail(50).mean())
            if len(observations) >= 50 and observations["volume"].tail(50).notna().all()
            else None
        ),
        "dividend_yield_ttm_pct": dividend_yield,
        "date_range": {
            "start": iso_date(observations["date"].iloc[0])
            if not observations.empty
            else None,
            "end": iso_date(observations["date"].iloc[-1])
            if not observations.empty
            else None,
        },
        "observations": len(period),
        "period_days_applied": period_days,
    }


def _price_series(values: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        date_column = "date" if "date" in values else "datetime"
        price_column = "close" if "close" in values else "value"
        if date_column not in values or price_column not in values:
            return pd.Series(dtype=float)
        result = values.set_index(date_column)[price_column]
    elif isinstance(values, pd.Series):
        result = values.copy()
    else:
        raise TypeError("price series must be a pandas Series or DataFrame.")
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = pd.to_numeric(result, errors="coerce")
    result = result.loc[result.index.notna() & result.map(math.isfinite)]
    return result.sort_index()


def relative_performance(
    stock_close: pd.Series | pd.DataFrame,
    index_close: pd.Series | pd.DataFrame,
) -> dict[str, Any]:
    """Compare stock and index returns on common dates.

    Correlation and beta use aligned daily returns. Fewer than 30 common price
    observations produce null metrics and an explanatory message.
    """
    stock = _price_series(stock_close).rename("stock")
    index = _price_series(index_close).rename("index")
    aligned = pd.concat([stock, index], axis=1, join="inner").dropna()
    common_count = len(aligned)
    if common_count < 30:
        return {
            "correlation": None,
            "beta": None,
            "excess_return_pct": None,
            "observations": common_count,
            "message": (
                f"At least 30 common date observations are required; "
                f"only {common_count} are available."
            ),
        }
    stock_returns = aligned["stock"].pct_change(fill_method=None).dropna()
    index_returns = aligned["index"].pct_change(fill_method=None).dropna()
    returns = pd.concat([stock_returns, index_returns], axis=1, join="inner").dropna()
    variance = float(returns["index"].var(ddof=1)) if len(returns) > 1 else 0.0
    beta = (
        _json_float(returns["stock"].cov(returns["index"]) / variance)
        if math.isfinite(variance) and variance > 0
        else None
    )
    stock_std = float(returns["stock"].std(ddof=1)) if len(returns) > 1 else 0.0
    index_std = float(returns["index"].std(ddof=1)) if len(returns) > 1 else 0.0
    correlation = (
        _json_float(returns["stock"].corr(returns["index"]))
        if stock_std > 0 and index_std > 0
        else None
    )
    stock_return = _return_pct(aligned["stock"])
    index_return = _return_pct(aligned["index"])
    excess_return = (
        _json_float(stock_return - index_return)
        if stock_return is not None and index_return is not None
        else None
    )
    return {
        "correlation": correlation,
        "beta": beta,
        "excess_return_pct": excess_return,
        "observations": common_count,
        "message": None,
    }


def key_levels(bars: pd.DataFrame) -> dict[str, float | None]:
    """Return 52-week, moving-average, and trailing 60-session swing levels."""
    observations = _stock_observations(bars)
    if observations.empty:
        return {
            "week52_high": None,
            "week52_low": None,
            "sma_20": None,
            "sma_50": None,
            "sma_200": None,
            "swing_high_60d": None,
            "swing_low_60d": None,
        }
    latest_date = observations["date"].iloc[-1]
    week = observations.loc[
        observations["date"] >= latest_date - pd.Timedelta(days=365)
    ]
    swing = observations.tail(60)
    return {
        "week52_high": (
            _json_float(week["high"].max())
            if week["high"].notna().any()
            else _json_float(week["close"].max())
        ),
        "week52_low": (
            _json_float(week["low"].min())
            if week["low"].notna().any()
            else _json_float(week["close"].min())
        ),
        "sma_20": (
            _json_float(observations["close"].tail(20).mean())
            if len(observations) >= 20
            else None
        ),
        "sma_50": (
            _json_float(observations["close"].tail(50).mean())
            if len(observations) >= 50
            else None
        ),
        "sma_200": (
            _json_float(observations["close"].tail(200).mean())
            if len(observations) >= 200
            else None
        ),
        "swing_high_60d": (
            _json_float(swing["high"].max())
            if swing["high"].notna().any()
            else _json_float(swing["close"].max())
        ),
        "swing_low_60d": (
            _json_float(swing["low"].min())
            if swing["low"].notna().any()
            else _json_float(swing["close"].min())
        ),
    }
