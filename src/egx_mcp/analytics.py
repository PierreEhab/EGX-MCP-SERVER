"""Analytics for EGX index history frames."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _json_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


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
    return_5d = (
        _json_float((latest / float(values.iloc[-5]) - 1) * 100)
        if len(values) >= 5 and float(values.iloc[-5]) != 0
        else None
    )

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
