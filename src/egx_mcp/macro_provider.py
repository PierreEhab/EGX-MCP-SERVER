"""Free Federal Reserve Economic Data access for Egypt macro indicators."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import date, datetime, timedelta
import json
import logging
import math
import os
import sqlite3
from typing import Any
from zoneinfo import ZoneInfo

import requests

from .stock_cache import cache_database_path

_logger = logging.getLogger(__name__)
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
_TIMEOUT_SECONDS = 15
_CACHE_TTL = timedelta(hours=12)
_CACHE_TABLE = "egypt_macro_cache"
_CAIRO = ZoneInfo("Africa/Cairo")
_SERIES = {
    "inflation": {
        "series_id": "FPCPITOTLZGEGY",
        "name": "Egypt consumer price inflation (annual percent)",
        "unit": "percent",
        "frequency": "annual",
        "stale_after_days": 550,
    },
    "discount_rate": {
        "series_id": "INTDSRTRM193N",
        "name": "Egypt discount rate",
        "unit": "percent",
        "frequency": "monthly",
        "stale_after_days": 75,
    },
    "egp_usd_exchange_rate": {
        "series_id": "DEXEGUS",
        "name": "Egyptian pounds per US dollar",
        "unit": "EGP per USD",
        "frequency": "daily",
        "stale_after_days": 10,
    },
}


class MacroDataError(Exception):
    """Base error for Egypt macroeconomic data retrieval."""

    code = "DATA_SOURCE_ERROR"


class MissingFredApiKeyError(MacroDataError):
    """FRED access requires an API key in the process environment."""

    code = "MISSING_API_KEY"


def _now() -> datetime:
    return datetime.now(_CAIRO)


def _observation_value(observations: Any) -> tuple[str | None, float | None]:
    if not isinstance(observations, list) or not observations:
        return None, None
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        date_value = observation.get("date")
        if not isinstance(date_value, str):
            continue
        try:
            value = float(observation.get("value"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return date_value, value
    return None, None


def _status_code(exc: Exception, response: Any = None) -> int | None:
    candidate = response if response is not None else getattr(exc, "response", None)
    status = getattr(candidate, "status_code", None)
    return status if isinstance(status, int) else None


def _safe_log_request_failure(exc: Exception, response: Any = None) -> None:
    """Log only exception class and HTTP status, never provider text or URL."""
    _logger.warning(
        "FRED request failure exception_class=%s http_status=%s",
        type(exc).__name__,
        _status_code(exc, response),
    )


def _failure_message(exc: Exception, response: Any = None) -> str:
    status = _status_code(exc, response)
    if status is not None:
        return f"FRED request failed with HTTP status {status}."
    return f"FRED request failed ({type(exc).__name__})."


def _fetch_metric(metric: str, series: dict[str, Any], api_key: str) -> dict[str, Any]:
    params = {
        "series_id": series["series_id"],
        "api_key": api_key,
        "file_type": "json",
        "limit": 10,
        "sort_order": "desc",
    }
    response = None
    try:
        response = requests.get(
            FRED_OBSERVATIONS_URL,
            params=params,
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("FRED response was not a JSON object.")
        observed_on, value = _observation_value(payload.get("observations"))
        if observed_on is None or value is None:
            return {
                "metric": metric,
                "data": _metric_data(series, observed_on, value),
                "error": "FRED returned no valid observation.",
                "fetch_failed": True,
            }
        return {
            "metric": metric,
            "data": _metric_data(series, observed_on, value),
            "error": None,
            "fetch_failed": False,
        }
    except Exception as exc:
        _safe_log_request_failure(exc, response)
        return {
            "metric": metric,
            "data": None,
            "error": _failure_message(exc, response),
            "fetch_failed": True,
        }


def _metric_data(
    series: dict[str, Any],
    observed_on: str | None,
    value: float | None,
) -> dict[str, Any]:
    return {
        "series_id": series["series_id"],
        "name": series["name"],
        "unit": series["unit"],
        "frequency": series["frequency"],
        "date": observed_on,
        "value": value,
    }


def _decorate_metric(
    data: dict[str, Any],
    error: str | None,
    today: date,
) -> dict[str, Any]:
    try:
        observation_date = date.fromisoformat(data["date"]) if data.get("date") else None
    except ValueError:
        observation_date = None
    age_days = max(0, (today - observation_date).days) if observation_date else None
    stale = (
        age_days is None
        or age_days > _SERIES[data["metric_key"]]["stale_after_days"]
    )
    return {
        "value": data.get("value"),
        "date": data.get("date"),
        "error": error,
        "series_id": data["series_id"],
        "name": data["name"],
        "unit": data["unit"],
        "frequency": data["frequency"],
        "age_days": age_days,
        "stale": stale,
    }


def _read_cache() -> dict[str, tuple[dict[str, Any], datetime]]:
    path = cache_database_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=5)) as connection:
            with connection:
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_CACHE_TABLE} (
                        metric TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        fetched_at TEXT NOT NULL
                    )
                    """
                )
                rows = connection.execute(
                    f"SELECT metric, payload, fetched_at FROM {_CACHE_TABLE}"
                ).fetchall()
        return {
            metric: (json.loads(payload), datetime.fromisoformat(fetched_at))
            for metric, payload, fetched_at in rows
        }
    except (OSError, sqlite3.DatabaseError, ValueError, TypeError) as exc:
        _logger.warning("FRED cache read unavailable exception_class=%s", type(exc).__name__)
        return {}


def _write_cache(metric_data: dict[str, dict[str, Any]], fetched_at: datetime) -> None:
    if not metric_data:
        return
    try:
        path = cache_database_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=5)) as connection:
            with connection:
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_CACHE_TABLE} (
                        metric TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        fetched_at TEXT NOT NULL
                    )
                    """
                )
                connection.executemany(
                    f"""
                    INSERT INTO {_CACHE_TABLE} (metric, payload, fetched_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(metric) DO UPDATE SET
                        payload=excluded.payload,
                        fetched_at=excluded.fetched_at
                    """,
                    [
                        (
                            metric,
                            json.dumps(data, allow_nan=False),
                            fetched_at.isoformat(),
                        )
                        for metric, data in metric_data.items()
                    ],
                )
    except (OSError, sqlite3.DatabaseError, ValueError, TypeError) as exc:
        _logger.warning("FRED cache write unavailable exception_class=%s", type(exc).__name__)


def get_latest_egypt_macro_data(api_key: str | None = None) -> dict[str, Any]:
    """Fetch or return cached latest Egypt inflation, discount-rate, and FX data.

    Requests FPCPITOTLZGEGY (annual inflation), INTDSRTRM193N (monthly
    discount rate), and DEXEGUS (daily EGP per USD) concurrently. Requires
    ``FRED_API_KEY``. Each metric includes its observation date, age, stale
    status, and any retrieval error. A failed refresh falls back to the last
    cached value with an explicit metric error; one series failure does not
    discard other results. If all series fail, the result includes a
    top-level error. Data is cached in the existing SQLite cache directory
    for 12 hours.
    """
    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        raise MissingFredApiKeyError(
            "FRED_API_KEY is required to fetch Egypt macroeconomic data."
        )

    now = _now()
    today = now.astimezone(_CAIRO).date()
    cache = _read_cache()
    if all(
        metric in cache
        and now.astimezone(cache[metric][1].tzinfo or _CAIRO)
        - cache[metric][1].astimezone(_CAIRO)
        <= _CACHE_TTL
        for metric in _SERIES
    ):
        metrics: dict[str, Any] = {}
        for metric in _SERIES:
            data, _ = cache[metric]
            metrics[metric] = _decorate_metric(
                {**data, "metric_key": metric},
                None,
                today,
            )
        return {
            "source": "FRED",
            "metrics": metrics,
            "data_quality": [
                metric
                for metric, result in metrics.items()
                if result["stale"]
            ],
            "cache_note": "Returned from the 12-hour SQLite cache.",
        }

    fetched: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_fetch_metric, metric, series, key): metric
            for metric, series in _SERIES.items()
        }
        for future in as_completed(futures):
            result = future.result()
            metric = result["metric"]
            if result["fetch_failed"]:
                failures[metric] = result["error"]
            elif result["data"] is not None:
                fetched[metric] = result["data"]

    _write_cache(fetched, now)
    metrics = {}
    used_cached_fallback = False
    for metric, series in _SERIES.items():
        if metric in fetched:
            data = fetched[metric]
            error = None
        elif metric in cache:
            data = cache[metric][0]
            used_cached_fallback = True
            error = (
                f"{failures[metric]} Serving the last cached value."
                if metric in failures
                else "FRED refresh failed. Serving the last cached value."
            )
        else:
            data = _metric_data(series, None, None)
            error = failures.get(metric, "FRED did not return a valid observation.")
        metrics[metric] = _decorate_metric(
            {**data, "metric_key": metric},
            error,
            today,
        )

    data_quality = [
        metric for metric, result in metrics.items() if result["stale"]
    ]
    result: dict[str, Any] = {
        "source": "FRED",
        "metrics": metrics,
        "data_quality": data_quality,
    }
    if used_cached_fallback:
        result["cache_note"] = (
            "One or more failed FRED refreshes use the last cached observation."
        )
    if len(failures) == len(_SERIES):
        result["error"] = {
            "code": "DATA_SOURCE_ERROR",
            "message": "All three FRED series requests failed.",
        }
    return result
