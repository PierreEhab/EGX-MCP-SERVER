"""Egypt macro indicators from FRED, local policy-rate config, and free FX API."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
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
EXCHANGE_RATE_URL = "https://open.er-api.com/v6/latest/USD"
_TIMEOUT_SECONDS = 15
_CACHE_TTL = timedelta(hours=12)
_CACHE_TABLE = "egypt_macro_cache_v2"
_CAIRO = ZoneInfo("Africa/Cairo")
_INFLATION_SERIES = {
    "series_id": "FPCPITOTLZGEGY",
    "name": "Egypt annual CPI inflation",
    "unit": "percent",
    "frequency": "calendar-year average",
}
_EXCHANGE_ATTRIBUTION = (
    "Rates By Exchange Rate API (https://www.exchangerate-api.com)"
)


def _now() -> datetime:
    return datetime.now(_CAIRO)


def _error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _metric_error(
    *,
    name: str,
    source: str,
    unit: str,
    frequency: str,
    error: dict[str, str],
    series_id: str | None = None,
    note: str | None = None,
    attribution: str | None = None,
) -> dict[str, Any]:
    metric = {
        "value": None,
        "date": None,
        "error": error,
        "age_days": None,
        "stale": True,
        "source": source,
        "name": name,
        "unit": unit,
        "frequency": frequency,
    }
    if series_id is not None:
        metric["series_id"] = series_id
    if note is not None:
        metric["note"] = note
    if attribution is not None:
        metric["attribution"] = attribution
    return metric


def _observation_value(observations: Any) -> tuple[str | None, float | None]:
    if not isinstance(observations, list):
        return None, None
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        observed_on = observation.get("date")
        if not isinstance(observed_on, str):
            continue
        try:
            value = float(observation.get("value"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return observed_on, value
    return None, None


def _status_code(exc: Exception, response: Any = None) -> int | None:
    candidate = response if response is not None else getattr(exc, "response", None)
    status = getattr(candidate, "status_code", None)
    return status if isinstance(status, int) else None


def _safe_log_failure(source: str, exc: Exception, response: Any = None) -> None:
    """Log only source, exception class, and HTTP status; never provider text."""
    _logger.warning(
        "%s request failure exception_class=%s http_status=%s",
        source,
        type(exc).__name__,
        _status_code(exc, response),
    )


def _request_error(exc: Exception, response: Any = None) -> dict[str, str]:
    status = _status_code(exc, response)
    if status is not None:
        message = f"Data source request failed with HTTP status {status}."
    else:
        message = f"Data source request failed ({type(exc).__name__})."
    return _error("DATA_SOURCE_ERROR", message)


def _fetch_inflation(api_key: str) -> dict[str, Any]:
    response = None
    try:
        response = requests.get(
            FRED_OBSERVATIONS_URL,
            params={
                "series_id": _INFLATION_SERIES["series_id"],
                "api_key": api_key,
                "file_type": "json",
                "limit": 10,
                "sort_order": "desc",
            },
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        observed_on, value = _observation_value(
            payload.get("observations") if isinstance(payload, dict) else None
        )
        if observed_on is None or value is None:
            return {
                "data": None,
                "error": _error(
                    "EMPTY_DATA",
                    "FRED returned no finite inflation observation in the latest 10 rows.",
                ),
            }
        return {
            "data": {
                "value": value,
                "date": observed_on,
                "source": "FRED",
                "name": _INFLATION_SERIES["name"],
                "unit": _INFLATION_SERIES["unit"],
                "frequency": _INFLATION_SERIES["frequency"],
                "series_id": _INFLATION_SERIES["series_id"],
                "note": (
                    "Calendar-year average; this annual measure lags current "
                    "monthly inflation."
                ),
            },
            "error": None,
        }
    except Exception as exc:
        _safe_log_failure("FRED", exc, response)
        return {"data": None, "error": _request_error(exc, response)}


def _fetch_exchange_rate() -> dict[str, Any]:
    response = None
    try:
        response = requests.get(EXCHANGE_RATE_URL, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("result") != "success":
            return {
                "data": None,
                "error": _error(
                    "DATA_SOURCE_ERROR",
                    "Exchange Rate API did not report a successful result.",
                ),
            }
        rates = payload.get("rates")
        raw_value = rates.get("EGP") if isinstance(rates, dict) else None
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value) or value <= 0:
            return {
                "data": None,
                "error": _error(
                    "EMPTY_DATA",
                    "Exchange Rate API returned no finite positive EGP rate.",
                ),
            }
        update_time = payload.get("time_last_update_utc")
        try:
            observed_on = parsedate_to_datetime(update_time).date().isoformat()
        except (TypeError, ValueError, OverflowError):
            return {
                "data": None,
                "error": _error(
                    "EMPTY_DATA",
                    "Exchange Rate API returned an invalid update date.",
                ),
            }
        return {
            "data": {
                "value": value,
                "date": observed_on,
                "source": "ExchangeRate-API",
                "name": "Egyptian pounds per US dollar",
                "unit": "EGP per USD",
                "frequency": "daily",
                "attribution": _EXCHANGE_ATTRIBUTION,
            },
            "error": None,
        }
    except Exception as exc:
        _safe_log_failure("Exchange Rate API", exc, response)
        return {"data": None, "error": _request_error(exc, response)}


def _policy_rate(today: date) -> dict[str, Any]:
    name = os.getenv(
        "EGYPT_POLICY_RATE_NAME",
        "Central Bank of Egypt policy rate",
    ).strip()
    raw_value = os.getenv("EGYPT_POLICY_RATE_PCT")
    raw_date = os.getenv("EGYPT_POLICY_RATE_DATE")
    common = {
        "name": name or "Central Bank of Egypt policy rate",
        "source": "user_configured",
        "unit": "percent",
        "frequency": "user-configured",
    }
    if not raw_value or not raw_date:
        missing = [
            key
            for key, value in (
                ("EGYPT_POLICY_RATE_PCT", raw_value),
                ("EGYPT_POLICY_RATE_DATE", raw_date),
            )
            if not value
        ]
        return _metric_error(
            **common,
            error=_error(
                "NOT_CONFIGURED",
                "Set " + " and ".join(missing) + " to provide the Egypt policy rate.",
            ),
        )
    try:
        value = float(raw_value)
        observed_on = date.fromisoformat(raw_date)
    except (TypeError, ValueError):
        return _metric_error(
            **common,
            error=_error(
                "INVALID_CONFIGURATION",
                "EGYPT_POLICY_RATE_PCT must be numeric and EGYPT_POLICY_RATE_DATE "
                "must be an ISO date.",
            ),
        )
    if not math.isfinite(value) or not 0 <= value <= 100:
        return _metric_error(
            **common,
            error=_error(
                "INVALID_CONFIGURATION",
                "EGYPT_POLICY_RATE_PCT must be a finite number from 0 to 100.",
            ),
        )
    if observed_on > today:
        return _metric_error(
            **common,
            error=_error(
                "INVALID_CONFIGURATION",
                "EGYPT_POLICY_RATE_DATE cannot be in the future.",
            ),
        )
    age_days = (today - observed_on).days
    return {
        "value": value,
        "date": observed_on.isoformat(),
        "error": None,
        "age_days": age_days,
        "stale": age_days > 60,
        **common,
    }


def _decorate_cached_data(
    metric: str,
    data: dict[str, Any],
    today: date,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    observed_on = date.fromisoformat(data["date"])
    age_days = max(0, (today - observed_on).days)
    if metric == "inflation":
        stale = today.year - observed_on.year > 1
    else:
        stale = age_days > 3
    result = {
        **data,
        "error": error,
        "age_days": age_days,
        "stale": stale,
    }
    return result


def _valid_cache_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    value = payload.get("value")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        return False
    observed_on = payload.get("date")
    if not isinstance(observed_on, str):
        return False
    try:
        if date.fromisoformat(observed_on).isoformat() != observed_on:
            return False
    except ValueError:
        return False
    return all(
        isinstance(payload.get(key), str) and bool(payload[key].strip())
        for key in ("source", "name", "unit", "frequency")
    )


def _ensure_cache_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_CACHE_TABLE} (
            metric TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            fetched_at TEXT NOT NULL
        )
        """
    )


def _read_cache() -> dict[str, tuple[dict[str, Any], datetime]]:
    path = cache_database_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=5)) as connection:
            with connection:
                _ensure_cache_table(connection)
                rows = connection.execute(
                    f"SELECT metric, payload, fetched_at FROM {_CACHE_TABLE}"
                ).fetchall()
        cache = {}
        for metric, payload, fetched_at in rows:
            try:
                data = json.loads(payload)
                if not _valid_cache_payload(data):
                    continue
                fetched = datetime.fromisoformat(fetched_at)
            except (ValueError, TypeError):
                continue
            cache[metric] = (data, fetched)
        return cache
    except (OSError, sqlite3.DatabaseError, ValueError, TypeError) as exc:
        _logger.warning("Macro cache read unavailable exception_class=%s", type(exc).__name__)
        return {}


def _write_cache(metric_data: dict[str, dict[str, Any]], fetched_at: datetime) -> None:
    if not metric_data:
        return
    try:
        path = cache_database_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=5)) as connection:
            with connection:
                _ensure_cache_table(connection)
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
        _logger.warning("Macro cache write unavailable exception_class=%s", type(exc).__name__)


def _cache_is_fresh(entry: tuple[dict[str, Any], datetime] | None, now: datetime) -> bool:
    if entry is None:
        return False
    fetched_at = entry[1]
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=_CAIRO)
    return now.astimezone(_CAIRO) - fetched_at.astimezone(_CAIRO) <= _CACHE_TTL


def get_latest_egypt_macro_data() -> dict[str, Any]:
    """Fetch Egypt inflation and FX data, with a locally configured policy rate.

    Inflation uses FRED's FPCPITOTLZGEGY annual calendar-year average and
    requires ``FRED_API_KEY``. The policy rate is provided by
    ``EGYPT_POLICY_RATE_PCT`` and ``EGYPT_POLICY_RATE_DATE``; its optional
    display name is ``EGYPT_POLICY_RATE_NAME``. EGP per USD comes from the free
    Exchange Rate API. Network observations are cached in SQLite for 12 hours.
    Each metric carries its own source, error, age, and stale flag; a top-level
    error is returned only if every metric has no usable value.
    """
    now = _now()
    today = now.astimezone(_CAIRO).date()
    cache = _read_cache()
    key = os.getenv("FRED_API_KEY")
    inflation_fresh = bool(key) and _cache_is_fresh(cache.get("inflation"), now)
    exchange_fresh = _cache_is_fresh(cache.get("egp_usd_exchange_rate"), now)

    fetched: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, str]] = {}
    jobs = {}
    if key and not inflation_fresh:
        jobs["inflation"] = ("FRED", _fetch_inflation, (key,))
    elif not key:
        failures["inflation"] = _error(
            "MISSING_API_KEY",
            "Set FRED_API_KEY to fetch Egypt annual CPI inflation from FRED.",
        )
    if not exchange_fresh:
        jobs["egp_usd_exchange_rate"] = (
            "Exchange Rate API",
            _fetch_exchange_rate,
            (),
        )

    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as executor:
        futures = {
            executor.submit(function, *args): metric
            for metric, (_, function, args) in jobs.items()
        }
        for future in as_completed(futures):
            metric = futures[future]
            result = future.result()
            if result["error"] is not None:
                failures[metric] = result["error"]
            else:
                fetched[metric] = result["data"]

    _write_cache(fetched, now)
    metrics: dict[str, dict[str, Any]] = {}
    for metric in ("inflation", "egp_usd_exchange_rate"):
        if metric in fetched:
            data = fetched[metric]
            error = None
        elif metric in cache and (
            metric not in failures or failures[metric]["code"] != "MISSING_API_KEY"
        ):
            data = cache[metric][0]
            error = failures.get(metric)
            if error is not None:
                error = {
                    **error,
                    "message": error["message"] + " Serving the last cached value.",
                }
        else:
            data = None
            error = failures.get(
                metric,
                _error("EMPTY_DATA", "No observation is available."),
            )
        if data is None:
            series = _INFLATION_SERIES if metric == "inflation" else {
                "name": "Egyptian pounds per US dollar",
                "source": "ExchangeRate-API",
                "unit": "EGP per USD",
                "frequency": "daily",
            }
            extra = (
                {
                    "series_id": _INFLATION_SERIES["series_id"],
                    "note": (
                        "Calendar-year average; this annual measure lags current "
                        "monthly inflation."
                    ),
                }
                if metric == "inflation"
                else {"attribution": _EXCHANGE_ATTRIBUTION}
            )
            metrics[metric] = _metric_error(
                name=series["name"],
                source=series.get("source", "FRED"),
                unit=series["unit"],
                frequency=series["frequency"],
                error=error,
                **extra,
            )
        else:
            metrics[metric] = _decorate_cached_data(metric, data, today, error)

    metrics["policy_rate"] = _policy_rate(today)
    sources = list(
        dict.fromkeys(
            source
            for metric in metrics.values()
            if isinstance(source := metric.get("source"), str)
        )
    )
    data_quality = [
        name for name, metric in metrics.items() if metric["stale"]
    ]
    result: dict[str, Any] = {
        "source": sources,
        "metrics": metrics,
        "data_quality": data_quality,
    }
    if fetched or any(
        _cache_is_fresh(cache.get(metric), now)
        for metric in ("inflation", "egp_usd_exchange_rate")
    ):
        result["cache_note"] = "FRED and exchange-rate observations use a 12-hour SQLite cache."
    if all(metric["value"] is None for metric in metrics.values()):
        result["error"] = {
            "code": "DATA_SOURCE_ERROR",
            "message": "No usable values are available for any Egypt macro metric.",
        }
    return result
