"""Adapter around the external ``egx`` data package."""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime

import egx
import pandas as pd
import requests

from .cache import TTLCache

SOURCE = "EGX via egx package"

_log_level_name = os.getenv("EGX_LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_name, None)
if not isinstance(_log_level, int):
    raise ValueError("EGX_LOG_LEVEL must be a valid logging level")
logging.basicConfig(
    level=_log_level,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
_logger = logging.getLogger(__name__)


class EGXClientError(Exception):
    """Base class for errors returned by the EGX data adapter."""


class InvalidIndexError(EGXClientError):
    """The requested index is not supported by the installed ``egx`` package."""


class EmptyDataError(EGXClientError):
    """The source returned no index observations for the requested period."""


class EGXNetworkError(EGXClientError):
    """The EGX request failed at the network or HTTP layer."""


class InvalidPeriodError(EGXClientError):
    """The requested period is not an integer."""


def _cache_ttl() -> float:
    configured = os.getenv("EGX_CACHE_TTL_SECONDS", "300")
    try:
        ttl = float(configured)
    except ValueError as exc:
        raise ValueError("EGX_CACHE_TTL_SECONDS must be a non-negative number") from exc
    if not math.isfinite(ttl) or ttl < 0:
        raise ValueError("EGX_CACHE_TTL_SECONDS must be a finite non-negative number")
    return ttl


_DATA_LOCKS: dict[str, threading.Lock] = {}
_DATA_LOCKS_LOCK = threading.Lock()
_INFLIGHT: dict[tuple[str, int], Future["_CachedFrame"]] = {}
_INFLIGHT_LOCK = threading.Lock()


@dataclass(frozen=True)
class _CachedFrame:
    period_days: int
    requested_on: pd.Timestamp
    frame: pd.DataFrame


_DATA_CACHE: TTLCache[_CachedFrame] = TTLCache(ttl_seconds=_cache_ttl())


def supported_indices() -> list[str]:
    """Return canonical index codes exposed by the installed ``egx`` package."""
    return sorted(egx.get_supported_indices())


def resolve_index(index: str) -> str:
    """Resolve an index code case-insensitively to its canonical package spelling."""
    if not isinstance(index, str):
        raise InvalidIndexError("Index must be a string.")
    canonical_by_casefold = {code.casefold(): code for code in supported_indices()}
    canonical = canonical_by_casefold.get(index.strip().casefold())
    if canonical is None:
        raise InvalidIndexError(
            f"Unsupported index '{index}'. Use list_indices to see available codes."
        )
    return canonical


def clamp_period_days(period_days: int) -> int:
    """Clamp a period to the supported 1-1825 calendar-day range."""
    if isinstance(period_days, bool) or not isinstance(period_days, int):
        raise InvalidPeriodError("period_days must be an integer.")
    return max(1, min(1825, period_days))


def fetch_index_data(index: str, period_days: int) -> pd.DataFrame:
    """Fetch daily ``datetime``/``value`` data for an already-canonical index."""
    canonical = index
    period = clamp_period_days(period_days)
    cache_key = canonical.casefold()
    request_key = (cache_key, period)
    with _DATA_LOCKS_LOCK:
        lock = _DATA_LOCKS.setdefault(cache_key, threading.Lock())
    with _INFLIGHT_LOCK:
        future = _INFLIGHT.get(request_key)
        leader = future is None
        if future is None:
            future = Future()
            _INFLIGHT[request_key] = future

    if not leader:
        return _slice_cached(future.result(), period, canonical)

    try:
        with lock:
            cached = _DATA_CACHE.get(cache_key)
            if cached is None or cached.period_days < period:
                try:
                    frame = egx.get_index_data(canonical, period=period)
                except requests.exceptions.RequestException:
                    raise EGXNetworkError(
                        f"Unable to fetch {canonical} data from EGX. Please try again later."
                    ) from None
                except Exception:
                    _logger.exception("Unexpected failure fetching %s from EGX", canonical)
                    raise EGXClientError(
                        "The EGX data source failed unexpectedly. Please try again later."
                    ) from None

                if not isinstance(frame, pd.DataFrame):
                    raise EGXClientError(
                        "The EGX package returned data in an unsupported format."
                    )
                if frame.empty:
                    raise EmptyDataError(
                        f"No {canonical} data is available for the requested period."
                    )
                if "datetime" not in frame.columns or "value" not in frame.columns:
                    raise EGXClientError(
                        "The EGX package returned data without the expected datetime and value columns."
                    )

                normalized = frame.copy(deep=True)
                normalized["datetime"] = pd.to_datetime(
                    normalized["datetime"], errors="coerce"
                )
                normalized["value"] = pd.to_numeric(
                    normalized["value"], errors="coerce"
                )
                normalized = normalized.dropna(subset=["datetime", "value"])
                normalized = normalized.loc[normalized["value"].map(math.isfinite)]
                normalized = normalized.sort_values("datetime").reset_index(drop=True)
                if normalized.empty:
                    raise EmptyDataError(
                        f"No valid {canonical} observations are available for the requested period."
                    )
                cached = _CachedFrame(
                    period_days=period,
                    requested_on=pd.Timestamp(datetime.now().date()),
                    frame=normalized,
                )
                _DATA_CACHE.set(cache_key, cached)

        result = _slice_cached(cached, period, canonical)
        future.set_result(cached)
        return result
    except BaseException as exc:
        if not future.done():
            future.set_exception(exc)
        raise
    finally:
        with _INFLIGHT_LOCK:
            if _INFLIGHT.get(request_key) is future:
                del _INFLIGHT[request_key]


def _slice_cached(
    cached: _CachedFrame,
    period: int,
    canonical: str,
) -> pd.DataFrame:
    cutoff = cached.requested_on - pd.Timedelta(days=period)
    sliced = cached.frame.loc[cached.frame["datetime"] >= cutoff]
    if sliced.empty:
        raise EmptyDataError(
            f"No {canonical} data is available for the requested period."
        )
    result = sliced.copy(deep=True)
    result.index = pd.RangeIndex(len(result))
    return result
