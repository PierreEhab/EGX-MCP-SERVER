"""Durable SQLite cache for Yahoo stock bars and dividend history."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .egx_client import _cairo_timezone
from .stock_provider import StockDataProvider

_logger = logging.getLogger(__name__)
_BAR_FIELDS = ("open", "high", "low", "close", "adj_close", "volume")
_BAR_COLUMNS = ("date", *_BAR_FIELDS)
_OVERLAP_TOLERANCE = 1e-6
_MARKET_CLOSE = time(14, 15)
_COMPLETION_DELAY = timedelta(minutes=30)
_DIVIDEND_TTL = timedelta(days=7)


def _is_trading_day(value: date) -> bool:
    return value.weekday() in {6, 0, 1, 2, 3}


def cache_database_path() -> Path:
    """Return the SQLite cache path configured for this operating system."""
    configured = os.getenv("EGX_CACHE_DIR")
    if configured:
        directory = Path(configured).expanduser()
    elif os.name == "nt" and os.getenv("LOCALAPPDATA"):
        directory = Path(os.environ["LOCALAPPDATA"]) / "egx-mcp"
    else:
        directory = Path.home() / ".cache" / "egx-mcp"
    return directory / "stock_cache.sqlite3"


def most_recent_completed_session(now: datetime) -> date:
    """Return the latest completed Sunday-Thursday EGX session."""
    local_now = now.astimezone(_cairo_timezone())
    current = local_now.date()
    if _is_trading_day(current) and local_now >= completed_close(current):
        return current
    candidate = current - timedelta(days=1)
    while not _is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def completed_close(session_date: date) -> datetime:
    """Return an EGX session's 14:15 close plus 30 minutes in Cairo time."""
    close = datetime.combine(session_date, _MARKET_CLOSE, tzinfo=_cairo_timezone())
    return close + _COMPLETION_DELAY


def _completed_bars(frame: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Never store or return today's bar before its completed close."""
    local_now = now.astimezone(_cairo_timezone())
    if (
        _is_trading_day(local_now.date())
        and local_now < completed_close(local_now.date())
    ):
        dates = pd.to_datetime(frame["date"]).dt.date
        return frame.loc[dates < local_now.date()].reset_index(drop=True)
    return frame.copy().reset_index(drop=True)


def _as_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _numeric_equal(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    left_float = float(left)
    right_float = float(right)
    return abs(left_float - right_float) <= _OVERLAP_TOLERANCE * max(
        1.0, abs(left_float), abs(right_float)
    )


class CachedStockProvider:
    """Persist provider bars and dividends in SQLite with session-aware refresh.

    SQLite transactions provide atomic upserts. If the database is corrupt it
    is rebuilt; if it cannot be opened, this wrapper logs the issue and
    transparently uses the wrapped provider without caching.
    """

    def __init__(
        self,
        provider: StockDataProvider,
        database_path: Path | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.provider = provider
        self.database_path = database_path or cache_database_path()
        self._clock = clock or (lambda: datetime.now(_cairo_timezone()))
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self.last_history_decision: dict[str, str | None] = {}
        self.last_dividend_decision: dict[str, str | None] = {}
        self._open_cache()

    def _open_cache(self) -> None:
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                self.database_path,
                timeout=5,
                check_same_thread=False,
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            integrity = self._connection.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise sqlite3.DatabaseError(
                    f"integrity check failed: {integrity[0] if integrity else 'no result'}"
                )
            self._create_schema()
        except sqlite3.DatabaseError as exc:
            _logger.warning("Stock cache database is corrupt; rebuilding: %s", exc)
            self._close_connection()
            self._remove_database_files()
            try:
                self._connection = sqlite3.connect(
                    self.database_path,
                    timeout=5,
                    check_same_thread=False,
                )
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=FULL")
                self._create_schema()
            except (OSError, sqlite3.DatabaseError) as rebuild_exc:
                _logger.warning(
                    "Unable to rebuild stock cache; continuing without cache: %s",
                    rebuild_exc,
                )
                self._close_connection()
        except (OSError, sqlite3.DatabaseError) as exc:
            _logger.warning("Unable to open stock cache; continuing without cache: %s", exc)
            self._close_connection()

    def _create_schema(self) -> None:
        if self._connection is None:
            return
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS bars (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume REAL,
                PRIMARY KEY (symbol, date)
            );
            CREATE TABLE IF NOT EXISTS dividends (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                amount REAL NOT NULL,
                PRIMARY KEY (symbol, date)
            );
            CREATE TABLE IF NOT EXISTS meta (
                symbol TEXT PRIMARY KEY,
                fetched_at TEXT NOT NULL,
                last_bar_date TEXT,
                dividends_fetched_at TEXT,
                covered_from TEXT
            );
            """
        )
        meta_columns = {
            row[1]
            for row in self._connection.execute("PRAGMA table_info(meta)").fetchall()
        }
        if "dividends_fetched_at" not in meta_columns:
            self._connection.execute(
                "ALTER TABLE meta ADD COLUMN dividends_fetched_at TEXT"
            )
        if "covered_from" not in meta_columns:
            self._connection.execute("ALTER TABLE meta ADD COLUMN covered_from TEXT")
        self._connection.commit()

    def close(self) -> None:
        """Close the SQLite connection; primarily useful for orderly shutdown."""
        with self._lock:
            self._close_connection()

    def _close_connection(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except sqlite3.DatabaseError:
                pass
            self._connection = None

    def _remove_database_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(f"{self.database_path}{suffix}").unlink(missing_ok=True)
            except OSError as exc:
                _logger.warning("Unable to remove corrupt stock cache file: %s", exc)

    def _disable_cache(self, exc: Exception) -> None:
        _logger.warning("Stock cache is unavailable; continuing without it: %s", exc)
        self._close_connection()

    def _safe_call_cache(self, function: Callable[..., Any], *args: Any) -> Any:
        if self._connection is None:
            return None
        try:
            with self._lock:
                return function(*args)
        except (OSError, sqlite3.DatabaseError) as exc:
            self._disable_cache(exc)
            return None

    def _read_bars(self, symbol: str) -> pd.DataFrame | None:
        def read() -> pd.DataFrame:
            assert self._connection is not None
            rows = self._connection.execute(
                "SELECT date, open, high, low, close, adj_close, volume "
                "FROM bars WHERE symbol = ? ORDER BY date",
                (symbol,),
            ).fetchall()
            return pd.DataFrame(rows, columns=_BAR_COLUMNS)

        return self._safe_call_cache(read)

    def _read_meta(
        self,
        symbol: str,
    ) -> tuple[datetime, str | None, datetime | None, date | None] | None:
        def read() -> tuple[datetime, str | None, datetime | None, date | None] | None:
            assert self._connection is not None
            row = self._connection.execute(
                "SELECT fetched_at, last_bar_date, dividends_fetched_at, covered_from "
                "FROM meta WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            return (
                (
                    datetime.fromisoformat(row[0]),
                    row[1],
                    datetime.fromisoformat(row[2]) if row[2] else None,
                    date.fromisoformat(row[3]) if row[3] else None,
                )
                if row
                else None
            )

        return self._safe_call_cache(read)

    def _write_bars(
        self,
        symbol: str,
        frame: pd.DataFrame,
        replace_all: bool = False,
        covered_from: date | None = None,
    ) -> None:
        def write() -> None:
            assert self._connection is not None
            rows = [
                (
                    symbol,
                    _date_text(row.date),
                    *(_number_or_none(getattr(row, field)) for field in _BAR_FIELDS),
                )
                for row in frame.itertuples(index=False)
            ]
            with self._connection:
                if replace_all:
                    self._connection.execute(
                        "DELETE FROM bars WHERE symbol = ?", (symbol,)
                    )
                self._connection.executemany(
                    """
                    INSERT INTO bars
                        (symbol, date, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, date) DO UPDATE SET
                        open=excluded.open,
                        high=excluded.high,
                        low=excluded.low,
                        close=excluded.close,
                        adj_close=excluded.adj_close,
                        volume=excluded.volume
                    """,
                    rows,
                )
                last_date = (
                    max((_date_text(value) for value in frame["date"]), default=None)
                )
                self._connection.execute(
                    """
                    INSERT INTO meta
                        (symbol, fetched_at, last_bar_date, covered_from)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        fetched_at=excluded.fetched_at,
                        last_bar_date=COALESCE(
                            excluded.last_bar_date, meta.last_bar_date
                        ),
                        covered_from=CASE
                            WHEN excluded.covered_from IS NULL
                                THEN meta.covered_from
                            WHEN meta.covered_from IS NULL
                                THEN excluded.covered_from
                            ELSE MIN(meta.covered_from, excluded.covered_from)
                        END
                    """,
                    (
                        symbol,
                        _as_datetime(self._clock()).isoformat(),
                        last_date,
                        covered_from.isoformat() if covered_from else None,
                    ),
                )

        self._safe_call_cache(write)

    def get_history(self, entry: dict[str, Any], period_days: int) -> pd.DataFrame:
        """Return bars excluding an in-progress session until 14:45 Cairo.

        Bars dated today are neither cached nor returned until the EGX session
        close at 14:15 plus the 30-minute completion buffer has passed.
        """
        if isinstance(period_days, bool) or not isinstance(period_days, int):
            raise ValueError("period_days must be an integer.")
        if period_days < 1:
            raise ValueError("period_days must be positive.")
        symbol = entry["symbol"]
        now = self._clock()
        session_date = most_recent_completed_session(now)
        today = now.astimezone(_cairo_timezone()).date()
        cutoff = today - timedelta(days=period_days)
        if self._connection is None:
            self._record_history_decision(
                "full", None, session_date, "cache unavailable"
            )
            return _slice_bars(
                _completed_bars(self.provider.get_history(entry, period_days), now),
                cutoff,
            )

        cached = self._read_bars(symbol)
        meta = self._read_meta(symbol)
        if cached is None or meta is None or cached.empty:
            self._record_history_decision(
                "full", None, session_date, "cache miss or no cached bars"
            )
            fetched = _completed_bars(
                self.provider.get_history(entry, period_days),
                now,
            )
            self._write_bars(symbol, fetched, covered_from=cutoff)
            return _slice_bars(fetched, cutoff)

        cached["date"] = pd.to_datetime(cached["date"]).dt.date
        last_date = max(cached["date"])
        fetched_at = _as_datetime(meta[0])
        covered_from = meta[3]
        if covered_from is None or covered_from > cutoff:
            self._record_history_decision(
                "full",
                last_date,
                session_date,
                "cached coverage starts after the requested cutoff",
            )
            fetched = _completed_bars(
                self.provider.get_history(entry, period_days),
                now,
            )
            self._write_bars(
                symbol,
                fetched,
                replace_all=True,
                covered_from=cutoff,
            )
            return _slice_bars(fetched, cutoff)

        freshness_threshold = completed_close(session_date)
        is_fresh = fetched_at >= freshness_threshold
        reason = (
            "fetched_at reaches completed close for latest completed session"
            if is_fresh
            else "fetched_at precedes completed close for latest completed session"
        )
        if is_fresh:
            self._record_history_decision("fresh", last_date, session_date, reason)
            return _slice_bars(_completed_bars(cached, now), cutoff)

        self._record_history_decision("incremental", last_date, session_date, reason)
        last_date_for_fetch = last_date - timedelta(days=7)
        incremental_days = max(1, (today - last_date_for_fetch).days)
        fetch_days = max(1, incremental_days)
        incremental = _completed_bars(
            self.provider.get_history(entry, fetch_days),
            now,
        )
        overlap_changed = _overlap_differs(cached, incremental)
        if overlap_changed:
            self._record_history_decision(
                "full",
                last_date,
                session_date,
                "overlap values changed; possible retroactive adjustment",
            )
            full_days = max(36_500, period_days)
            full_history = _completed_bars(
                self.provider.get_history(entry, full_days),
                now,
            )
            self._write_bars(
                symbol,
                full_history,
                replace_all=True,
                covered_from=today - timedelta(days=full_days),
            )
            return _slice_bars(full_history, cutoff)

        merged = _merge_bars(cached, incremental)
        self._write_bars(symbol, incremental, covered_from=covered_from)
        return _slice_bars(merged, cutoff)

    def _record_history_decision(
        self,
        decision: str,
        cached_last_bar_date: date | None,
        session_date: date,
        reason: str,
    ) -> None:
        self.last_history_decision = {
            "decision": decision,
            "cached_last_bar_date": (
                cached_last_bar_date.isoformat() if cached_last_bar_date else None
            ),
            "last_completed_session": session_date.isoformat(),
            "reason": reason,
        }
        _logger.debug(
            "Stock cache decision=%s cached_last_bar_date=%s "
            "last_completed_session=%s reason=%s",
            decision,
            self.last_history_decision["cached_last_bar_date"],
            session_date.isoformat(),
            reason,
        )

    def _record_dividend_decision(
        self,
        decision: str,
        cached_last_bar_date: str | None,
        session_date: date,
        reason: str,
    ) -> None:
        self.last_dividend_decision = {
            "decision": decision,
            "cached_last_bar_date": cached_last_bar_date,
            "last_completed_session": session_date.isoformat(),
            "reason": reason,
        }
        _logger.debug(
            "Stock dividend cache decision=%s cached_last_bar_date=%s "
            "last_completed_session=%s reason=%s",
            decision,
            cached_last_bar_date,
            session_date.isoformat(),
            reason,
        )

    def get_dividends(self, entry: dict[str, Any], years: int) -> pd.DataFrame:
        """Return dividends from SQLite when recent, fetching and storing on miss."""
        if isinstance(years, bool) or not isinstance(years, int) or years < 1:
            raise ValueError("years must be a positive integer.")
        symbol = entry["symbol"]
        cached = self._read_dividends(symbol)
        meta = self._read_meta(symbol)
        now = self._clock()
        session_date = most_recent_completed_session(now)
        cached_last_bar = meta[1] if meta else None
        dividends_fetched_at = meta[2] if meta else None
        if (
            cached is not None
            and meta is not None
            and dividends_fetched_at is not None
            and _as_datetime(now) - _as_datetime(dividends_fetched_at)
            <= _DIVIDEND_TTL
        ):
            self._record_dividend_decision(
                "fresh",
                cached_last_bar,
                session_date,
                "seven-day dividend TTL is valid",
            )
            return _slice_dividends(cached, years, now)
        self._record_dividend_decision(
            "full",
            cached_last_bar,
            session_date,
            "no cached dividend history or seven-day TTL expired",
        )
        dividends = self.provider.get_dividends(entry, 30)
        self._write_dividends(symbol, dividends)
        return _slice_dividends(dividends, years, now)

    def _read_dividends(self, symbol: str) -> pd.DataFrame | None:
        def read() -> pd.DataFrame:
            assert self._connection is not None
            rows = self._connection.execute(
                "SELECT date, amount FROM dividends WHERE symbol = ? ORDER BY date",
                (symbol,),
            ).fetchall()
            return pd.DataFrame(rows, columns=["date", "amount"])

        return self._safe_call_cache(read)

    def _write_dividends(self, symbol: str, frame: pd.DataFrame) -> None:
        def write() -> None:
            assert self._connection is not None
            rows = [
                (symbol, _date_text(row.date), float(row.amount))
                for row in frame.itertuples(index=False)
            ]
            with self._connection:
                self._connection.execute(
                    "DELETE FROM dividends WHERE symbol = ?", (symbol,)
                )
                self._connection.executemany(
                    """
                    INSERT INTO dividends (symbol, date, amount)
                    VALUES (?, ?, ?)
                    ON CONFLICT(symbol, date) DO UPDATE SET amount=excluded.amount
                    """,
                    rows,
                )
                last_date = self._connection.execute(
                    "SELECT last_bar_date FROM meta WHERE symbol = ?", (symbol,)
                ).fetchone()
                self._connection.execute(
                    """
                    INSERT INTO meta
                        (symbol, fetched_at, last_bar_date, dividends_fetched_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        dividends_fetched_at=excluded.dividends_fetched_at
                    """,
                    (
                        symbol,
                        _as_datetime(self._clock()).isoformat(),
                        last_date[0] if last_date else None,
                        _as_datetime(self._clock()).isoformat(),
                    ),
                )

        self._safe_call_cache(write)

    def get_info_light(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Delegate uncached lightweight company information to the source."""
        return self.provider.get_info_light(entry)


def _date_text(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return pd.Timestamp(value).date().isoformat()


def _number_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _slice_bars(frame: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.date
    return result.loc[result["date"] >= cutoff].reset_index(drop=True)


def _slice_dividends(frame: pd.DataFrame, years: int, now: datetime) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    result["date"] = pd.to_datetime(result["date"]).dt.date
    cutoff = now.astimezone(_cairo_timezone()).date() - timedelta(days=365 * years)
    return result.loc[result["date"] >= cutoff].reset_index(drop=True)


def _merge_bars(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([left, right], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date
    return (
        combined.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
        .loc[:, list(_BAR_COLUMNS)]
    )


def _overlap_differs(cached: pd.DataFrame, fetched: pd.DataFrame) -> bool:
    if cached.empty or fetched.empty:
        return False
    left = cached.copy()
    right = fetched.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.date
    right["date"] = pd.to_datetime(right["date"]).dt.date
    overlap = left.merge(right, on="date", suffixes=("_cached", "_new"))
    for row in overlap.itertuples(index=False):
        for field in ("close", "adj_close"):
            if not _numeric_equal(
                getattr(row, f"{field}_cached"),
                getattr(row, f"{field}_new"),
            ):
                return True
    return False
