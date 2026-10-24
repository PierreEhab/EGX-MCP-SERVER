from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from egx_mcp import stock_provider
from egx_mcp.stock_cache import (
    CachedStockProvider,
    most_recent_completed_session,
)


CAIRO = ZoneInfo("Africa/Cairo")
ENTRY = {"symbol": "TEST", "yahoo_ticker": "TEST.CA"}


class FakeProvider:
    def __init__(self, factory):
        self.factory = factory
        self.history_calls: list[int] = []
        self.dividend_calls = 0
        self.dividend_factory = lambda years: pd.DataFrame(columns=["date", "amount"])

    def get_history(self, entry, period_days):
        self.history_calls.append(period_days)
        return self.factory(period_days)

    def get_dividends(self, entry, years):
        self.dividend_calls += 1
        return self.dividend_factory(years)

    def get_info_light(self, entry):
        return {"name": "Test"}


def _frame(today, value=100.0, start_days=40):
    dates = [today - timedelta(days=offset) for offset in range(start_days, -1, -1)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": [value] * len(dates),
            "high": [value] * len(dates),
            "low": [value] * len(dates),
            "close": [value] * len(dates),
            "adj_close": [value] * len(dates),
            "volume": [1000.0] * len(dates),
        }
    )


def _clock(value: datetime):
    return lambda: value


def test_recent_completed_session_skips_weekends_and_current_session() -> None:
    assert most_recent_completed_session(
        datetime(2026, 10, 25, 11, 0, tzinfo=CAIRO)
    ).isoformat() == "2026-10-22"
    assert most_recent_completed_session(
        datetime(2026, 10, 25, 15, 0, tzinfo=CAIRO)
    ).isoformat() == "2026-10-25"
    assert most_recent_completed_session(
        datetime(2026, 10, 26, 12, 0, tzinfo=CAIRO)
    ).isoformat() == "2026-10-25"


def test_cache_miss_fetches_and_fresh_cache_is_a_hit(tmp_path: Path) -> None:
    now = datetime(2026, 10, 25, 16, 0, tzinfo=CAIRO)
    provider = FakeProvider(lambda period: _frame(now.date()))
    cached = CachedStockProvider(
        provider,
        database_path=tmp_path / "cache.sqlite3",
        clock=_clock(now),
    )

    first = cached.get_history(ENTRY, 30)
    second = cached.get_history(ENTRY, 30)

    assert len(first) == len(second)
    assert provider.history_calls == [30]
    cached.close()


@pytest.mark.parametrize(
    ("request_time", "last_bar_date", "expected_decision"),
    [
        (datetime(2026, 10, 23, 16, 0, tzinfo=CAIRO), "2026-10-22", "fresh"),
        (datetime(2026, 10, 24, 16, 0, tzinfo=CAIRO), "2026-10-22", "fresh"),
        (datetime(2026, 10, 25, 11, 0, tzinfo=CAIRO), "2026-10-22", "fresh"),
        (datetime(2026, 10, 25, 15, 0, tzinfo=CAIRO), "2026-10-22", "incremental"),
    ],
)
def test_freshness_decision_for_friday_saturday_and_sunday(
    tmp_path: Path,
    request_time: datetime,
    last_bar_date: str,
    expected_decision: str,
) -> None:
    provider = FakeProvider(lambda period: _frame(request_time.date()))
    database = tmp_path / "cache.sqlite3"
    seed_time = datetime(2026, 10, 22, 16, 0, tzinfo=CAIRO)
    cached = CachedStockProvider(provider, database_path=database, clock=_clock(seed_time))
    seed = _frame(pd.Timestamp(last_bar_date).date(), start_days=365)
    cached._write_bars(
        "TEST",
        seed,
        covered_from=request_time.date() - timedelta(days=365),
    )
    cached.close()

    later_provider = FakeProvider(lambda period: _frame(request_time.date(), start_days=365))
    reopened = CachedStockProvider(
        later_provider,
        database_path=database,
        clock=_clock(request_time),
    )
    calls_before = stock_provider.yahoo_call_count()
    reopened.get_history(ENTRY, 30)

    assert reopened.last_history_decision["decision"] == expected_decision
    if expected_decision == "fresh":
        assert later_provider.history_calls == []
        assert stock_provider.yahoo_call_count() == calls_before
    else:
        assert later_provider.history_calls
    reopened.close()


def test_cache_does_not_expire_after_six_hours_when_session_is_complete(
    tmp_path: Path,
) -> None:
    request_time = datetime(2026, 10, 26, 12, 0, tzinfo=CAIRO)
    provider = FakeProvider(lambda period: _frame(request_time.date(), start_days=365))
    cached = CachedStockProvider(
        provider,
        database_path=tmp_path / "cache.sqlite3",
        clock=_clock(datetime(2026, 10, 26, 5, 0, tzinfo=CAIRO)),
    )
    cached._write_bars(
        "TEST",
        _frame(datetime(2026, 10, 25).date(), start_days=365),
        covered_from=request_time.date() - timedelta(days=365),
    )
    cached._clock = _clock(request_time)

    cached.get_history(ENTRY, 30)

    assert cached.last_history_decision["decision"] == "fresh"
    assert provider.history_calls == []
    cached.close()


def test_partial_current_session_bar_is_not_stored_or_returned_then_refetched(
    tmp_path: Path,
) -> None:
    sunday = datetime(2026, 10, 25, 11, 0, tzinfo=CAIRO)
    after_close = datetime(2026, 10, 25, 16, 0, tzinfo=CAIRO)
    thursday = sunday.date() - timedelta(days=3)
    partial = pd.concat(
        [
            _frame(thursday, value=100.0, start_days=0),
            _frame(sunday.date(), value=101.0, start_days=0),
        ],
        ignore_index=True,
    )
    final = pd.concat(
        [
            _frame(thursday, value=100.0, start_days=0),
            _frame(sunday.date(), value=102.0, start_days=0),
        ],
        ignore_index=True,
    )
    provider = FakeProvider(lambda _: partial)
    cached = CachedStockProvider(
        provider,
        database_path=tmp_path / "cache.sqlite3",
        clock=_clock(sunday),
    )

    first = cached.get_history(ENTRY, 30)
    assert sunday.date() not in set(first["date"])
    stored = cached._read_bars("TEST")
    assert sunday.date() not in set(pd.to_datetime(stored["date"]).dt.date)

    provider.factory = lambda _: final
    cached._clock = _clock(after_close)
    second = cached.get_history(ENTRY, 30)

    assert provider.history_calls == [30, 10]
    assert second.iloc[-1]["date"] == sunday.date()
    assert second.iloc[-1]["close"] == 102.0
    cached.close()


def test_coverage_expands_for_longer_request_then_stays_fresh(tmp_path: Path) -> None:
    now = datetime(2026, 10, 25, 16, 0, tzinfo=CAIRO)
    provider = FakeProvider(lambda days: _frame(now.date(), start_days=days))
    cached = CachedStockProvider(
        provider,
        database_path=tmp_path / "cache.sqlite3",
        clock=_clock(now),
    )

    short = cached.get_history(ENTRY, 30)
    long = cached.get_history(ENTRY, 365)
    repeated = cached.get_history(ENTRY, 365)

    assert len(short) < len(long)
    assert len(long) > 300
    assert len(repeated) == len(long)
    assert provider.history_calls == [30, 365]
    cached.close()


def test_holiday_without_new_bar_refreshes_once_then_is_fresh(tmp_path: Path) -> None:
    sunday = datetime(2026, 10, 25, 16, 0, tzinfo=CAIRO)
    thursday = sunday.date() - timedelta(days=3)
    provider = FakeProvider(lambda _: _frame(thursday, start_days=365))
    cached = CachedStockProvider(
        provider,
        database_path=tmp_path / "cache.sqlite3",
        clock=_clock(sunday),
    )

    cached.get_history(ENTRY, 30)
    first_calls = len(provider.history_calls)
    cached.get_history(ENTRY, 30)

    assert first_calls == 1
    assert len(provider.history_calls) == 1
    assert cached.last_history_decision["decision"] == "fresh"
    cached.close()


def test_overlap_volume_difference_does_not_trigger_full_refetch(tmp_path: Path) -> None:
    initial_now = datetime(2026, 10, 25, 16, 0, tzinfo=CAIRO)
    next_now = datetime(2026, 10, 26, 16, 0, tzinfo=CAIRO)
    provider = FakeProvider(lambda _: _frame(initial_now.date(), start_days=365))
    cached = CachedStockProvider(
        provider,
        database_path=tmp_path / "cache.sqlite3",
        clock=_clock(initial_now),
    )
    cached.get_history(ENTRY, 30)

    def volume_change(period):
        result = _frame(next_now.date(), start_days=365)
        result["volume"] = result["volume"] + 50
        return result

    provider.factory = volume_change
    cached._clock = _clock(next_now)
    cached.get_history(ENTRY, 30)

    assert provider.history_calls == [30, 8]
    assert cached.last_history_decision["decision"] == "incremental"
    cached.close()


def test_young_listing_with_no_old_bars_is_fresh_after_initial_fetch(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 10, 24, 16, 0, tzinfo=CAIRO)
    listing_start = now.date() - timedelta(days=10)
    provider = FakeProvider(lambda _: _frame(now.date(), start_days=10))
    cached = CachedStockProvider(
        provider,
        database_path=tmp_path / "cache.sqlite3",
        clock=_clock(now),
    )

    first = cached.get_history(ENTRY, 365)
    second = cached.get_history(ENTRY, 365)

    assert first["date"].min() == listing_start
    assert len(second) == len(first)
    assert provider.history_calls == [365]
    cached.close()


def test_stale_cache_fetches_incrementally_and_upserts(tmp_path: Path) -> None:
    initial_now = datetime(2026, 10, 25, 16, 0, tzinfo=CAIRO)
    next_now = datetime(2026, 10, 26, 16, 0, tzinfo=CAIRO)
    provider = FakeProvider(lambda period: _frame(initial_now.date()))
    cached = CachedStockProvider(
        provider,
        database_path=tmp_path / "cache.sqlite3",
        clock=_clock(initial_now),
    )
    cached.get_history(ENTRY, 30)

    provider.factory = lambda period: _frame(next_now.date())
    cached._clock = _clock(next_now)
    result = cached.get_history(ENTRY, 30)

    assert provider.history_calls == [30, 8]
    assert result["date"].max() == next_now.date()
    cached.close()


def test_overlap_adjustment_change_triggers_full_history_refetch(
    tmp_path: Path,
) -> None:
    initial_now = datetime(2026, 10, 25, 16, 0, tzinfo=CAIRO)
    next_now = datetime(2026, 10, 26, 16, 0, tzinfo=CAIRO)
    full_fetch = False

    def initial_factory(period):
        return _frame(initial_now.date(), value=100.0)

    provider = FakeProvider(initial_factory)
    cached = CachedStockProvider(
        provider,
        database_path=tmp_path / "cache.sqlite3",
        clock=_clock(initial_now),
    )
    cached.get_history(ENTRY, 30)

    def changed_factory(period):
        nonlocal full_fetch
        if period >= 36_500:
            full_fetch = True
            return _frame(next_now.date(), value=99.0, start_days=7)
        return _frame(next_now.date(), value=99.0, start_days=7)

    provider.factory = changed_factory
    cached._clock = _clock(next_now)
    result = cached.get_history(ENTRY, 30)

    assert provider.history_calls == [30, 8, 36_500]
    assert full_fetch
    assert set(result["close"]) == {99.0}
    cached.close()


def test_dividends_are_fetched_full_once_then_filtered_from_cache(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 10, 24, 16, 0, tzinfo=CAIRO)
    provider = FakeProvider(lambda period: _frame(now.date(), start_days=365))
    dates = [now.date() - timedelta(days=365 * year + 1) for year in range(24)]
    provider.dividend_factory = lambda years: pd.DataFrame(
        {"date": dates, "amount": [0.25] * len(dates)}
    )
    cached = CachedStockProvider(
        provider,
        database_path=tmp_path / "cache.sqlite3",
        clock=_clock(now),
    )

    all_dividends = cached.get_dividends(ENTRY, 30)
    five_years = cached.get_dividends(ENTRY, 5)

    assert len(all_dividends) == 24
    assert len(five_years) == 5
    assert provider.dividend_calls == 1
    cached.close()


def test_empty_dividend_history_is_cached_and_fetched_once(tmp_path: Path) -> None:
    now = datetime(2026, 10, 24, 16, 0, tzinfo=CAIRO)
    provider = FakeProvider(lambda period: _frame(now.date()))
    cached = CachedStockProvider(
        provider,
        database_path=tmp_path / "cache.sqlite3",
        clock=_clock(now),
    )

    assert cached.get_dividends(ENTRY, 5).empty
    assert cached.get_dividends(ENTRY, 5).empty

    assert provider.dividend_calls == 1
    cached.close()


def test_dividend_ttl_is_separate_and_expires_after_seven_days(tmp_path: Path) -> None:
    first_now = datetime(2026, 10, 24, 16, 0, tzinfo=CAIRO)
    provider = FakeProvider(lambda period: _frame(first_now.date(), start_days=365))
    provider.dividend_factory = lambda years: pd.DataFrame(columns=["date", "amount"])
    cached = CachedStockProvider(
        provider,
        database_path=tmp_path / "cache.sqlite3",
        clock=_clock(first_now),
    )

    cached.get_dividends(ENTRY, 5)
    cached._clock = _clock(first_now + timedelta(days=6))
    cached.get_dividends(ENTRY, 5)
    assert provider.dividend_calls == 1

    cached._clock = _clock(first_now + timedelta(days=8))
    cached.get_dividends(ENTRY, 5)

    assert provider.dividend_calls == 2
    cached.close()


def test_bar_refresh_does_not_remove_dividend_rows(tmp_path: Path) -> None:
    first_now = datetime(2026, 10, 25, 16, 0, tzinfo=CAIRO)
    next_now = datetime(2026, 10, 26, 16, 0, tzinfo=CAIRO)
    provider = FakeProvider(lambda period: _frame(first_now.date(), start_days=365))
    provider.dividend_factory = lambda years: pd.DataFrame(
        {"date": [first_now.date() - timedelta(days=30)], "amount": [0.5]}
    )
    cached = CachedStockProvider(
        provider,
        database_path=tmp_path / "cache.sqlite3",
        clock=_clock(first_now),
    )
    cached.get_dividends(ENTRY, 5)
    cached.get_history(ENTRY, 365)
    provider.factory = lambda period: _frame(next_now.date(), start_days=365)
    cached._clock = _clock(next_now)

    cached.get_history(ENTRY, 365)

    assert len(cached.get_dividends(ENTRY, 5)) == 1
    assert provider.dividend_calls == 1
    cached.close()


def test_cache_decisions_are_logged_at_debug_level(tmp_path: Path, caplog) -> None:
    caplog.set_level("DEBUG")
    now = datetime(2026, 10, 24, 16, 0, tzinfo=CAIRO)
    provider = FakeProvider(lambda period: _frame(datetime(2026, 10, 22).date(), start_days=365))
    cached = CachedStockProvider(
        provider,
        database_path=tmp_path / "cache.sqlite3",
        clock=_clock(now),
    )
    cached.get_history(ENTRY, 30)
    cached.get_history(ENTRY, 30)

    assert any(
        record.levelname == "DEBUG"
        and "decision=fresh" in record.getMessage()
        and "cached_last_bar_date=2026-10-22" in record.getMessage()
        and "last_completed_session=2026-10-22" in record.getMessage()
        for record in caplog.records
    )
    cached.close()


def test_corrupt_database_is_rebuilt(tmp_path: Path) -> None:
    database = tmp_path / "broken.sqlite3"
    database.write_bytes(b"not a sqlite database")
    now = datetime(2026, 10, 25, 16, 0, tzinfo=CAIRO)
    provider = FakeProvider(lambda period: _frame(now.date()))

    cached = CachedStockProvider(
        provider,
        database_path=database,
        clock=_clock(now),
    )
    result = cached.get_history(ENTRY, 30)

    assert not result.empty
    assert provider.history_calls == [30]
    cached.close()
