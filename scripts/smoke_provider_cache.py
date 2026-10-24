"""Exercise Yahoo stock history and the persistent cache for a few EGX symbols."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from egx_mcp.stock_cache import CachedStockProvider, cache_database_path
from egx_mcp.stock_provider import (
    EmptyStockDataError,
    StockDataError,
    YahooProvider,
    yahoo_call_count,
)
from egx_mcp.symbols import (
    SymbolUnavailableError,
    resolve_symbol,
    require_yahoo_available,
)


def _elapsed_call(operation: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


def _print_history(label: str, entry: dict[str, Any], cache: CachedStockProvider) -> None:
    before = yahoo_call_count()
    bars, elapsed = _elapsed_call(lambda: cache.get_history(entry, 365))
    yahoo_calls = yahoo_call_count() - before
    latest = bars.iloc[-1]
    decision = cache.last_history_decision
    print(
        f"{label}: {elapsed:.3f}s, decision={decision['decision']}, "
        f"yfinance calls={yahoo_calls}, {len(bars)} bars, "
        f"{bars.iloc[0]['date']} to {latest['date']}, "
        f"latest close={latest['close']:.4f}"
    )


def _print_dividends(
    label: str,
    entry: dict[str, Any],
    cache: CachedStockProvider,
) -> None:
    before = yahoo_call_count()
    dividends, elapsed = _elapsed_call(lambda: cache.get_dividends(entry, 5))
    yahoo_calls = yahoo_call_count() - before
    decision = cache.last_dividend_decision
    first_date = dividends.iloc[0]["date"] if not dividends.empty else None
    last_date = dividends.iloc[-1]["date"] if not dividends.empty else None
    print(
        f"{label}: {elapsed:.3f}s, decision={decision['decision']}, "
        f"yfinance calls={yahoo_calls}, {len(dividends)} dividend rows, "
        f"{first_date} to {last_date}"
    )


def _print_database_summary(path: Path) -> None:
    counts = {"bars": 0, "dividends": 0, "meta": 0}
    if path.exists():
        try:
            with sqlite3.connect(path) as connection:
                for table in counts:
                    counts[table] = int(
                        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    )
        except sqlite3.DatabaseError as exc:
            print(f"Cache table counts unavailable: {exc}")
    size_kb = path.stat().st_size / 1024 if path.exists() else 0.0
    print(f"Cache database: {path}")
    print(f"Cache size: {size_kb:.2f} KB")
    print(
        "Cache rows: "
        f"bars={counts['bars']}, dividends={counts['dividends']}, meta={counts['meta']}"
    )


def main() -> None:
    source = YahooProvider()
    cache = CachedStockProvider(source)
    database_path = cache_database_path()

    comi = resolve_symbol("COMI")
    if isinstance(comi, list):
        raise RuntimeError("COMI unexpectedly resolved to multiple catalog entries.")
    print("COMI history:")
    _print_history("First fetch", comi, cache)
    _print_history("Second fetch", comi, cache)

    print("COMI dividends:")
    _print_dividends("First fetch", comi, cache)
    _print_dividends("Second fetch", comi, cache)

    qnbe = resolve_symbol("QNBE")
    if isinstance(qnbe, list):
        raise RuntimeError("QNBE unexpectedly resolved to multiple catalog entries.")
    try:
        require_yahoo_available(qnbe)
        cache.get_history(qnbe, 365)
    except (SymbolUnavailableError, EmptyStockDataError, StockDataError) as exc:
        print(f"QNBE: error code={getattr(exc, 'code', 'DATA_SOURCE_ERROR')}; {exc}")
    else:
        print("QNBE: no error; Yahoo returned daily history.")

    _print_database_summary(database_path)
    cache.close()


if __name__ == "__main__":
    main()
