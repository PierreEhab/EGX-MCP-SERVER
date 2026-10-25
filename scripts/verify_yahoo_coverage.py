"""Verify Yahoo Finance history coverage for non-variant EGX symbols."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import warnings
from contextlib import redirect_stdout
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import yfinance as yf


CATALOG = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "egx_mcp"
    / "data"
    / "egx_symbols.json"
)
MIN_INTERVAL_SECONDS = 0.5
MAX_ATTEMPTS = 3
_logger = logging.getLogger("verify_yahoo_coverage")


def _write_catalog(path: Path, entries: list[dict[str, Any]]) -> None:
    """Replace the catalog atomically so interrupted runs do not corrupt it."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _coverage(ticker: str) -> tuple[bool, str | None, str | None]:
    """Fetch maximum daily history, retrying errors with bounded backoff."""
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with redirect_stdout(sys.stderr):
                history = yf.Ticker(ticker).history(
                    period="max",
                    auto_adjust=False,
                )
            if history.empty or "Close" not in history:
                return False, None, None
            dates = history.index
            if getattr(dates, "tz", None) is not None:
                dates = dates.tz_convert("Africa/Cairo")
            normalized = sorted({timestamp.date().isoformat() for timestamp in dates})
            return bool(normalized), normalized[0] if normalized else None, normalized[-1] if normalized else None
        except Exception as exc:
            last_error = exc
            _logger.warning(
                "Yahoo coverage attempt %d/%d failed for %s: %s",
                attempt + 1,
                MAX_ATTEMPTS,
                ticker,
                exc,
            )
            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep((2**attempt) + random.uniform(0, 0.25))
    _logger.error("Yahoo coverage failed for %s: %s", ticker, last_error)
    return False, None, None


def main(argv: Sequence[str] | None = None) -> None:
    """Update catalog coverage fields, optionally for one requested symbol."""
    parser = argparse.ArgumentParser(
        description="Verify Yahoo Finance daily-history coverage for EGX symbols."
    )
    parser.add_argument(
        "--only",
        metavar="SYMBOL",
        help="verify only this catalog symbol (case-insensitive), for example HALN",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    warnings.filterwarnings("ignore", module="yfinance")
    entries = json.loads(CATALOG.read_text(encoding="utf-8"))
    selected = [
        entry
        for entry in entries
        if entry["variant_of"] is None
        and (
            args.only is None
            or entry["symbol"].casefold() == args.only.strip().casefold()
        )
    ]
    if args.only is not None and not selected:
        parser.error(
            f"symbol {args.only!r} was not found as a non-variant catalog entry"
        )

    available: list[str] = []
    missing: list[str] = []
    stale: list[str] = []
    last_call = 0.0
    today = datetime.now(ZoneInfo("Africa/Cairo")).date()

    for entry in selected:
        wait = MIN_INTERVAL_SECONDS - (time.monotonic() - last_call)
        if last_call and wait > 0:
            time.sleep(wait)
        entry["yahoo_available"], entry["first_bar_date"], entry["last_bar_date"] = _coverage(
            entry["yahoo_ticker"]
        )
        entry["catalog_checked_on"] = today.isoformat()
        last_call = time.monotonic()
        if entry["yahoo_available"]:
            available.append(entry["symbol"])
            last_date = date.fromisoformat(entry["last_bar_date"])
            if (today - last_date).days > 30:
                stale.append(entry["symbol"])
        else:
            missing.append(entry["symbol"])
        _write_catalog(CATALOG, entries)

    print(
        f"Yahoo coverage: {len(available)} available, {len(missing)} missing, "
        f"{len(stale)} stale (>30 days)."
    )
    if stale:
        print("Stale symbols: " + ", ".join(stale))


if __name__ == "__main__":
    main()
