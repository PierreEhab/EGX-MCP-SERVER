"""Catalog lookup and search for Egyptian Exchange companies."""

from __future__ import annotations

import difflib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


class SymbolNotFoundError(ValueError):
    """The catalog contains no company matching the supplied symbol or name."""

    code = "SYMBOL_NOT_FOUND"

    def __init__(self, query: str, suggestions: list[str]) -> None:
        self.query = query
        self.suggestions = suggestions
        message = f"No EGX company matches '{query}'."
        if suggestions:
            message += f" Did you mean: {', '.join(suggestions)}?"
        super().__init__(message)


class SymbolUnavailableError(ValueError):
    """A known catalog company has no available Yahoo data."""

    code = "SYMBOL_UNAVAILABLE"
    message = "Known EGX company, but no data is available from the Yahoo source"

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        super().__init__(self.message)


def require_yahoo_available(entry: dict[str, Any]) -> None:
    """Raise a typed error when Yahoo coverage explicitly marks a symbol missing.

    Unknown coverage (``None``) is not treated as unavailable; this lets tools
    attempt a fetch until the optional coverage verification has run.
    """
    if entry.get("yahoo_available") is False:
        raise SymbolUnavailableError(entry["symbol"])


@lru_cache(maxsize=1)
def _load_catalog() -> tuple[dict[str, Any], ...]:
    """Load the packaged catalog once, preserving its stable source order."""
    path = Path(__file__).parent / "data" / "egx_symbols.json"
    with path.open(encoding="utf-8") as catalog_file:
        catalog = json.load(catalog_file)
    if not isinstance(catalog, list):
        raise ValueError("EGX symbol catalog must be a JSON array.")
    return tuple(catalog)


def _entries() -> tuple[dict[str, Any], ...]:
    return _load_catalog()


def _without_yahoo_suffix(text: str) -> str:
    normalized = text.strip().casefold()
    return normalized[:-3] if normalized.endswith(".ca") else normalized


def _matches(entry: dict[str, Any], query: str) -> bool:
    needle = query.casefold()
    return needle in entry["symbol"].casefold() or needle in entry["name_en"].casefold() or any(
        needle in alias.casefold() for alias in entry["aliases"]
    )


def resolve_symbol(text: str) -> dict[str, Any] | list[dict[str, Any]]:
    """Resolve a code, Yahoo code, alias, or company-name fragment.

    Returns a catalog entry for a unique match, a list of candidates for an
    ambiguous fragment, or raises ``SymbolNotFoundError`` with close matches.
    """
    if not isinstance(text, str):
        raise TypeError("symbol or company name must be a string.")
    query = text.strip()
    normalized = _without_yahoo_suffix(query)
    entries = _entries()

    for entry in entries:
        if entry["symbol"].casefold() == normalized:
            return dict(entry)
    exact_aliases = [
        entry
        for entry in entries
        if any(alias.casefold() == query.casefold() for alias in entry["aliases"])
    ]
    if len(exact_aliases) == 1:
        return dict(exact_aliases[0])
    if len(exact_aliases) > 1:
        return [dict(entry) for entry in exact_aliases]

    matches = [
        entry
        for entry in entries
        if entry["variant_of"] is None and _matches(entry, query.casefold())
    ]
    if len(matches) == 1:
        return dict(matches[0])
    if len(matches) > 1:
        return [dict(entry) for entry in matches]

    searchable = {
        candidate
        for entry in entries
        if entry["variant_of"] is None
        for candidate in (
            entry["symbol"],
            entry["name_en"],
            *entry["aliases"],
        )
    }
    close = difflib.get_close_matches(
        query,
        sorted(searchable, key=str.casefold),
        n=5,
        cutoff=0.45,
    )
    suggestions: list[str] = []
    by_code = {entry["symbol"].casefold(): entry["symbol"] for entry in entries}
    for candidate in close:
        canonical = by_code.get(candidate.casefold())
        if canonical and canonical not in suggestions:
            suggestions.append(canonical)
        else:
            owner = next(
                (
                    entry["symbol"]
                    for entry in entries
                    if entry["variant_of"] is None
                    if candidate.casefold() == entry["name_en"].casefold()
                    or any(
                        candidate.casefold() == alias.casefold()
                        for alias in entry["aliases"]
                    )
                ),
                None,
            )
            if owner and owner not in suggestions:
                suggestions.append(owner)
    raise SymbolNotFoundError(query, suggestions[:5])


def search(
    query: str,
    limit: int = 50,
    include_variants: bool = False,
) -> list[dict[str, Any]]:
    """Find companies by case-insensitive code, alias, or name substring."""
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer.")
    if not isinstance(include_variants, bool):
        raise TypeError("include_variants must be a boolean.")
    needle = query.strip()
    if not needle:
        return []
    return [
        dict(entry)
        for entry in _entries()
        if (include_variants or entry["variant_of"] is None)
        and _matches(entry, needle.casefold())
    ][:limit]


def list_symbols(
    sector: str | None = None,
    include_variants: bool = False,
    only_available: bool = True,
    include_unavailable: bool = False,
) -> list[dict[str, Any]]:
    """List catalog entries, excluding unavailable symbols and variants by default.

    Unknown availability is retained until Yahoo coverage has been verified;
    only symbols explicitly marked unavailable are excluded.
    ``include_unavailable`` overrides the availability filter and adds an
    ``unavailable`` boolean to each returned entry.
    """
    if sector is not None and not isinstance(sector, str):
        raise TypeError("sector must be a string or None.")
    if not all(
        isinstance(flag, bool)
        for flag in (include_variants, only_available, include_unavailable)
    ):
        raise TypeError(
            "include_variants, only_available, and include_unavailable must be booleans."
        )
    requested_sector = sector.casefold() if sector else None
    results: list[dict[str, Any]] = []
    for entry in _entries():
        if not include_variants and entry["variant_of"] is not None:
            continue
        if (
            not include_unavailable
            and only_available
            and entry["yahoo_available"] is False
        ):
            continue
        if (
            requested_sector is not None
            and entry["sector"].casefold() != requested_sector
        ):
            continue
        result = dict(entry)
        if include_unavailable:
            result["unavailable"] = entry["yahoo_available"] is False
        results.append(result)
    return results


def sectors() -> list[str]:
    """Return the distinct catalog sector labels, excluding ``unknown``."""
    return sorted(
        {entry["sector"] for entry in _entries() if entry["sector"] != "unknown"},
        key=str.casefold,
    )
