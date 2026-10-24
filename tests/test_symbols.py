import json
from pathlib import Path

import pytest

from egx_mcp import symbols


def test_resolves_code_yahoo_code_alias_and_name() -> None:
    assert symbols.resolve_symbol("comi")["symbol"] == "COMI"
    assert symbols.resolve_symbol("ACTF.CA")["symbol"] == "ACTF"
    assert symbols.resolve_symbol("CIB")["symbol"] == "COMI"
    assert symbols.resolve_symbol("Telecom Egypt")["symbol"] == "ETEL"


def test_ambiguous_name_fragment_returns_candidates() -> None:
    candidates = symbols.resolve_symbol("Alexandria")

    assert isinstance(candidates, list)
    assert len(candidates) > 1
    assert all(entry["symbol"] for entry in candidates)


def test_not_found_error_includes_close_suggestions() -> None:
    with pytest.raises(symbols.SymbolNotFoundError) as error:
        symbols.resolve_symbol("COMX")

    assert error.value.code == "SYMBOL_NOT_FOUND"
    assert len(error.value.suggestions) <= 5
    assert "COMI" in error.value.suggestions


def test_variants_are_hidden_from_search_and_default_listing() -> None:
    default_symbols = symbols.list_symbols(only_available=False)
    default_search = symbols.search("Valmore")
    variant_search = symbols.search("Valmore", include_variants=True)
    variants = symbols.list_symbols(include_variants=True, only_available=False)

    assert "VLMRA" not in {entry["symbol"] for entry in default_symbols}
    assert "VLMRA" not in {entry["symbol"] for entry in default_search}
    assert "VLMRA" in {entry["symbol"] for entry in variant_search}
    assert next(entry for entry in variants if entry["symbol"] == "VLMRA")[
        "variant_of"
    ] == "VLMR"


@pytest.mark.parametrize(
    ("query", "expected_symbol"),
    [
        ("valmore", "VLMR"),
        ("Faisal", "FAIT"),
        ("Saudi Egyptian", "SEIG"),
    ],
)
def test_name_resolution_ignores_share_line_variants(
    query: str,
    expected_symbol: str,
) -> None:
    result = symbols.resolve_symbol(query)
    assert isinstance(result, dict)
    assert result["symbol"] == expected_symbol


def test_variant_code_and_exact_variant_alias_still_resolve() -> None:
    assert symbols.resolve_symbol("VLMRA")["symbol"] == "VLMRA"
    assert symbols.resolve_symbol("FAITA")["symbol"] == "FAITA"


def test_sector_filter_keeps_source_metadata_and_known_sector() -> None:
    financials = symbols.list_symbols(sector="financials", only_available=False)

    assert financials
    assert all(entry["sector"] == "Financials" for entry in financials)
    assert all(entry["sector_source"] in {"known", "inferred"} for entry in financials)
    assert next(entry for entry in financials if entry["symbol"] == "COMI")[
        "sector_source"
    ] == "known"


def _unavailable_catalog_entry() -> dict[str, object]:
    return {
        "symbol": "TEST",
        "yahoo_ticker": "TEST.CA",
        "name_en": "Test Company",
        "name_ar": None,
        "sector": "Financials",
        "sector_source": "known",
        "aliases": ["Test"],
        "variant_of": None,
        "yahoo_available": False,
        "first_bar_date": None,
        "last_bar_date": None,
    }


def test_unavailable_catalog_symbol_resolves_but_has_a_typed_source_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _unavailable_catalog_entry()
    monkeypatch.setattr(symbols, "_entries", lambda: (entry,))

    resolved = symbols.resolve_symbol("test")
    assert isinstance(resolved, dict)
    assert resolved["symbol"] == "TEST"
    assert resolved["yahoo_available"] is False

    with pytest.raises(symbols.SymbolUnavailableError) as error:
        symbols.require_yahoo_available(resolved)
    assert error.value.code == "SYMBOL_UNAVAILABLE"
    assert str(error.value) == (
        "Known EGX company, but no data is available from the Yahoo source"
    )
    assert error.value.symbol == "TEST"


def test_unavailable_symbols_are_hidden_by_default_and_flagged_when_included(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _unavailable_catalog_entry()
    monkeypatch.setattr(symbols, "_entries", lambda: (entry,))

    assert symbols.list_symbols() == []
    included = symbols.list_symbols(include_unavailable=True)

    assert included[0]["symbol"] == "TEST"
    assert included[0]["unavailable"] is True


def test_catalog_schema_uniqueness_and_variant_references() -> None:
    path = Path(symbols.__file__).parent / "data" / "egx_symbols.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "symbol",
        "yahoo_ticker",
        "name_en",
        "name_ar",
        "sector",
        "sector_source",
        "aliases",
        "variant_of",
        "yahoo_available",
        "first_bar_date",
        "last_bar_date",
    }
    codes = [entry["symbol"] for entry in catalog]
    assert len(codes) == len(set(codes))
    assert all(set(entry) == required for entry in catalog)
    assert all(
        entry["sector_source"] in {"known", "inferred", "unknown"}
        for entry in catalog
    )
    allowed_sectors = {
        "Financials",
        "Real Estate",
        "Industrials",
        "Materials",
        "Energy",
        "Healthcare",
        "Consumer Staples",
        "Consumer Discretionary",
        "Information Technology",
        "Communication Services",
        "Utilities",
        "unknown",
    }
    assert all(entry["sector"] in allowed_sectors for entry in catalog)
    expected_sectors = {
        "ATQA": "Materials",
        "EGAS": "Utilities",
        "MOSC": "Consumer Staples",
        "COSG": "Consumer Staples",
        "ZEOT": "Consumer Staples",
        "ZMID": "Real Estate",
        "OBRI": "Real Estate",
        "NARE": "Real Estate",
        "TANM": "Real Estate",
        "ARAB": "Real Estate",
        "GGCC": "Real Estate",
        "CFGH": "Consumer Discretionary",
        "GGRN": "Consumer Staples",
        "KRDI": "Consumer Staples",
        "LUTS": "Consumer Staples",
        "ELNA": "Consumer Staples",
        "ENGC": "Industrials",
        "DAPH": "Industrials",
        "GDWA": "Industrials",
    }
    sector_by_symbol = {entry["symbol"]: entry for entry in catalog}
    for code, expected_sector in expected_sectors.items():
        assert sector_by_symbol[code]["sector"] == expected_sector
        assert sector_by_symbol[code]["sector_source"] == "inferred"
    assert "Domty" not in sector_by_symbol["DOMT"]["aliases"]
    assert "DOMTY" in sector_by_symbol["DOMT"]["aliases"]
    assert all(isinstance(entry["aliases"], list) for entry in catalog)
    assert all(
        entry["variant_of"] is None or entry["variant_of"] in codes
        for entry in catalog
    )
    for entry in catalog:
        assert isinstance(entry["symbol"], str) and entry["symbol"]
        assert isinstance(entry["yahoo_ticker"], str) and entry["yahoo_ticker"]
        assert isinstance(entry["name_en"], str) and entry["name_en"]
        assert entry["name_ar"] is None or isinstance(entry["name_ar"], str)
        assert isinstance(entry["sector"], str) and entry["sector"]
        assert isinstance(entry["aliases"], list)
        assert all(isinstance(alias, str) for alias in entry["aliases"])
        assert entry["yahoo_available"] in {None, True, False}
        assert entry["first_bar_date"] is None or isinstance(
            entry["first_bar_date"], str
        )
        assert entry["last_bar_date"] is None or isinstance(
            entry["last_bar_date"], str
        )
