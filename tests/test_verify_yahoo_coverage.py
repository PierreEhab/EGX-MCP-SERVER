import json
import importlib.util
from pathlib import Path
import sys


_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "verify_yahoo_coverage.py"
_SPEC = importlib.util.spec_from_file_location("verify_yahoo_coverage", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
verify_yahoo_coverage = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verify_yahoo_coverage
_SPEC.loader.exec_module(verify_yahoo_coverage)


def test_only_flag_verifies_and_updates_requested_symbol(monkeypatch, tmp_path, capsys):
    catalog = [
        {
            "symbol": "HALN",
            "yahoo_ticker": "HALN.CA",
            "variant_of": None,
            "yahoo_available": None,
            "first_bar_date": None,
            "last_bar_date": None,
        },
        {
            "symbol": "OTHER",
            "yahoo_ticker": "OTHER.CA",
            "variant_of": None,
            "yahoo_available": None,
            "first_bar_date": None,
            "last_bar_date": None,
        },
    ]
    catalog_path = tmp_path / "symbols.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    calls = []

    monkeypatch.setattr(verify_yahoo_coverage, "CATALOG", catalog_path)
    monkeypatch.setattr(verify_yahoo_coverage, "MIN_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(
        verify_yahoo_coverage,
        "_coverage",
        lambda ticker: (
            calls.append(ticker) or (True, "2025-01-01", "2026-10-22")
        ),
    )

    verify_yahoo_coverage.main(["--only", "HALN"])

    updated = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert calls == ["HALN.CA"]
    assert updated[0]["yahoo_available"] is True
    assert updated[0]["first_bar_date"] == "2025-01-01"
    assert updated[0]["last_bar_date"] == "2026-10-22"
    assert updated[1]["yahoo_available"] is None
    assert "1 available" in capsys.readouterr().out
