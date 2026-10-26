import asyncio
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from egx_mcp import server as server_module
from egx_mcp import stock_provider
from egx_mcp.server import (
    analysis_snapshot,
    build_trade_plan,
    compare_stocks,
    get_egypt_macro_data,
    find_stock,
    get_dividends,
    get_stock_fundamentals,
    get_stock_history,
    get_stock_summary,
    list_stocks,
    research_company,
    stock_vs_index,
)


def _bars(count: int = 400) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=count)
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0 + i for i in range(count)],
            "high": [102.0 + i for i in range(count)],
            "low": [98.0 + i for i in range(count)],
            "close": [100.0 + i for i in range(count)],
            "adj_close": [99.0 + i for i in range(count)],
            "volume": [1000.0 + i for i in range(count)],
        }
    )


class FakeProvider:
    def __init__(self):
        self.history_calls: list[str] = []
        self.dividend_calls: list[str] = []
        self.fail_symbols: set[str] = set()

    def get_history(self, entry, period_days):
        self.history_calls.append(entry["symbol"])
        if entry["symbol"] in self.fail_symbols:
            raise stock_provider.EmptyStockDataError("No stock data.")
        bars = _bars()
        cutoff = bars["date"].max() - pd.Timedelta(days=period_days)
        return bars.loc[bars["date"] >= cutoff].reset_index(drop=True)

    def get_dividends(self, entry, years):
        self.dividend_calls.append(entry["symbol"])
        return pd.DataFrame(
            {
                "date": [date.today() - timedelta(days=30)],
                "amount": [1.0],
            }
        )

    def get_info_light(self, entry):
        return {
            "name": "Yahoo name",
            "market_cap": 1000.0,
            "trailing_pe": None,
            "forward_pe": None,
            "price_to_book": None,
            "return_on_equity": None,
            "profit_margins": None,
            "operating_margins": None,
            "dividend_yield": None,
            "payout_ratio": None,
            "total_debt": None,
            "total_cash": None,
            "week52_high": 500.0,
            "week52_low": 250.0,
            "currency": "EGP",
            "financial_currency": None,
        }


@pytest.fixture
def fake_provider(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(server_module, "_get_stock_provider", lambda: provider)
    return provider


def test_list_stocks_and_find_stock_include_catalog_metadata() -> None:
    listed = asyncio.run(list_stocks(limit=3))
    found = asyncio.run(find_stock("Telecom Egypt"))

    assert len(listed["stocks"]) == 3
    assert listed["source"] == "EGX symbol catalog"
    assert found["stocks"][0]["symbol"] == "ETEL"
    assert found["stocks"][0]["sector_source"] == "known"


def test_stock_catalog_tools_default_to_compact_rows_and_support_verbose() -> None:
    compact_list = asyncio.run(list_stocks(limit=1))
    detailed_list = asyncio.run(list_stocks(limit=1, verbose=True))
    compact_search = asyncio.run(find_stock("COMI"))
    detailed_search = asyncio.run(find_stock("COMI", verbose=True))

    assert set(compact_list["stocks"][0]) == {
        "symbol",
        "name_en",
        "sector",
        "sector_source",
        "yahoo_available",
        "catalog_checked_on",
    }
    assert "aliases" in detailed_list["stocks"][0]
    assert set(compact_search["stocks"][0]) == set(compact_list["stocks"][0])
    assert "aliases" in detailed_search["stocks"][0]
    assert compact_list["as_of"] is None
    assert "stale" not in compact_list["stocks"][0]


def test_stock_history_thursday_weekly_sampling_and_period_clamp(fake_provider) -> None:
    result = asyncio.run(
        get_stock_history("COMI", period_days=9500, frequency="weekly", max_rows=3)
    )

    assert result["symbol"] == "COMI"
    assert result["period_days_applied"] == 9000
    assert result["frequency"] == "weekly"
    assert result["observations"] == 3
    assert result["price_basis"] == "Close (split-adjusted)"
    assert fake_provider.history_calls == ["COMI"]


def test_summary_and_fundamentals_have_requested_metadata(fake_provider) -> None:
    summary = asyncio.run(get_stock_summary("COMI"))
    fundamentals = asyncio.run(get_stock_fundamentals("COMI"))

    assert summary["latest_close"] is not None
    assert summary["period_days_applied"] == 365
    assert summary["price_basis"].startswith("Close (split-adjusted)")
    assert fundamentals["name"] == "Yahoo name"
    assert fundamentals["sector"] == "Financials"
    assert fundamentals["sector_source"] == "known"
    assert fundamentals["avg_volume_20d"] is not None
    assert fundamentals["trailing_pe"] is None
    assert fundamentals["data_quality"]
    assert set(fundamentals["valuation"]) == {
        "trailing_pe",
        "forward_pe",
        "price_to_book",
    }
    assert "dividend_yield_ttm" in fundamentals["dividends"]
    assert fundamentals["financial_currency"] is None


def test_fundamentals_merge_yahoo_metrics_and_warn_on_yield_discrepancy(
    fake_provider,
) -> None:
    def detailed_info(entry):
        return {
            **FakeProvider().get_info_light(entry),
            "trailing_pe": 10.0,
            "forward_pe": 9.0,
            "price_to_book": 1.2,
            "return_on_equity": 0.12,
            "profit_margins": 0.15,
            "operating_margins": 0.08,
            "dividend_yield": 0.000001,
            "payout_ratio": 0.4,
            "total_debt": 200.0,
            "total_cash": 80.0,
            "financial_currency": "EGP",
        }

    fake_provider.get_info_light = detailed_info

    result = asyncio.run(get_stock_fundamentals("COMI"))

    assert result["valuation"] == {
        "trailing_pe": 10.0,
        "forward_pe": 9.0,
        "price_to_book": 1.2,
    }
    assert result["profitability"]["return_on_equity"] == 0.12
    assert result["dividends"]["dividend_yield"] == 0.000001
    assert result["dividends"]["dividend_yield_ttm"] == result["dividend_yield_ttm"]
    assert result["dividends"]["note"]
    assert "100x" in result["dividends"]["note"]
    assert result["balance_sheet"] == {"total_debt": 200.0, "total_cash": 80.0}
    assert result["financial_currency"] == "EGP"
    assert any(
        "Financials-sector" in note
        for note in result["data_quality"].split("; ")
    )
    assert result["ratio_units"]["return_on_equity"] == "fraction"
    assert result["ratio_units"]["dividend_yield_ttm"] == "percent"


def test_stale_status_uses_returned_bars_not_catalog_snapshot(
    monkeypatch,
    fake_provider,
) -> None:
    real_timestamp = pd.Timestamp

    class FrozenTimestamp(real_timestamp):
        @classmethod
        def now(cls, tz=None):
            return real_timestamp("2026-12-21", tz=tz)

    monkeypatch.setattr(server_module.pd, "Timestamp", FrozenTimestamp)

    def fresh_history(entry, period_days):
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-12-20", "2026-12-21"]),
                "open": [10.0, 11.0],
                "high": [11.0, 12.0],
                "low": [9.0, 10.0],
                "close": [10.0, 11.0],
                "adj_close": [9.5, 10.5],
                "volume": [100.0, 110.0],
            }
        )

    monkeypatch.setattr(fake_provider, "get_history", fresh_history)

    result = asyncio.run(get_stock_summary("COMI"))

    assert result["as_of"] == "2026-12-21"
    assert result["stale"] is False


def test_get_dividends_reports_rows_and_dates(fake_provider) -> None:
    result = asyncio.run(get_dividends("COMI", years=5))

    assert result["observations"] == 1
    assert result["rows"][0]["amount"] == 1.0
    assert result["source"] == stock_provider.SOURCE


def test_get_dividends_clamps_years_to_fifty(fake_provider) -> None:
    result = asyncio.run(get_dividends("COMI", years=10**9))
    min_result = asyncio.run(get_dividends("COMI", years=0))

    assert result["years_applied"] == 50
    assert min_result["years_applied"] == 1
    assert fake_provider.dividend_calls == ["COMI", "COMI"]


def test_unexpected_stock_errors_are_logged_mapped_and_raised_when_configured(
    monkeypatch,
    fake_provider,
    caplog,
) -> None:
    def fail(entry, period_days):
        raise RuntimeError("private provider failure")

    monkeypatch.setattr(fake_provider, "get_history", fail)

    result = asyncio.run(get_stock_summary("COMI"))
    assert result["error"]["code"] == "DATA_SOURCE_ERROR"
    assert "private provider failure" not in result["error"]["message"]
    assert "private provider failure" not in caplog.text
    assert "exception_class=RuntimeError" in caplog.text
    assert "location=test_stock_tools.py:" in caplog.text

    monkeypatch.setenv("EGX_RAISE_TOOL_ERRORS", "true")
    with pytest.raises(ToolError, match="unexpected stock data error"):
        asyncio.run(get_stock_summary("COMI"))


def test_compare_stocks_deduplicates_ranks_and_keeps_partial_failure(fake_provider) -> None:
    fake_provider.fail_symbols.add("TMGH")

    result = asyncio.run(compare_stocks(["COMI", "comi", "TMGH", "SWDY"]))

    assert [row["symbol"] for row in result["comparisons"]] == [
        "COMI",
        "SWDY",
        "TMGH",
    ]
    assert result["comparisons"][2]["error"]["code"] == "EMPTY_DATA"
    assert result["comparisons"][0]["total_return_pct"] is not None


def test_compare_all_failures_obey_raise_mode(monkeypatch, fake_provider) -> None:
    monkeypatch.setenv("EGX_RAISE_TOOL_ERRORS", "true")
    fake_provider.fail_symbols.update({"COMI", "TMGH"})

    with pytest.raises(ToolError, match="No stock data"):
        asyncio.run(compare_stocks(["COMI", "TMGH"]))


def test_stock_vs_index_and_snapshot_combine_data(fake_provider, monkeypatch) -> None:
    monkeypatch.setattr(
        server_module.egx_client,
        "fetch_index_data",
        lambda index, period: pd.DataFrame(
            {
                "datetime": pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=400),
                "value": range(100, 500),
            }
        ),
    )

    relative = asyncio.run(stock_vs_index("COMI", "EGX30", 365))
    fake_provider.history_calls.clear()
    fake_provider.dividend_calls.clear()
    snapshot = asyncio.run(analysis_snapshot("COMI"))

    assert relative["index"] == "EGX30"
    assert relative["relative_performance"]["observations"] >= 30
    assert relative["stock_price_return_pct"] is not None
    assert relative["index_price_return_pct"] is not None
    assert relative["stock_total_return_pct"] is not None
    assert relative["excess_return_pct"] == pytest.approx(
        relative["stock_price_return_pct"] - relative["index_price_return_pct"]
    )
    assert "Close (split-adjusted)" in relative["price_basis"]
    assert relative["index_total_return_pct"] is None
    assert relative["index_total_return_note"]
    assert snapshot["symbol"] == "COMI"
    assert "summary" in snapshot
    assert "key_levels" in snapshot
    assert "dividends" in snapshot
    assert fake_provider.history_calls == ["COMI"]
    assert fake_provider.dividend_calls == ["COMI"]


def test_stock_errors_return_typed_payload_and_raise_mode(
    monkeypatch,
    fake_provider,
) -> None:
    fake_provider.fail_symbols.add("COMI")
    result = asyncio.run(get_stock_summary("COMI"))
    assert result["error"]["code"] == "EMPTY_DATA"
    fundamentals_error = asyncio.run(get_stock_fundamentals("QNBE"))
    assert fundamentals_error["error"]["code"] == "SYMBOL_UNAVAILABLE"

    monkeypatch.setenv("EGX_RAISE_TOOL_ERRORS", "true")
    with pytest.raises(ToolError, match="No stock data"):
        asyncio.run(get_stock_summary("COMI"))
    with pytest.raises(ToolError, match="no data is available"):
        asyncio.run(get_stock_fundamentals("QNBE"))


def test_ambiguous_and_known_unavailable_symbols_are_not_not_found(monkeypatch) -> None:
    ambiguous = asyncio.run(get_stock_summary("Egypt"))
    assert ambiguous["error"]["code"] == "AMBIGUOUS_SYMBOL"
    assert ambiguous["candidates"]
    unavailable = asyncio.run(get_stock_summary("QNBE"))
    assert unavailable["error"]["code"] == "SYMBOL_UNAVAILABLE"
    assert unavailable["error"]["message"] == (
        "Known EGX company, but no data is available from the Yahoo source"
    )

    monkeypatch.setenv("EGX_RAISE_TOOL_ERRORS", "true")
    with pytest.raises(ToolError, match="no data is available"):
        asyncio.run(get_stock_summary("QNBE"))


def test_list_stocks_can_include_known_unavailable_symbols() -> None:
    default = asyncio.run(list_stocks(limit=1000))
    all_symbols = asyncio.run(
        list_stocks(limit=1000, include_unavailable=True)
    )
    assert all(entry["symbol"] != "QNBE" for entry in default["stocks"])
    qnbe = next(entry for entry in all_symbols["stocks"] if entry["symbol"] == "QNBE")
    assert qnbe["unavailable"] is True


def test_all_stock_tools_are_async() -> None:
    for tool in (
        get_stock_history,
        get_stock_summary,
        get_stock_fundamentals,
        get_dividends,
        compare_stocks,
        stock_vs_index,
        analysis_snapshot,
        find_stock,
    ):
        assert asyncio.iscoroutinefunction(tool)


def test_stock_tools_write_nothing_to_stdout(fake_provider, capsys) -> None:
    async def call_tools() -> None:
        await list_stocks(limit=2)
        await get_stock_history("COMI", period_days=30)
        await get_stock_summary("COMI")
        await get_stock_fundamentals("COMI")
        await get_dividends("COMI")
        await compare_stocks(["COMI", "SWDY"])
        await stock_vs_index("COMI")
        await analysis_snapshot("COMI")
        await find_stock("COMI")

    asyncio.run(call_tools())

    assert capsys.readouterr().out == ""


def test_stock_tools_are_registered_with_fastmcp() -> None:
    server = server_module.create_mcp_server("127.0.0.1", 8000)
    tools = asyncio.run(server.list_tools())

    assert {
        "list_stocks",
        "find_stock",
        "get_stock_history",
        "get_stock_summary",
        "get_stock_fundamentals",
        "get_dividends",
        "compare_stocks",
        "stock_vs_index",
        "analysis_snapshot",
        "get_egypt_macro_data",
    }.issubset({tool.name for tool in tools})


def test_egypt_macro_tool_returns_separate_macro_sources(monkeypatch, tmp_path) -> None:
    from egx_mcp import macro_provider

    monkeypatch.setenv("FRED_API_KEY", "test-api-key")
    monkeypatch.setattr(
        macro_provider,
        "cache_database_path",
        lambda: tmp_path / "stock_cache.sqlite3",
    )
    calls: list[dict] = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, params=None, timeout=15):
        calls.append({"url": url, "params": params, "timeout": timeout})
        if url == macro_provider.FRED_OBSERVATIONS_URL:
            return Response(
                {"observations": [{"date": "2025-01-01", "value": "14.07"}]}
            )
        return Response(
            {
                "result": "success",
                "time_last_update_utc": "Sat, 24 Oct 2026 00:00:01 +0000",
                "rates": {"EGP": 47.5},
            }
        )

    monkeypatch.setattr(macro_provider.requests, "get", fake_get)
    payload = asyncio.run(get_egypt_macro_data())

    assert len(calls) == 2
    fred_call = next(
        call for call in calls if call["url"] == macro_provider.FRED_OBSERVATIONS_URL
    )
    assert fred_call["params"]["series_id"] == "FPCPITOTLZGEGY"
    assert fred_call["params"]["api_key"] == "test-api-key"
    assert fred_call["params"]["limit"] == 10
    assert fred_call["params"]["sort_order"] == "desc"
    assert set(payload["source"]) == {"FRED", "user_configured", "ExchangeRate-API"}
    assert isinstance(payload, dict)
    assert payload["metrics"]["inflation"]["value"] == 14.07
    assert payload["metrics"]["policy_rate"]["error"]["code"] == "NOT_CONFIGURED"
    assert payload["metrics"]["egp_usd_exchange_rate"]["value"] == 47.5


def test_egypt_macro_tool_missing_key_only_marks_inflation(monkeypatch, tmp_path) -> None:
    from egx_mcp import macro_provider

    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(
        macro_provider,
        "cache_database_path",
        lambda: tmp_path / "stock_cache.sqlite3",
    )
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda *args, **kwargs: type(
            "Response",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {
                    "result": "success",
                    "time_last_update_utc": "Sat, 24 Oct 2026 00:00:01 +0000",
                    "rates": {"EGP": 47.5},
                },
            },
        )(),
    )

    payload = asyncio.run(get_egypt_macro_data())

    assert payload["metrics"]["inflation"]["error"]["code"] == "MISSING_API_KEY"
    assert "FRED_API_KEY" in payload["metrics"]["inflation"]["error"]["message"]
    assert payload["metrics"]["egp_usd_exchange_rate"]["value"] == 47.5
    assert "error" not in payload


def test_egypt_macro_tool_maps_source_errors_and_raise_mode(
    monkeypatch,
    caplog,
    tmp_path,
) -> None:
    from egx_mcp import macro_provider

    monkeypatch.setenv("FRED_API_KEY", "test-api-key")
    monkeypatch.setattr(
        macro_provider,
        "cache_database_path",
        lambda: tmp_path / "stock_cache.sqlite3",
    )
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private response")),
    )

    result = asyncio.run(get_egypt_macro_data())
    assert result["error"]["code"] == "DATA_SOURCE_ERROR"
    assert "private response" not in result["error"]["message"]
    assert "private response" not in caplog.text

    monkeypatch.setenv("EGX_RAISE_TOOL_ERRORS", "true")
    with pytest.raises(ToolError, match="No usable values are available"):
        asyncio.run(get_egypt_macro_data())


def test_research_prompts_include_workflow_and_required_caveats() -> None:
    company_prompt = research_company("COMI")
    trade_prompt = build_trade_plan("COMI", "swing", "moderate")

    for tool_name in (
        "analysis_snapshot",
        "get_stock_fundamentals",
        "get_egypt_macro_data",
        "stock_vs_index",
    ):
        assert tool_name in company_prompt
    for phrase in (
        "not financial advice",
        "delayed",
        "split-adjusted",
        "daily price limits",
        "thin liquidity",
        "stale or missing data",
        "trend",
        "risk",
        "valuation caveats",
        "state that macro context is unavailable or stale",
        "user-configured policy rate",
        "calendar-year CPI",
        "Exchange Rate API",
    ):
        assert phrase.casefold() in company_prompt.casefold()
    for phrase in (
        "scenarios",
        "levels",
        "position-sizing",
        "invalidation",
        "not financial advice",
        "delayed",
        "split-adjusted",
        "daily price limits",
        "thin liquidity",
        "state that macro context is unavailable or stale",
        "user-configured",
        "Exchange Rate API",
        "stale or missing data",
    ):
        assert phrase.casefold() in trade_prompt.casefold()
    assert "COMI" in company_prompt
    assert "COMI" in trade_prompt
    assert "swing" in trade_prompt
    assert "moderate" in trade_prompt


def test_research_prompts_are_registered_with_fastmcp() -> None:
    server = server_module.create_mcp_server("127.0.0.1", 8000)
    prompts = asyncio.run(server.list_prompts())

    assert {"research_company", "build_trade_plan"}.issubset(
        {prompt.name for prompt in prompts}
    )


def test_phase_documentation_has_required_stdio() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert (
        'claude mcp add --transport stdio egx -- uv --directory "$PWD" '
        "run egx-mcp-server"
    ) in readme
    assert "EGX_INDEX_LIST_TTL_SECONDS" not in readme
