import asyncio
import inspect

import pandas as pd
import pytest
from starlette.testclient import TestClient
from mcp.server.fastmcp.exceptions import ToolError

from egx_mcp import egx_client
from egx_mcp.server import (
    compare_indices,
    create_mcp_server,
    get_index_history,
    get_index_summary,
    list_indices,
)
from egx_mcp.http_app import create_app


def make_frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=len(values), freq="D"),
            "value": values,
        }
    )


def test_list_indices_returns_codes_and_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        egx_client,
        "supported_indices",
        lambda: ["EGX30", "EGX70_EWI"],
    )

    result = list_indices()

    assert result["index"] is None
    assert result["source"] == "EGX via egx package"
    assert [entry["code"] for entry in result["indices"]] == [
        "EGX30",
        "EGX70_EWI",
    ]
    assert result["indices"][0]["description"]


def test_history_returns_iso_dated_value_rows_and_clamps_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    periods: list[int] = []
    monkeypatch.setattr(egx_client, "resolve_index", lambda index: "EGX30")

    def fake_fetch(index: str, period_days: int) -> pd.DataFrame:
        periods.append(period_days)
        return pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "value": [100.0, float("nan")],
            }
        )

    monkeypatch.setattr(egx_client, "fetch_index_data", fake_fetch)

    result = asyncio.run(get_index_history("egx30", 0))

    assert periods == [1]
    assert result["index"] == "EGX30"
    assert result["as_of"] == "2026-01-01"
    assert result["rows"] == [{"date": "2026-01-01", "value": 100.0}]
    assert result["period_days_applied"] == 1
    assert result["observations"] == 1


def test_summary_and_compare_include_required_common_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {"EGX30": [100.0, 110.0], "EGX70_EWI": [100.0, 105.0]}
    monkeypatch.setattr(
        egx_client,
        "resolve_index",
        lambda index: index.upper() if index.upper() == "EGX30" else "EGX70_EWI",
    )
    monkeypatch.setattr(
        egx_client,
        "fetch_index_data",
        lambda index, period_days: make_frame(values[index]),
    )

    summary = asyncio.run(get_index_summary("egx30"))
    comparison = asyncio.run(compare_indices(["egx30", "EGX70_EWI"]))

    assert summary["index"] == "EGX30"
    assert summary["latest_value"] == 110.0
    assert summary["source"] == "EGX via egx package"
    assert summary["period_days_applied"] == 90
    assert summary["observations"] == 2
    assert comparison["index"] == ["EGX30", "EGX70_EWI"]
    assert comparison["best_performer"] == "EGX30"
    assert comparison["source"] == "EGX via egx package"
    assert all(row["as_of"] == "2026-01-02" for row in comparison["comparisons"])
    assert all(row["period_days_applied"] == 90 for row in comparison["comparisons"])
    assert all(row["observations"] == 2 for row in comparison["comparisons"])


def test_bad_index_and_empty_comparison_return_structured_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_index(index: str) -> str:
        raise egx_client.InvalidIndexError("Unsupported index.")

    monkeypatch.setattr(egx_client, "resolve_index", reject_index)
    invalid = asyncio.run(get_index_history("BAD"))
    empty = asyncio.run(compare_indices([]))

    assert invalid["error"]["code"] == "INVALID_INDEX"
    assert invalid["as_of"] is None
    assert empty["error"]["code"] == "INVALID_ARGUMENT"
    assert empty["source"] == "EGX via egx package"


def test_network_failure_is_returned_as_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(egx_client, "resolve_index", lambda index: "EGX30")

    def fail_fetch(index: str, period_days: int) -> pd.DataFrame:
        raise egx_client.EGXNetworkError("Unable to fetch EGX30 data.")

    monkeypatch.setattr(egx_client, "fetch_index_data", fail_fetch)

    result = asyncio.run(get_index_summary("EGX30"))

    assert result["error"]["code"] == "NETWORK_ERROR"
    assert "stack" not in result["error"]["message"].lower()


def test_unexpected_source_failure_maps_to_safe_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(egx_client, "resolve_index", lambda index: "EGX30")
    monkeypatch.setattr(egx_client, "_DATA_CACHE", egx_client.TTLCache(ttl_seconds=0))

    def fail(index: str, period: int) -> pd.DataFrame:
        raise RuntimeError("internal credentials should not escape")

    monkeypatch.setattr(egx_client.egx, "get_index_data", fail)
    result = asyncio.run(get_index_summary("EGX30"))

    assert result["error"]["code"] == "DATA_SOURCE_ERROR"
    assert "internal credentials" not in result["error"]["message"]


def test_http_health_is_public_and_mcp_requires_the_configured_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "testserver")
    app = create_app(
        {"": create_mcp_server("127.0.0.1", 8000)},
        auth_token="test-token",
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        health = client.get("/health")
        unauthorized = client.get("/mcp")
        malformed = client.get("/mcp", headers={"Authorization": b"Bearer \xe9"})
        authorized = client.get(
            "/mcp",
            headers={"Authorization": "Bearer test-token"},
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert unauthorized.status_code == 401
    assert malformed.status_code == 401
    assert authorized.status_code == 406


def test_data_tools_are_async() -> None:
    assert inspect.iscoroutinefunction(get_index_history)
    assert inspect.iscoroutinefunction(get_index_summary)
    assert inspect.iscoroutinefunction(compare_indices)


def test_history_frequency_limit_and_applied_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    monkeypatch.setattr(egx_client, "resolve_index", lambda index: "EGX30")
    monkeypatch.setattr(
        egx_client,
        "fetch_index_data",
        lambda index, period_days: pd.DataFrame(
            {"datetime": dates, "value": range(100, 140)}
        ),
    )

    weekly = asyncio.run(
        get_index_history("EGX30", period_days=5000, frequency="weekly")
    )
    limited = asyncio.run(
        get_index_history("EGX30", period_days=30, max_rows=2)
    )
    monthly = asyncio.run(
        get_index_history("EGX30", period_days=40, frequency="monthly")
    )

    assert weekly["period_days_applied"] == 1825
    assert weekly["observations"] == len(weekly["rows"])
    assert len(weekly["rows"]) < 40
    assert limited["period_days_applied"] == 30
    assert limited["observations"] == 2
    assert limited["rows"][-1]["value"] == 139
    assert monthly["observations"] == 2


def test_compare_deduplicates_and_returns_partial_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "EGX30": [100.0, 120.0],
        "EGX70_EWI": [100.0, 105.0],
    }
    calls: list[str] = []
    monkeypatch.setattr(
        egx_client,
        "supported_indices",
        lambda: ["EGX30", "EGX70_EWI", "EGX30_TR"],
    )
    monkeypatch.setattr(
        egx_client,
        "resolve_index",
        lambda index: {
            "egx30": "EGX30",
            "egx70_ewi": "EGX70_EWI",
            "egx30_tr": "EGX30_TR",
        }[index.casefold()],
    )

    def fetch(index: str, period_days: int) -> pd.DataFrame:
        calls.append(index)
        if index == "EGX30_TR":
            raise egx_client.EGXNetworkError("EGX unavailable")
        return make_frame(values[index])

    monkeypatch.setattr(egx_client, "fetch_index_data", fetch)
    result = asyncio.run(
        compare_indices(["egx30", "EGX30", "EGX70_EWI", "EGX30_TR"])
    )

    assert sorted(calls) == ["EGX30", "EGX30_TR", "EGX70_EWI"]
    assert result["index"] == ["EGX30", "EGX70_EWI", "EGX30_TR"]
    assert result["best_performer"] == "EGX30"
    assert len(result["comparisons"]) == 3
    failed = result["comparisons"][2]
    assert failed["error"]["code"] == "NETWORK_ERROR"
    assert result["comparisons"][0]["observations"] == 2
    assert result["comparisons"][0]["drawdown_from_period_high_pct"] == 0.0


def test_compare_rejects_too_many_indices_and_non_list_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(egx_client, "supported_indices", lambda: ["EGX30", "EGX70_EWI"])

    result = asyncio.run(compare_indices(["EGX30", "EGX70_EWI", "EGX30_TR"]))
    assert result["error"]["code"] == "INVALID_ARGUMENT"

    empty = asyncio.run(compare_indices([]))
    assert empty["error"]["code"] == "INVALID_ARGUMENT"


def test_tool_errors_can_be_raised_for_mcp_error_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EGX_RAISE_TOOL_ERRORS", "true")
    monkeypatch.setattr(
        egx_client,
        "resolve_index",
        lambda index: (_ for _ in ()).throw(
            egx_client.InvalidIndexError("Unsupported index.")
        ),
    )

    with pytest.raises(ToolError, match="Unsupported index"):
        asyncio.run(get_index_summary("bad"))
