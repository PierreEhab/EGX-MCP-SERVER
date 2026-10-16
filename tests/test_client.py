from concurrent.futures import ThreadPoolExecutor
import time

import pandas as pd
import pytest
import requests

from egx_mcp import egx_client
from egx_mcp.cache import TTLCache


def test_index_resolution_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        egx_client.egx,
        "get_supported_indices",
        lambda: {"EGX30", "EGX_33_Shariah"},
    )

    assert egx_client.resolve_index("egx30") == "EGX30"
    assert egx_client.resolve_index("  egx_33_shariah ") == "EGX_33_Shariah"


def test_invalid_index_raises_a_specific_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        egx_client.egx,
        "get_supported_indices",
        lambda: {"EGX30"},
    )

    with pytest.raises(egx_client.InvalidIndexError):
        egx_client.resolve_index("INVALID")


def test_fetch_clamps_period_and_caches_a_defensive_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        egx_client.egx,
        "get_supported_indices",
        lambda: {"EGX30"},
    )

    def fake_get_index_data(index: str, period: int) -> pd.DataFrame:
        calls.append((index, period))
        return pd.DataFrame(
            {"datetime": ["2026-01-01"], "value": [100.0]}
        )

    monkeypatch.setattr(egx_client.egx, "get_index_data", fake_get_index_data)
    monkeypatch.setattr(egx_client, "_DATA_CACHE", TTLCache(ttl_seconds=300))

    monkeypatch.setattr(
        egx_client,
        "resolve_index",
        lambda index: pytest.fail("fetch_index_data must receive a canonical index"),
    )
    first = egx_client.fetch_index_data("EGX30", 5000)
    first.loc[0, "value"] = -1
    second = egx_client.fetch_index_data("EGX30", 5000)

    assert calls == [("EGX30", 1825)]
    assert second.loc[0, "value"] == 100.0


def test_longest_window_cache_slices_shorter_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime

    today = pd.Timestamp(datetime.now().date())
    dates = pd.date_range(today - pd.Timedelta(days=30), today, freq="D")
    calls: list[int] = []
    monkeypatch.setattr(egx_client, "_DATA_CACHE", TTLCache(ttl_seconds=300))
    monkeypatch.setattr(
        egx_client.egx,
        "get_index_data",
        lambda index, period: (
            calls.append(period)
            or pd.DataFrame({"datetime": dates, "value": range(len(dates))})
        ),
    )

    long_result = egx_client.fetch_index_data("EGX30", 30)
    short_result = egx_client.fetch_index_data("EGX30", 7)

    assert calls == [30]
    assert len(short_result) <= 8
    assert short_result["datetime"].min() >= today - pd.Timedelta(days=7)
    assert len(long_result) > len(short_result)


def test_identical_concurrent_requests_are_coalesced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    monkeypatch.setattr(egx_client, "_DATA_CACHE", TTLCache(ttl_seconds=0))

    def slow_source(index: str, period: int) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        time.sleep(0.1)
        return pd.DataFrame(
            {"datetime": [pd.Timestamp.now().normalize()], "value": [100.0]}
        )

    monkeypatch.setattr(egx_client.egx, "get_index_data", slow_source)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda _: egx_client.fetch_index_data("EGX30", 30), range(8))
        )

    assert calls == 1
    assert all(result.loc[0, "value"] == 100 for result in results)


def test_unexpected_source_exception_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(egx_client, "_DATA_CACHE", TTLCache(ttl_seconds=0))

    def fail(index: str, period: int) -> pd.DataFrame:
        raise RuntimeError("internal credentials should not escape")

    monkeypatch.setattr(egx_client.egx, "get_index_data", fail)
    with pytest.raises(egx_client.EGXClientError) as error:
        egx_client.fetch_index_data("EGX30", 30)
    assert "internal credentials" not in str(error.value)


def test_empty_and_network_responses_raise_clear_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        egx_client.egx,
        "get_supported_indices",
        lambda: {"EGX30"},
    )
    monkeypatch.setattr(egx_client, "_DATA_CACHE", TTLCache(ttl_seconds=0))
    monkeypatch.setattr(
        egx_client.egx,
        "get_index_data",
        lambda index, period: pd.DataFrame(columns=["datetime", "value"]),
    )
    with pytest.raises(egx_client.EmptyDataError):
        egx_client.fetch_index_data("EGX30", 17)

    def raise_network_error(index: str, period: int) -> pd.DataFrame:
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(egx_client.egx, "get_index_data", raise_network_error)
    with pytest.raises(egx_client.EGXNetworkError, match="Unable to fetch EGX30"):
        egx_client.fetch_index_data("EGX30", 18)


def test_fetch_drops_invalid_rows_and_rejects_all_invalid_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    today = pd.Timestamp.now().normalize()
    monkeypatch.setattr(egx_client, "resolve_index", lambda index: "EGX30")
    monkeypatch.setattr(egx_client, "_DATA_CACHE", TTLCache(ttl_seconds=0))
    monkeypatch.setattr(
        egx_client.egx,
        "get_index_data",
        lambda index, period: pd.DataFrame(
            {
                "datetime": [
                    today,
                    "not-a-date",
                    today + pd.Timedelta(days=1),
                ],
                "value": [100.0, 200.0, float("inf")],
            }
        ),
    )

    result = egx_client.fetch_index_data("EGX30", 7)
    assert result.to_dict("records") == [
        {"datetime": today, "value": 100.0}
    ]

    monkeypatch.setattr(
        egx_client.egx,
        "get_index_data",
        lambda index, period: pd.DataFrame(
            {"datetime": ["not-a-date"], "value": [float("nan")]}
        ),
    )
    with pytest.raises(egx_client.EmptyDataError):
        egx_client.fetch_index_data("EGX30", 8)
