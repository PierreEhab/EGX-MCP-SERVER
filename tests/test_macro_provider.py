from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3

import pytest

from egx_mcp import macro_provider


@pytest.fixture
def macro_cache(monkeypatch, tmp_path):
    database = tmp_path / "fred-cache.sqlite3"
    monkeypatch.setattr(macro_provider, "cache_database_path", lambda: database)
    monkeypatch.setattr(
        macro_provider,
        "_now",
        lambda: datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc),
    )
    return database


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _fred_response(params):
    return FakeResponse(
        {
            "observations": [
                {"date": "2025-01-01", "value": "14.07"},
            ]
        }
    )


def _exchange_response(_url, params, timeout):
    assert params is None
    assert timeout == 15
    return FakeResponse(
        {
            "result": "success",
            "time_last_update_utc": "Sat, 24 Oct 2026 00:00:01 +0000",
            "rates": {"EGP": 47.5},
        }
    )


def _mock_valid_sources(monkeypatch):
    def get(url, params=None, timeout=15):
        if url == macro_provider.FRED_OBSERVATIONS_URL:
            return _fred_response(params)
        return _exchange_response(url, params, timeout)

    monkeypatch.setattr(macro_provider.requests, "get", get)


def _configure_policy(monkeypatch):
    monkeypatch.setenv("EGYPT_POLICY_RATE_PCT", "19.25")
    monkeypatch.setenv("EGYPT_POLICY_RATE_DATE", "2026-10-01")
    monkeypatch.delenv("EGYPT_POLICY_RATE_NAME", raising=False)


def test_macro_returns_correct_sources_metrics_and_stale_rules(
    monkeypatch,
    macro_cache,
):
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")
    _configure_policy(monkeypatch)
    _mock_valid_sources(monkeypatch)

    result = macro_provider.get_latest_egypt_macro_data()

    assert set(result["metrics"]) == {
        "inflation",
        "policy_rate",
        "egp_usd_exchange_rate",
    }
    inflation = result["metrics"]["inflation"]
    assert inflation["series_id"] == "FPCPITOTLZGEGY"
    assert inflation["value"] == 14.07
    assert inflation["date"] == "2025-01-01"
    assert inflation["stale"] is False
    assert "calendar-year average" in inflation["note"].casefold()
    assert result["metrics"]["policy_rate"]["source"] == "user_configured"
    assert result["metrics"]["policy_rate"]["value"] == 19.25
    assert result["metrics"]["policy_rate"]["name"] == (
        "Central Bank of Egypt policy rate"
    )
    exchange = result["metrics"]["egp_usd_exchange_rate"]
    assert exchange["source"] == "ExchangeRate-API"
    assert exchange["value"] == 47.5
    assert exchange["date"] == "2026-10-24"
    assert "Rates By Exchange Rate API (https://www.exchangerate-api.com)" in exchange[
        "attribution"
    ]
    assert set(result["source"]) == {
        "FRED",
        "user_configured",
        "ExchangeRate-API",
    }


def test_inflation_stale_only_when_observation_year_is_more_than_one_behind(
    monkeypatch,
    macro_cache,
):
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")
    _configure_policy(monkeypatch)

    class HistoricalFredResponse(FakeResponse):
        def json(self):
            return {
                "observations": [{"date": "2024-01-01", "value": "14.07"}]
            }

    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda url, params=None, timeout=15: (
            HistoricalFredResponse({})
            if url == macro_provider.FRED_OBSERVATIONS_URL
            else _exchange_response(url, params, timeout)
        ),
    )

    result = macro_provider.get_latest_egypt_macro_data()
    assert result["metrics"]["inflation"]["stale"] is True


def test_exchange_rate_is_stale_after_three_days(monkeypatch, macro_cache):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    _configure_policy(monkeypatch)

    def get(url, params=None, timeout=15):
        if url == macro_provider.FRED_OBSERVATIONS_URL:
            raise AssertionError("FRED must not be called without a key")
        return FakeResponse(
            {
                "result": "success",
                "time_last_update_utc": "Wed, 21 Oct 2026 00:00:01 +0000",
                "rates": {"EGP": 47.5},
            }
        )

    monkeypatch.setattr(macro_provider.requests, "get", get)

    result = macro_provider.get_latest_egypt_macro_data()

    assert result["metrics"]["egp_usd_exchange_rate"]["stale"] is True


def test_missing_fred_key_only_fails_inflation(monkeypatch, macro_cache):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    _configure_policy(monkeypatch)
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda url, params=None, timeout=15: _exchange_response(
            url,
            params,
            timeout,
        ),
    )

    result = macro_provider.get_latest_egypt_macro_data()

    error = result["metrics"]["inflation"]["error"]
    assert error["code"] == "MISSING_API_KEY"
    assert "FRED_API_KEY" in error["message"]
    assert result["metrics"]["policy_rate"]["value"] == 19.25
    assert result["metrics"]["egp_usd_exchange_rate"]["value"] == 47.5
    assert "error" not in result


@pytest.mark.parametrize(
    ("rate", "observed_on", "expected_code"),
    [
        (None, None, "NOT_CONFIGURED"),
        ("101", "2026-10-01", "INVALID_CONFIGURATION"),
        ("abc", "2026-10-01", "INVALID_CONFIGURATION"),
        ("19", "2026-10-26", "INVALID_CONFIGURATION"),
        ("19", "not-a-date", "INVALID_CONFIGURATION"),
    ],
)
def test_policy_rate_configuration_validation(
    monkeypatch,
    macro_cache,
    rate,
    observed_on,
    expected_code,
):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("EGYPT_POLICY_RATE_PCT", raising=False)
    monkeypatch.delenv("EGYPT_POLICY_RATE_DATE", raising=False)
    if rate is not None:
        monkeypatch.setenv("EGYPT_POLICY_RATE_PCT", rate)
    if observed_on is not None:
        monkeypatch.setenv("EGYPT_POLICY_RATE_DATE", observed_on)
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no call expected")),
    )

    result = macro_provider.get_latest_egypt_macro_data()

    error = result["metrics"]["policy_rate"]["error"]
    assert error["code"] == expected_code
    if expected_code == "NOT_CONFIGURED":
        for variable in (
            "EGYPT_POLICY_RATE_PCT",
            "EGYPT_POLICY_RATE_DATE",
        ):
            assert variable in error["message"]


def test_policy_rate_stale_after_sixty_days(monkeypatch, macro_cache):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("EGYPT_POLICY_RATE_PCT", "19")
    monkeypatch.setenv("EGYPT_POLICY_RATE_DATE", "2026-08-20")
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no call expected")),
    )

    result = macro_provider.get_latest_egypt_macro_data()
    assert result["metrics"]["policy_rate"]["stale"] is True


def test_exchange_rate_failure_does_not_log_url_and_has_safe_metric_error(
    monkeypatch,
    macro_cache,
    caplog,
):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    _configure_policy(monkeypatch)

    def fail(url, params=None, timeout=15):
        if url != macro_provider.FRED_OBSERVATIONS_URL:
            raise macro_provider.requests.ConnectionError(
                "https://open.er-api.com/v6/latest/USD?api_key=SECRET123"
            )
        raise AssertionError("FRED must not be called without a key")

    monkeypatch.setattr(macro_provider.requests, "get", fail)
    with caplog.at_level(logging.DEBUG):
        result = macro_provider.get_latest_egypt_macro_data()

    assert "SECRET123" not in caplog.text
    assert "open.er-api.com" not in caplog.text
    assert result["metrics"]["egp_usd_exchange_rate"]["error"]["code"] == (
        "DATA_SOURCE_ERROR"
    )
    assert "SECRET123" not in str(result)


def test_exchange_rate_requires_success_and_positive_finite_rate(
    monkeypatch,
    macro_cache,
):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    _configure_policy(monkeypatch)
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda url, params=None, timeout=15: FakeResponse(
            {"result": "error", "rates": {"EGP": 47}}
        ),
    )

    result = macro_provider.get_latest_egypt_macro_data()
    assert result["metrics"]["egp_usd_exchange_rate"]["value"] is None
    assert result["metrics"]["egp_usd_exchange_rate"]["error"]


def test_macro_result_is_cached_for_twelve_hours(monkeypatch, macro_cache):
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")
    _configure_policy(monkeypatch)
    calls = []

    def get(url, params=None, timeout=15):
        calls.append(url)
        return (
            _fred_response(params)
            if url == macro_provider.FRED_OBSERVATIONS_URL
            else _exchange_response(url, params, timeout)
        )

    monkeypatch.setattr(macro_provider.requests, "get", get)

    first = macro_provider.get_latest_egypt_macro_data()
    second = macro_provider.get_latest_egypt_macro_data()

    assert len(calls) == 2
    assert first["metrics"] == second["metrics"]
    with sqlite3.connect(macro_cache) as connection:
        cached_metrics = {
            row[0]
            for row in connection.execute(
                "SELECT metric FROM egypt_macro_cache_v2"
            )
        }
    assert cached_metrics == {"inflation", "egp_usd_exchange_rate"}


def test_v2_cache_payload_without_source_is_ignored_and_refetched(
    monkeypatch,
    macro_cache,
):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    _configure_policy(monkeypatch)
    with sqlite3.connect(macro_cache) as connection:
        connection.execute(
            """
            CREATE TABLE egypt_macro_cache_v2 (
                metric TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO egypt_macro_cache_v2 (metric, payload, fetched_at)
            VALUES (?, ?, ?)
            """,
            (
                "egp_usd_exchange_rate",
                '{"value": 1, "date": "2026-10-24", "name": "old", '
                '"unit": "EGP per USD", "frequency": "daily"}',
                datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc).isoformat(),
            ),
        )
    calls = []

    def get(url, params=None, timeout=15):
        calls.append(url)
        return _exchange_response(url, params, timeout)

    monkeypatch.setattr(macro_provider.requests, "get", get)

    result = macro_provider.get_latest_egypt_macro_data()

    assert calls == [macro_provider.EXCHANGE_RATE_URL]
    assert result["metrics"]["egp_usd_exchange_rate"]["value"] == 47.5


def test_unknown_fresh_cache_metrics_do_not_add_cache_note(monkeypatch, macro_cache):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    _configure_policy(monkeypatch)
    with sqlite3.connect(macro_cache) as connection:
        connection.execute(
            """
            CREATE TABLE egypt_macro_cache_v2 (
                metric TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO egypt_macro_cache_v2 (metric, payload, fetched_at)
            VALUES (?, ?, ?)
            """,
            (
                "discount_rate",
                '{"value": 1, "date": "2026-10-24", "source": "legacy", '
                '"name": "legacy", "unit": "percent", "frequency": "daily"}',
                datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc).isoformat(),
            ),
        )
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            macro_provider.requests.ConnectionError("offline")
        ),
    )

    result = macro_provider.get_latest_egypt_macro_data()

    assert "cache_note" not in result


def test_missing_source_on_one_metric_does_not_crash_source_list(
    monkeypatch,
    macro_cache,
):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    _configure_policy(monkeypatch)
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda url, params=None, timeout=15: _exchange_response(
            url, params, timeout
        ),
    )
    policy_rate = macro_provider._policy_rate

    def missing_source(today):
        result = policy_rate(today)
        result.pop("source")
        return result

    monkeypatch.setattr(macro_provider, "_policy_rate", missing_source)

    result = macro_provider.get_latest_egypt_macro_data()

    assert "user_configured" not in result["source"]
    assert None not in result["source"]


def test_cache_fallback_preserves_sources_and_reports_failed_refresh(
    monkeypatch,
    macro_cache,
):
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")
    _configure_policy(monkeypatch)
    _mock_valid_sources(monkeypatch)
    cached = macro_provider.get_latest_egypt_macro_data()

    monkeypatch.setattr(
        macro_provider,
        "_now",
        lambda: datetime(2026, 10, 26, 1, 0, tzinfo=timezone.utc),
    )

    def fail(url, params=None, timeout=15):
        if url == macro_provider.FRED_OBSERVATIONS_URL:
            raise macro_provider.requests.Timeout("offline")
        return _exchange_response(url, params, timeout)

    monkeypatch.setattr(macro_provider.requests, "get", fail)
    result = macro_provider.get_latest_egypt_macro_data()

    assert result["metrics"]["inflation"]["value"] == cached["metrics"]["inflation"]["value"]
    assert "cached" in result["metrics"]["inflation"]["error"]["message"].casefold()


def test_no_invalid_series_ids_remain_in_provider_or_tool_source():
    provider_text = Path(macro_provider.__file__).read_text(encoding="utf-8")
    from egx_mcp import server

    server_text = Path(server.__file__).read_text(encoding="utf-8")
    invalid_ids = ("INTDSRTRM" + "193N", "DEXE" + "GUS")
    assert all(identifier not in provider_text + server_text for identifier in invalid_ids)
