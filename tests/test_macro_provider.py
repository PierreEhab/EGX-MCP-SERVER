from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
import threading

import pytest

from egx_mcp import macro_provider


class FakeResponse:
    def __init__(self, series_id: str, value: str, observed_on: str, status: int = 200):
        self.series_id = series_id
        self.value = value
        self.observed_on = observed_on
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            error = macro_provider.requests.HTTPError(
                f"Request failed: https://fred.example/?api_key=SECRET123"
            )
            error.response = self
            raise error

    def json(self):
        return {
            "observations": [
                {"date": self.observed_on, "value": self.value}
            ]
        }


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


def _response_for(params):
    sid = params["series_id"]
    value = {
        "FPCPITOTLZGEGY": "12.3",
        "INTDSRTRM193N": "18.0",
        "DEXEGUS": "48.0",
    }[sid]
    return FakeResponse(sid, value, "2026-10-20")


def _response_object(observations):
    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"observations": observations}

    return Response()


def test_macro_requests_run_concurrently_and_return_frequency_aware_freshness(
    monkeypatch,
    macro_cache,
):
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")
    barrier = threading.Barrier(3)
    active = 0
    active_lock = threading.Lock()
    max_active = 0
    observed_params = []

    def get(url, params, timeout):
        nonlocal active, max_active
        observed_params.append(params.copy())
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        barrier.wait(timeout=2)
        with active_lock:
            active -= 1
        return _response_for(params)

    monkeypatch.setattr(macro_provider.requests, "get", get)

    result = macro_provider.get_latest_egypt_macro_data()

    assert max_active == 3
    assert result["metrics"]["inflation"]["frequency"] == "annual"
    assert result["metrics"]["discount_rate"]["frequency"] == "monthly"
    assert result["metrics"]["egp_usd_exchange_rate"]["frequency"] == "daily"
    assert result["metrics"]["inflation"]["age_days"] == 5
    assert result["metrics"]["inflation"]["stale"] is False
    assert result["metrics"]["discount_rate"]["stale"] is False
    assert result["metrics"]["egp_usd_exchange_rate"]["stale"] is False
    assert result["data_quality"] == []
    assert all(metric["error"] is None for metric in result["metrics"].values())
    assert all(params["limit"] == 10 for params in observed_params)


def test_observation_parser_skips_missing_and_non_finite_values(
    monkeypatch,
    macro_cache,
):
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")

    def get(url, params, timeout):
        if params["series_id"] == "FPCPITOTLZGEGY":
            observations = [
                {"date": "2026-10-21", "value": "."},
                {"date": "2026-10-20", "value": "12.5"},
                {"date": "2026-10-19", "value": "11.0"},
            ]
        else:
            observations = [{"date": "2026-10-20", "value": "1"}]
        return _response_object(observations)

    monkeypatch.setattr(macro_provider.requests, "get", get)

    result = macro_provider.get_latest_egypt_macro_data()
    inflation = result["metrics"]["inflation"]

    assert inflation["value"] == 12.5
    assert inflation["date"] == "2026-10-20"
    assert inflation["error"] is None


def test_all_missing_observations_return_clear_metric_error(monkeypatch, macro_cache):
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda url, params, timeout: _response_object(
            [
                {"date": "2026-10-21", "value": "."},
                {"date": "2026-10-20", "value": "."},
            ]
        ),
    )

    result = macro_provider.get_latest_egypt_macro_data()

    assert result["error"]["code"] == "DATA_SOURCE_ERROR"
    assert all(metric["value"] is None for metric in result["metrics"].values())
    assert all(
        "no valid observation" in metric["error"].casefold()
        for metric in result["metrics"].values()
    )


def test_macro_cache_database_can_be_deleted_after_call(monkeypatch, macro_cache):
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda url, params, timeout: _response_for(params),
    )

    macro_provider.get_latest_egypt_macro_data()

    macro_cache.unlink()
    assert not macro_cache.exists()


def test_debug_request_log_does_not_expose_fred_api_key(
    monkeypatch,
    macro_cache,
    caplog,
):
    from egx_mcp import egx_client

    monkeypatch.setenv("EGX_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")

    def fail_request(url, params, timeout):
        logging.getLogger("urllib3.connectionpool").debug(
            "Request params include api_key=%s",
            params["api_key"],
        )
        raise macro_provider.requests.ConnectionError("network unavailable")

    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        fail_request,
    )

    with caplog.at_level(logging.DEBUG):
        egx_client.configure_logging()
        macro_provider.get_latest_egypt_macro_data()

    assert "SECRET123" not in caplog.text
    assert logging.getLogger("urllib3").level == logging.WARNING
    assert logging.getLogger("requests").level == logging.WARNING


def test_one_failing_series_preserves_successful_metrics(monkeypatch, macro_cache):
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")

    def get(url, params, timeout):
        if params["series_id"] == "INTDSRTRM193N":
            raise macro_provider.requests.ConnectionError("private endpoint detail")
        return _response_for(params)

    monkeypatch.setattr(macro_provider.requests, "get", get)

    result = macro_provider.get_latest_egypt_macro_data()

    assert "error" not in result
    assert result["metrics"]["inflation"]["value"] == 12.3
    assert result["metrics"]["egp_usd_exchange_rate"]["value"] == 48.0
    assert result["metrics"]["discount_rate"]["value"] is None
    assert result["metrics"]["discount_rate"]["error"]


def test_failed_requests_never_log_or_return_api_key(monkeypatch, macro_cache, caplog):
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")

    def fail_request(*args, **kwargs):
        error = macro_provider.requests.HTTPError(
            "https://fred.example/?api_key=SECRET123"
        )
        error.response = FakeResponse("", "", "", status=429)
        raise error

    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        fail_request,
    )

    with caplog.at_level(logging.WARNING, logger=macro_provider.__name__):
        result = macro_provider.get_latest_egypt_macro_data()

    assert "SECRET123" not in caplog.text
    assert "SECRET123" not in str(result)
    assert "exception_class=HTTPError http_status=429" in caplog.text
    assert "HTTP status 429" in result["metrics"]["inflation"]["error"]
    assert result["error"]["code"] == "DATA_SOURCE_ERROR"
    assert all(metric["error"] for metric in result["metrics"].values())


def test_stale_metrics_are_reported_and_listed_in_data_quality(
    monkeypatch,
    macro_cache,
):
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda url, params, timeout: _response_object(
            [
                {
                    "date": {
                        "FPCPITOTLZGEGY": "2025-01-01",
                        "INTDSRTRM193N": "2026-08-20",
                        "DEXEGUS": "2026-10-20",
                    }[params["series_id"]],
                    "value": "1.0",
                }
            ]
        ),
    )

    result = macro_provider.get_latest_egypt_macro_data()

    assert result["metrics"]["inflation"]["stale"] is True
    assert result["metrics"]["discount_rate"]["stale"] is False
    assert result["metrics"]["egp_usd_exchange_rate"]["stale"] is False
    assert "inflation" in result["data_quality"]


def test_all_failed_series_return_individual_errors_and_top_level_error(
    monkeypatch,
    macro_cache,
):
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()),
    )

    result = macro_provider.get_latest_egypt_macro_data()

    assert result["error"]["code"] == "DATA_SOURCE_ERROR"
    assert len(result["metrics"]) == 3
    assert all(
        set(("value", "date", "error")).issubset(metric)
        for metric in result["metrics"].values()
    )


def test_macro_result_is_cached_for_twelve_hours(monkeypatch, macro_cache):
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")
    calls = []
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda url, params, timeout: (calls.append(params["series_id"]) or _response_for(params)),
    )

    first = macro_provider.get_latest_egypt_macro_data()
    second = macro_provider.get_latest_egypt_macro_data()

    assert len(calls) == 3
    assert first["metrics"] == second["metrics"]
    assert "cache" in second["cache_note"].casefold()
    with sqlite3.connect(macro_cache) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM egypt_macro_cache"
        ).fetchone()[0] == 3


def test_failed_refresh_serves_cached_metrics_with_note(monkeypatch, macro_cache):
    monkeypatch.setenv("FRED_API_KEY", "SECRET123")
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda url, params, timeout: _response_for(params),
    )
    cached = macro_provider.get_latest_egypt_macro_data()

    monkeypatch.setattr(
        macro_provider,
        "_now",
        lambda: datetime(2026, 10, 26, 1, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        macro_provider.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()),
    )

    result = macro_provider.get_latest_egypt_macro_data()

    assert result["metrics"]["inflation"]["value"] == cached["metrics"]["inflation"]["value"]
    assert result["metrics"]["inflation"]["error"]
    assert "cached" in result["metrics"]["inflation"]["error"].casefold()
    assert "cached" in result["cache_note"].casefold()
    assert result["error"]["code"] == "DATA_SOURCE_ERROR"


def test_cache_database_path_honors_egx_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("EGX_CACHE_DIR", str(tmp_path))

    assert macro_provider.cache_database_path() == tmp_path / "stock_cache.sqlite3"
