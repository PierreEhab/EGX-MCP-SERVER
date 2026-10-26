import pytest


@pytest.fixture(autouse=True)
def clear_server_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FRED_API_KEY",
        "EGYPT_POLICY_RATE_PCT",
        "EGYPT_POLICY_RATE_DATE",
        "EGYPT_POLICY_RATE_NAME",
        "EGX_CACHE_DIR",
        "EGX_LOG_LEVEL",
        "EGX_RAISE_TOOL_ERRORS",
        "EGX_YAHOO_MIN_INTERVAL_SECONDS",
        "EGX_CACHE_TTL_SECONDS",
        "MCP_TRANSPORT",
        "MCP_HOST",
        "MCP_PORT",
        "MCP_AUTH_TOKEN",
        "MCP_ALLOWED_HOSTS",
        "MCP_ALLOW_INSECURE",
        "LOCALAPPDATA",
    ):
        monkeypatch.delenv(name, raising=False)
