import asyncio
import logging

import pandas as pd
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from starlette.testclient import TestClient

from egx_mcp import egx_client
from egx_mcp.http_app import (
    BearerAuthMiddleware,
    _is_loopback,
    create_app,
    validate_http_security,
)
from egx_mcp.server import _transport_security, create_mcp_server


def test_auth_middleware_handles_non_ascii_header_and_health_is_exempt() -> None:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def health(_request):
        return JSONResponse({"status": "ok"})

    async def private(_request):
        return JSONResponse({"status": "ok"})

    app = Starlette(
        routes=[Route("/health", health), Route("/private", private)],
        middleware=[Middleware(BearerAuthMiddleware, token="secret")],
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        malformed = client.get("/private", headers={"Authorization": b"Bearer \xe9"})
        valid = client.get(
            "/private",
            headers={"Authorization": "Bearer secret"},
        )

    assert malformed.status_code == 401
    assert malformed.headers["www-authenticate"] == "Bearer"
    assert valid.status_code == 200


def test_auth_middleware_accepts_non_ascii_bearer_token_bytes() -> None:
    received: list[dict[str, object]] = []

    async def app(scope, receive, send):
        received.append({"status": 200})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        received.append(message)

    middleware = BearerAuthMiddleware(app, token="café")
    scope = {
        "type": "http",
        "path": "/private",
        "headers": [(b"authorization", b"bEaReR caf\xc3\xa9")],
    }

    asyncio.run(middleware(scope, receive, send))

    assert received == [{"status": 200}]


def test_localhost_subdomain_is_not_treated_as_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)
    monkeypatch.setenv("MCP_AUTH_TOKEN", "test-token")

    assert not _is_loopback("localhost.example.com")
    settings = _transport_security("localhost.example.com")
    assert settings is not None
    assert settings.enable_dns_rebinding_protection is False
    assert settings.allowed_hosts == []


def test_http_configuration_warns_and_refuses_insecure_bind(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("MCP_ALLOW_INSECURE", raising=False)

    with caplog.at_level(logging.WARNING):
        validate_http_security("127.0.0.1", None)
    assert "MCP_AUTH_TOKEN is not set" in caplog.text

    with pytest.raises(ValueError, match="MCP_ALLOW_INSECURE=true"):
        validate_http_security("0.0.0.0", None)

    monkeypatch.setenv("MCP_ALLOW_INSECURE", "true")
    validate_http_security("0.0.0.0", None)


def test_create_app_serves_tool_call_with_combined_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "testserver")
    monkeypatch.setattr(
        egx_client,
        "supported_indices",
        lambda: ["EGX30"],
    )
    monkeypatch.setattr(
        egx_client,
        "fetch_index_data",
        lambda index, period_days: pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-10-14", "2026-10-15"]),
                "value": [100.0, 101.0],
            }
        ),
    )
    monkeypatch.setenv("EGX_RAISE_TOOL_ERRORS", "true")
    server = create_mcp_server("127.0.0.1", 8000)
    app = create_app({"/egx": server}, auth_token=None)

    async def call_tool() -> None:
        import httpx

        def client_factory(headers, timeout, auth=None):
            headers = {**(headers or {}), "Origin": "http://testserver"}
            return httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
                headers=headers,
                timeout=timeout,
                auth=auth,
            )

        async with app.router.lifespan_context(app):
            async with streamablehttp_client(
                "http://testserver/egx/mcp",
                httpx_client_factory=client_factory,
            ) as (read_stream, write_stream, _get_session_id):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    response = await session.call_tool(
                        "get_index_history",
                        {"index": "EGX30", "period_days": 7},
                    )
                    assert response.isError is not True
                    assert response.structuredContent["rows"][-1]["value"] == 101.0
                    assert response.structuredContent["observations"] == 2
                    invalid = await session.call_tool(
                        "get_index_history",
                        {"index": "NOT_A_REAL_INDEX", "period_days": 7},
                    )
                    assert invalid.isError is True

    asyncio.run(call_tool())


def test_allowed_hosts_are_applied_when_server_is_constructed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "market.example.com,tunnel.example:443")
    server = create_mcp_server("0.0.0.0", 8123)

    assert server.settings.host == "0.0.0.0"
    assert server.settings.port == 8123
    assert server.settings.transport_security.allowed_hosts == [
        "market.example.com",
        "tunnel.example:443",
    ]
    assert "https://market.example.com" in server.settings.transport_security.allowed_origins
