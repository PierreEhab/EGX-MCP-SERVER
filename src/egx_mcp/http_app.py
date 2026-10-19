"""HTTP application assembly, lifecycle, and optional bearer authentication."""

from __future__ import annotations

import hmac
import ipaddress
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator, Mapping

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

_logger = logging.getLogger(__name__)


class BearerAuthMiddleware:
    """Pure ASGI bearer-token middleware with an unauthenticated health route."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.expected = token.encode("utf-8")

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http" or scope.get("path") == "/health":
            await self.app(scope, receive, send)
            return

        supplied = next(
            (
                value
                for name, value in scope.get("headers", [])
                if name.lower() == b"authorization"
            ),
            b"",
        )
        token = supplied[7:] if supplied[:7].lower() == b"bearer " else b""
        if not hmac.compare_digest(token, self.expected):
            response = JSONResponse(
                {"error": "Unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _is_loopback(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    address = host.strip("[]").split("%", maxsplit=1)[0]
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def validate_http_security(host: str, token: str | None) -> None:
    """Warn about unauthenticated HTTP and reject unsafe external binds."""
    if token:
        return
    _logger.warning(
        "MCP_AUTH_TOKEN is not set; HTTP MCP requests will be unauthenticated."
    )
    allow_insecure = os.getenv("MCP_ALLOW_INSECURE", "").casefold() == "true"
    if not _is_loopback(host) and not allow_insecure:
        raise ValueError(
            "Refusing unauthenticated non-loopback HTTP bind; set "
            "MCP_AUTH_TOKEN or MCP_ALLOW_INSECURE=true to proceed."
        )


def create_app(
    servers: Mapping[str, FastMCP] | None = None,
    auth_token: str | None = None,
) -> Starlette:
    """Mount one or more FastMCP HTTP apps with a combined server lifespan.

    Mapping keys are mount paths. The default root mount preserves this
    project's existing ``/mcp`` endpoint; additional servers can be mounted at
    paths such as ``/egx`` without giving up their session-manager lifecycles.
    """
    if servers is None:
        from .server import mcp

        servers = {"": mcp}
    if not servers:
        raise ValueError("At least one FastMCP server must be provided.")

    token = auth_token or os.getenv("MCP_AUTH_TOKEN")
    for host in {server.settings.host for server in servers.values()}:
        validate_http_security(host, token)

    allowed_hosts: list[str] = []
    for server in servers.values():
        transport_security = getattr(server.settings, "transport_security", None)
        allowed_hosts.extend(getattr(transport_security, "allowed_hosts", []))
    if allowed_hosts:
        _logger.info(
            "Effective HTTP allowed hosts: %s",
            ", ".join(sorted(set(allowed_hosts))),
        )

    server_apps = {
        server: server.streamable_http_app()
        for server in servers.values()
    }

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            for server in servers.values():
                await stack.enter_async_context(server.session_manager.run())
            yield

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    routes = [Route("/health", health, methods=["GET"])]
    for mount_path, server in sorted(
        servers.items(), key=lambda item: len(item[0].strip("/")), reverse=True
    ):
        prefix = mount_path.strip("/")
        routes.append(Mount(f"/{prefix}" if prefix else "/", app=server_apps[server]))

    middleware = [Middleware(BearerAuthMiddleware, token=token)] if token else []
    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)
