"""FastMCP server exposing Egyptian Exchange index data and analytics."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
from typing import Any, Literal, Sequence

import pandas as pd
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings

from . import analytics, egx_client
from .constants import INDEX_DESCRIPTIONS
from .http_app import _is_loopback

SOURCE = egx_client.SOURCE
_logger = logging.getLogger(__name__)
_mcp: FastMCP | None = None
_mcp_lock = threading.Lock()


def _transport_security(host: str) -> TransportSecuritySettings | None:
    configured_hosts = os.getenv("MCP_ALLOWED_HOSTS")
    token = os.getenv("MCP_AUTH_TOKEN")
    if configured_hosts is not None:
        allowed_hosts = [
            entry.strip() for entry in configured_hosts.split(",") if entry.strip()
        ]
        if not allowed_hosts:
            raise ValueError("MCP_ALLOWED_HOSTS must contain at least one host.")
        allowed_origins: list[str] = []
        for allowed_host in allowed_hosts:
            exact_host = (
                allowed_host[:-2]
                if allowed_host.endswith(":*")
                else allowed_host
            )
            for scheme in ("http", "https"):
                allowed_origins.append(f"{scheme}://{exact_host}")
                if allowed_host.endswith(":*"):
                    allowed_origins.append(f"{scheme}://{allowed_host}")
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts
            + [
                entry[:-2]
                for entry in allowed_hosts
                if entry.endswith(":*")
            ],
            allowed_origins=allowed_origins,
        )
    if token and not _is_loopback(host):
        _logger.warning(
            "MCP_AUTH_TOKEN is set for a non-loopback bind; disabling DNS-rebinding protection because bearer auth is enabled and MCP_ALLOWED_HOSTS is unset."
        )
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
            allowed_hosts=[],
            allowed_origins=[],
        )
    if _is_loopback(host):
        return None
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            f"{host}:*",
            "localhost:*",
            "localhost",
            "127.0.0.1:*",
            "127.0.0.1",
            "[::1]:*",
            "[::1]",
        ],
        allowed_origins=[
            "http://localhost:*",
            "http://localhost",
            "https://localhost:*",
            "https://localhost",
            "http://127.0.0.1:*",
            "http://127.0.0.1",
            "https://127.0.0.1:*",
            "https://127.0.0.1",
            "http://[::1]:*",
            "http://[::1]",
            "https://[::1]:*",
            "https://[::1]",
        ],
    )


def _new_mcp(host: str, port: int) -> FastMCP:
    settings = _transport_security(host)
    if settings is not None:
        _logger.info(
            "HTTP allowed hosts: %s",
            settings.allowed_hosts or "(disabled)",
        )
    return FastMCP(
        "egx-mcp-server",
        instructions=(
            "Retrieve historical Egyptian Exchange index values and calculate "
            "index performance analytics. Data is index-only."
        ),
        host=host,
        port=port,
        transport_security=settings,
    )


def _error(
    error_code: str,
    message: str,
    index: str | list[str] | None = None,
) -> dict[str, Any]:
    return {
        "index": index,
        "as_of": None,
        "source": SOURCE,
        "error": {"code": error_code, "message": message},
    }


def _return_error(
    error_code: str,
    message: str,
    index: str | list[str] | None = None,
) -> dict[str, Any]:
    return _error(error_code, message, index)


def _maybe_raise_tool_error(payload: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("EGX_RAISE_TOOL_ERRORS", "").casefold() == "true":
        error = payload.get("error")
        if error:
            raise ToolError(error["message"])
    return payload


def _error_for(
    exc: egx_client.EGXClientError,
    index: str | list[str] | None,
) -> dict[str, Any]:
    if isinstance(exc, egx_client.InvalidIndexError):
        code = "INVALID_INDEX"
    elif isinstance(exc, egx_client.EmptyDataError):
        code = "EMPTY_DATA"
    elif isinstance(exc, egx_client.EGXNetworkError):
        code = "NETWORK_ERROR"
    elif isinstance(exc, egx_client.InvalidPeriodError):
        code = "INVALID_PERIOD"
    else:
        code = "DATA_SOURCE_ERROR"
    return _error(code, str(exc), index)


def _load_sync(index: str, period_days: int) -> tuple[str, int, pd.DataFrame]:
    canonical = egx_client.resolve_index(index)
    period = egx_client.clamp_period_days(period_days)
    frame = egx_client.fetch_index_data(canonical, period)
    return canonical, period, frame


async def _load(
    index: str,
    period_days: int,
) -> tuple[str, int, pd.DataFrame] | dict[str, Any]:
    """Resolve and load an index off the event loop, returning safe tool errors."""
    try:
        return await asyncio.to_thread(_load_sync, index, period_days)
    except egx_client.EGXClientError as exc:
        return _error_for(exc, index)


def _sample_history(
    frame: pd.DataFrame,
    frequency: Literal["daily", "weekly", "monthly"],
    max_rows: int | None,
) -> pd.DataFrame:
    if max_rows is not None and (
        isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1
    ):
        raise ValueError("max_rows must be a positive integer or None.")
    observations = frame.copy(deep=False).sort_values("datetime")
    if frequency != "daily":
        key = "W-THU" if frequency == "weekly" else "M"
        periods = observations["datetime"].dt.to_period(key)
        observations = (
            observations.assign(_period=periods)
            .groupby("_period", sort=True)
            .tail(1)
            .drop(columns="_period")
        )
    observations = observations.reset_index(drop=True)
    if max_rows is not None:
        observations = observations.tail(max_rows).reset_index(drop=True)
    return observations


def list_indices() -> dict[str, Any]:
    """List the index codes supported by the installed EGX data package.

    Returns canonical index codes and a one-line description for each. Use
    these codes with the history, summary, and comparison tools.
    """
    codes = egx_client.supported_indices()
    indices = [
        {
            "code": code,
            "description": INDEX_DESCRIPTIONS.get(
                code, f"{code} index published by the Egyptian Exchange."
            ),
        }
        for code in codes
    ]
    return {"index": None, "as_of": None, "source": SOURCE, "indices": indices}


async def get_index_history(
    index: str,
    period_days: int = 30,
    frequency: Literal["daily", "weekly", "monthly"] = "daily",
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Return daily or downsampled historical EGX index values as ISO-date rows.

    ``index`` is matched case-insensitively and ``period_days`` is clamped to
    1-1825 calendar days. ``frequency`` keeps the last observation per day,
    week, or month; choose weekly/monthly for long periods to reduce payload
    size. ``max_rows`` optionally keeps only the most recent positive number of
    rows. The source provides ``datetime`` and ``value``, not OHLC fields.
    """
    if frequency not in ("daily", "weekly", "monthly"):
        payload = _return_error(
            "INVALID_ARGUMENT",
            "frequency must be 'daily', 'weekly', or 'monthly'.",
            index,
        )
        return _maybe_raise_tool_error(payload)
    loaded = await _load(index, period_days)
    if isinstance(loaded, dict):
        return _maybe_raise_tool_error(loaded)
    canonical, period, frame = loaded
    try:
        sampled = _sample_history(frame, frequency, max_rows)
    except ValueError as exc:
        payload = _return_error("INVALID_ARGUMENT", str(exc), canonical)
        return _maybe_raise_tool_error(payload)
    rows = analytics.history_rows(sampled)
    as_of = rows[-1]["date"] if rows else None
    payload = {
        "index": canonical,
        "as_of": as_of,
        "source": SOURCE,
        "rows": rows,
        "period_days_applied": period,
        "observations": len(rows),
    }
    return _maybe_raise_tool_error(payload)


async def get_index_summary(index: str, period_days: int = 90) -> dict[str, Any]:
    """Calculate recent performance and risk statistics for one EGX index.

    Returns latest value, previous close, daily move, period high/low and
    return, 20/50-observation simple moving averages when enough data exists,
    annualized volatility, drawdown from the period high, return over the last
    five sessions when available, and the valid observation date range.
    ``period_days`` is a calendar-day lookback clamped to 1-1825.
    """
    loaded = await _load(index, period_days)
    if isinstance(loaded, dict):
        return _maybe_raise_tool_error(loaded)
    canonical, period, frame = loaded
    summary = analytics.summarize(frame)
    payload = {
        "index": canonical,
        **summary,
        "source": SOURCE,
        "period_days_applied": period,
        "observations": len(frame),
    }
    return _maybe_raise_tool_error(payload)


async def compare_indices(
    indices: list[str],
    period_days: int = 90,
) -> dict[str, Any]:
    """Compare period return, volatility, and drawdown for EGX indices.

    Index codes are case-insensitive; duplicates are removed while preserving
    input order. The response includes each index's applied calendar-day
    period and observation count, per-index errors without discarding
    successful comparisons, and the best performer by period return.
    """
    if not isinstance(indices, list) or not indices:
        payload = _return_error(
            "INVALID_ARGUMENT",
            "indices must be a non-empty list.",
            [],
        )
        return _maybe_raise_tool_error(payload)
    if any(not isinstance(index, str) for index in indices):
        payload = _return_error(
            "INVALID_ARGUMENT",
            "every index code must be a string.",
            indices,
        )
        return _maybe_raise_tool_error(payload)

    unique_indices: list[str] = []
    seen: set[str] = set()
    for index in indices:
        key = index.strip().casefold()
        if key not in seen:
            seen.add(key)
            unique_indices.append(index)

    try:
        period = egx_client.clamp_period_days(period_days)
    except egx_client.InvalidPeriodError as exc:
        payload = _error_for(exc, unique_indices)
        return _maybe_raise_tool_error(payload)
    if len(unique_indices) > len(egx_client.supported_indices()):
        payload = _return_error(
            "INVALID_ARGUMENT",
            "The number of unique requested indices exceeds the supported index count.",
            unique_indices,
        )
        return _maybe_raise_tool_error(payload)

    loaded_results = await asyncio.gather(
        *(_load(index, period) for index in unique_indices)
    )
    comparisons: list[dict[str, Any]] = []
    for requested_index, loaded in zip(unique_indices, loaded_results):
        if isinstance(loaded, dict):
            comparisons.append(
                {
                    "index": requested_index,
                    "period_days_applied": period,
                    "observations": 0,
                    "period_return_pct": None,
                    "annualized_volatility_pct": None,
                    "drawdown_from_period_high_pct": None,
                    "as_of": None,
                    "error": loaded["error"],
                }
            )
            continue

        canonical, applied_period, frame = loaded
        summary = analytics.summarize(frame)
        comparisons.append(
            {
                "index": canonical,
                "period_days_applied": applied_period,
                "observations": len(frame),
                "period_return_pct": summary["period_return_pct"],
                "annualized_volatility_pct": summary[
                    "annualized_volatility_pct"
                ],
                "drawdown_from_period_high_pct": summary[
                    "drawdown_from_period_high_pct"
                ],
                "as_of": summary["as_of"],
            }
        )

    candidates = [
        row
        for row in comparisons
        if row.get("error") is None and row["period_return_pct"] is not None
    ]
    best = (
        max(candidates, key=lambda row: row["period_return_pct"])["index"]
        if candidates
        else None
    )
    successful_dates = [
        row["as_of"] for row in comparisons if row.get("error") is None and row["as_of"]
    ]
    payload = {
        "index": [row["index"] for row in comparisons],
        "as_of": max(successful_dates) if successful_dates else None,
        "source": SOURCE,
        "comparisons": comparisons,
        "best_performer": best,
    }
    if (
        os.getenv("EGX_RAISE_TOOL_ERRORS", "").casefold() == "true"
        and comparisons
        and all("error" in row for row in comparisons)
    ):
        first_error = next(
            (row.get("error", {"message": "Unknown error"}) for row in comparisons if row.get("error")),
            {"message": "All requested indices failed."},
        )
        raise ToolError(first_error["message"])
    return payload


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def create_mcp_server(host: str, port: int) -> FastMCP:
    """Create an independently configured MCP server with all EGX tools."""
    server = _new_mcp(host, port)
    for tool in (
        list_indices,
        get_index_history,
        get_index_summary,
        compare_indices,
    ):
        server.tool()(tool)
    return server


def get_mcp_server(host: str | None = None, port: int | None = None) -> FastMCP:
    """Return the process's lazily constructed default MCP server."""
    global _mcp
    if _mcp is None:
        with _mcp_lock:
            if _mcp is None:
                configured_host = host or os.getenv("MCP_HOST", "127.0.0.1")
                configured_port = port or _port(os.getenv("MCP_PORT", "8000"))
                _mcp = create_mcp_server(configured_host, configured_port)
    return _mcp


def __getattr__(name: str) -> Any:
    if name == "mcp":
        return get_mcp_server()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(argv: Sequence[str] | None = None) -> None:
    """Start the MCP server over stdio or streamable HTTP."""
    egx_client.configure_logging()
    default_transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if default_transport == "streamable-http":
        default_transport = "http"
    parser = argparse.ArgumentParser(description="Serve EGX index data over MCP.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "http", "streamable-http"),
        default=default_transport,
        help="MCP transport (default: stdio, or MCP_TRANSPORT).",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MCP_HOST", "127.0.0.1"),
        help="HTTP bind host (default: MCP_HOST or 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=_port(os.getenv("MCP_PORT", "8000")),
        help="HTTP bind port (default: MCP_PORT or 8000).",
    )
    args = parser.parse_args(argv)
    transport = "http" if args.transport == "streamable-http" else args.transport
    if transport == "stdio":
        server = get_mcp_server(args.host, args.port)
        server.run(transport="stdio")
        return
    from .http_app import create_app

    token = os.getenv("MCP_AUTH_TOKEN")
    http_server = get_mcp_server(args.host, args.port)
    app = create_app({"": http_server}, auth_token=token)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
