# EGX MCP Server

A Python MCP server for retrieving and analyzing Egyptian Exchange (EGX) index
data through the [`egx`](https://pypi.org/project/egx/) package. It supports
Python 3.11+, is managed with `uv`, and uses the official MCP Python SDK,
pandas, and requests.

The EGX market generally trades Sunday-Thursday, 10:00-14:15 Cairo time.
Consequently, the latest available observation can be from the previous
trading day, for example on weekends, public holidays, or before the market
has published a new observation.

## Setup

```console
uv sync
```

The EGX source response is cached in memory for five minutes by default.
`EGX_CACHE_TTL_SECONDS` changes this duration; `0` disables the cache. When a
longer period is already cached, shorter requests are sliced from that result.
Concurrent requests for the same index are coalesced.

## Run

### stdio (default)

```console
uv run egx-mcp-server
```

### Streamable HTTP

```console
uv run egx-mcp-server --transport http --host 127.0.0.1 --port 8000
```

The HTTP MCP endpoint is `/mcp`, and `/health` returns a small health status.
The same settings can be supplied with `MCP_TRANSPORT`, `MCP_HOST`, and
`MCP_PORT`. `MCP_TRANSPORT=streamable-http` is accepted as an alias for `http`.

HTTP requests should be protected by setting `MCP_AUTH_TOKEN`. It is sent by
clients as `Authorization: Bearer <token>`. All routes except `/health` require
the token. If no token is configured, the server logs a warning. A
non-loopback HTTP bind without a token is refused unless
`MCP_ALLOW_INSECURE=true` is explicitly set; do not use that override on an
untrusted network.

The MCP SDK's DNS-rebinding/Host-header protection is configured when each
server is constructed. Set `MCP_ALLOWED_HOSTS` to a comma-separated list of
allowed Host header values when using a real domain or tunnel, for example:

```powershell
$env:MCP_ALLOWED_HOSTS = "egx.example.com:*,localhost:*"
$env:MCP_AUTH_TOKEN = "replace-with-a-secret"
uv run egx-mcp-server --transport http --host 0.0.0.0 --port 8000
```

Allow-list entries may be exact hosts (including ports) or use `:*` to allow
any port for that host.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `EGX_CACHE_TTL_SECONDS` | `300` | Source-data cache TTL in seconds; `0` disables caching. |
| `EGX_INDEX_LIST_TTL_SECONDS` | Not used | No list cache is needed: installed `egx.get_supported_indices()` returns a local constant and does not make a network request. |
| `EGX_LOG_LEVEL` | `INFO` | Python logging level; source errors are logged to stderr so stdio MCP stays clean. |
| `EGX_RAISE_TOOL_ERRORS` | `false` | With `true`, raise the MCP SDK `ToolError` so MCP clients receive `isError`; otherwise return the existing structured error payload. |
| `MCP_TRANSPORT` | `stdio` | Default transport: `stdio`, `http`, or `streamable-http`. |
| `MCP_HOST` | `127.0.0.1` | HTTP bind host. |
| `MCP_PORT` | `8000` | HTTP bind port. |
| `MCP_AUTH_TOKEN` | Unset | Optional bearer token for HTTP; recommended for all remote deployments. |
| `MCP_ALLOW_INSECURE` | `false` | Explicitly allow a non-loopback HTTP bind without a token. |
| `MCP_ALLOWED_HOSTS` | SDK/local defaults | Comma-separated Host-header allow-list for MCP DNS-rebinding protection. |

## Claude Desktop configuration

Use the absolute path to this project directory in `--directory`. For example,
on Windows:

```json
{
  "mcpServers": {
    "egx": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\path\\to\\EGX-MCP-SERVER",
        "run",
        "egx-mcp-server"
      ]
    }
  }
}
```

## MCP Inspector

To inspect the local stdio server:

```console
npx @modelcontextprotocol/inspector uv --directory . run egx-mcp-server
```

For HTTP, start the server and then run `npx @modelcontextprotocol/inspector`;
connect to `http://localhost:8000/mcp`. If bearer authentication is enabled,
provide the token in the Inspector's authorization settings.

## Tools

- `list_indices()` — list supported index codes and short descriptions.
- `get_index_history(index, period_days=30, frequency="daily", max_rows=None)` —
  return ISO-dated history. `period_days` is clamped to 1-1825 calendar days.
  Use `frequency="weekly"` or `"monthly"` for long windows; each period keeps
  its last observation. `max_rows` keeps only the newest positive number of
  returned rows.
- `get_index_summary(index, period_days=90)` — calculate latest value, daily
  move, period high/low and return, available 20/50-observation simple moving
  averages, annualized volatility, drawdown from the period high, return over
  the last five observations (when available), and the date range used.
- `compare_indices(indices, period_days=90)` — compare period return,
  annualized volatility, and drawdown; duplicate codes are removed and a
  failing index is reported without discarding successful comparisons.

Successful history and summary responses include `index`, `as_of`, `source`,
`period_days_applied`, and `observations`. Each comparison row includes
`period_days_applied` and `observations`. `as_of` is the latest valid
observation used. The data source currently provides daily `datetime` and
`value` fields, not OHLC candles.

By default, errors retain the structured `error` object with a code and clear
message. Set `EGX_RAISE_TOOL_ERRORS=true` to make the server raise the SDK
`ToolError` instead, which lets MCP clients mark the result as an error.

## Limitations

- This server covers **indices only**; it does not provide individual stock
  data.
- It relies on the `egx` package, which may scrape the exchange website and
  could break if that site changes.
- The current package response contains daily `datetime` and `value` columns,
  not open/high/low/close candles.
- Source availability and the exchange's publication schedule determine how
  recent the returned data is.

## Tests

Tests use synthetic data, mocks, and an in-process HTTP transport; they do not
make network requests.

```console
uv run pytest
```

## Build distributions

Use `python scripts/build.py` to run `uv build` and remove the
`pyproject.toml.orig` compatibility backup that `uv_build` adds to source
distributions. The generated wheel is left unchanged.
