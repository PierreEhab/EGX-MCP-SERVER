# EGX MCP Server

A Python MCP server for retrieving and analyzing Egyptian Exchange (EGX)
indices and individual company stocks. Index data comes from the
[`egx`](https://pypi.org/project/egx/) package; delayed daily stock data comes
from Yahoo Finance through `yfinance`. It supports Python 3.11+, is managed
with `uv`, and uses the official MCP Python SDK.

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

Stock bars and dividend history are cached persistently in SQLite. The stock
cache database is `stock_cache.sqlite3` in `EGX_CACHE_DIR`, or by default
`%LOCALAPPDATA%\egx-mcp` on Windows when `LOCALAPPDATA` is set, and
`~/.cache/egx-mcp` otherwise.
Completed daily bars are not cached or returned until 30 minutes after the
EGX session close (14:15 Cairo time); a still-forming daily bar is excluded.
Dividend history has a separate seven-day cache lifetime. Yahoo calls are
serialized and throttled, with a configurable minimum interval.

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
| `EGX_CACHE_DIR` | `%LOCALAPPDATA%\egx-mcp` on Windows when available; `~/.cache/egx-mcp` otherwise | Directory for the persistent SQLite stock cache. |
| `EGX_YAHOO_MIN_INTERVAL_SECONDS` | `0.5` | Minimum delay between Yahoo calls; rate-limit responses are retried with backoff. |
| `EGX_LOG_LEVEL` | `INFO` | Python logging level; source errors are logged to stderr so stdio MCP stays clean. |
| `EGX_RAISE_TOOL_ERRORS` | `false` | With `true`, raise the MCP SDK `ToolError` so MCP clients receive `isError`; otherwise return the existing structured error payload. |
| `MCP_TRANSPORT` | `stdio` | Default transport: `stdio`, `http`, or `streamable-http`. |
| `MCP_HOST` | `127.0.0.1` | HTTP bind host. |
| `MCP_PORT` | `8000` | HTTP bind port. |
| `MCP_AUTH_TOKEN` | Unset | Optional bearer token for HTTP; recommended for all remote deployments. |
| `MCP_ALLOW_INSECURE` | `false` | Explicitly allow a non-loopback HTTP bind without a token. |
| `MCP_ALLOWED_HOSTS` | SDK/local defaults | Comma-separated Host-header allow-list for MCP DNS-rebinding protection. |

## Windows stdio client configuration

For all clients, point `--directory` at the absolute project path. Replace
`C:\\path\\to\\EGX-MCP-SERVER` with the directory containing this checkout.
The server uses stdio; its library output is kept off stdout so it does not
interfere with the MCP protocol.

### Claude Desktop

Add this to Claude Desktop's `claude_desktop_config.json`:

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

### Claude Code

Run this in PowerShell from the project directory to register the local server
with Claude Code:

```powershell
claude mcp add --transport stdio egx -- uv --directory "$PWD" run egx-mcp-server
```

Alternatively replace `$PWD` with the absolute project path.

### VS Code

Create `.vscode\mcp.json` in the project directory (or merge the server entry
into your existing MCP configuration):

```json
{
  "servers": {
    "egx": {
      "type": "stdio",
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
- `list_stocks(sector=None, limit=50, include_unavailable=False, verbose=False)` — list
  catalog companies. Unavailable Yahoo symbols are hidden by default;
  `include_unavailable=true` adds them with an `unavailable` flag. Inferred
  sectors are proposals, identified by `sector_source`. Catalog rows are
  compact by default; `verbose=true` includes the full catalog record.
- `find_stock(query, verbose=False)` — search available companies by ticker,
  alias, or name. Results are compact unless `verbose=true`.
- `get_stock_history(symbol, period_days=90, frequency="daily", max_rows=None,
  adjusted=False)` — return daily OHLCV bars or Thursday-based weekly/monthly
  samples. The lookback is clamped to 1-9000 calendar days. `Close` is
  split-adjusted; `adjusted=true` returns dividend-adjusted `Adj Close` as the
  row close.
- `get_stock_summary(symbol, period_days=365)` — summarize price and total
  returns, changes, moving averages, volatility, drawdowns, volume, 52-week
  levels, and trailing dividend yield when available.
- `get_stock_fundamentals(symbol)` — return catalog sector and best-effort
  Yahoo market capitalization, P/E, 52-week range, and currency, plus
  history-derived dividend yield and average volume. Missing values remain
  null and are noted in `data_quality`.
- `get_dividends(symbol, years=5)` — return available historical dividend
  payments filtered to 1-50 years (`years` is clamped to that range).
- `compare_stocks(symbols, period_days=365)` — compare up to 10 unique
  companies by dividend-adjusted total return, with volatility, drawdown, and
  per-symbol errors.
- `stock_vs_index(symbol, index="EGX30", period_days=365)` — compare daily
  stock and index returns on common dates, including correlation, beta, and
  excess return.
- `analysis_snapshot(symbol)` — compact stock summary, relative performance,
  fundamentals, recent dividends, key price levels, and data-quality notes.

The `research_company(symbol)` and `build_trade_plan(symbol, horizon,
risk_tolerance)` MCP prompts guide agents to call the relevant stock tools and
produce structured research or scenario analysis. A prompt is guidance for an
agent, not a substitute for reviewing the returned data.

Index history and summary responses include `index`, `as_of`, `source`,
`period_days_applied`, and `observations`. Stock history and summary responses
include the corresponding `symbol` and stock price-basis metadata. Each
comparison row includes `period_days_applied` and `observations`. `as_of` is
the latest valid observation used. The EGX index source provides daily
`datetime` and `value` fields, not OHLC candles.

By default, errors retain the structured `error` object with a code and clear
message. Set `EGX_RAISE_TOOL_ERRORS=true` to make the server raise the SDK
`ToolError` instead, which lets MCP clients mark the result as an error.

## Limitations

- Yahoo stock prices are delayed and may be missing or stale for a company.
  Catalog inclusion does not guarantee Yahoo coverage; tools report known
  unavailable symbols explicitly. Stale or missing data should be reported,
  never guessed.
- Yahoo daily `Close` is split-adjusted and may differ from the historical
  price actually traded after bonus shares or other splits. `Adj Close` is
  additionally adjusted for dividends. Stock tools do not request intraday
  bars.
- Yahoo's sector, industry, dividend-yield, and average-volume fields may be
  absent. Sectors come from the symbol catalog; inferred classifications are
  marked as such. Dividend yield and average volume are computed from history
  when the required data is available.
- The EGX generally trades Sunday-Thursday; the source and exchange
  publication schedule determine the latest available observation. No
  Friday/Saturday bars are expected. EGX daily price limits and thin liquidity
  for many names can affect interpretation.
- Index data relies on the `egx` package, which may scrape the exchange
  website and could break if that site changes. Its current response contains
  daily `datetime` and `value` fields, not OHLC candles.

**This service provides analysis, not financial advice.**

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
