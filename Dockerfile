FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	UV_COMPILE_BYTECODE=1 \
	UV_LINK_MODE=copy \
	MCP_TRANSPORT=http \
	MCP_HOST=0.0.0.0 \
	MCP_PORT=8000

WORKDIR /app

COPY . .

RUN uv sync --locked --no-dev

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "egx-mcp-server"]
