# syntax=docker/dockerfile:1
# BrokerIQ runtime image. Serves both entrypoints:
#   - API:    docker compose run api    (uvicorn, port 8000)
#   - MCP:    docker compose run mcp    (stdio server for MCP clients)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Layer-cache dependencies first (pyproject + lock only)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the package + corpus
COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev

EXPOSE 8000

# Default to dev; override at runtime for prod:
#   docker compose run --rm api env BROKERIQ_ENV=prod [...]
ENV BROKERIQ_ENV=dev \
    BROKERIQ_LOG_LEVEL=INFO

CMD ["uv", "run", "uvicorn", "brokeriq.api:app", "--host", "0.0.0.0", "--port", "8000"]
