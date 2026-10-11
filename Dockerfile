FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Cloud Run injects $PORT. The SQLite audit log lives on /tmp (in-memory on
# Cloud Run); mount a volume or point GAFFER_DB elsewhere to keep it.
ENV PORT=8080 GAFFER_DB=/tmp/gaffer.db PATH="/app/.venv/bin:$PATH"
CMD ["sh", "-c", "uvicorn gaffer.api:app --host 0.0.0.0 --port ${PORT}"]
