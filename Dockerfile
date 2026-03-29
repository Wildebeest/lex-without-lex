FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv sync --frozen --no-dev --no-install-project 2>/dev/null || uv sync --no-dev --no-install-project

COPY src/ src/

RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

CMD ["uv", "run", "uvicorn", "lex_without_lex.server:app", "--host", "0.0.0.0", "--port", "8080"]
