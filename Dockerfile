FROM python:3.12-slim

# Install uv (single line, no extras)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/* \
  && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Cacheable deps layer
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project

# App code + quick project install
COPY . .
RUN uv sync --frozen --no-dev

# Make your CLI available
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
ENTRYPOINT ["mla"]
CMD ["--help"]