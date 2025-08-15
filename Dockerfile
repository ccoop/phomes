FROM python:3.12-slim

# Install uv
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/* \
  && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Cacheable deps layer
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project

# App code + quick project install
COPY . .
RUN uv sync --frozen

# Make the CLI available
ENV PATH="/app/.venv/bin:$PATH"

# Train all models during build
RUN mla train --all

EXPOSE 8000
ENTRYPOINT ["bash"]