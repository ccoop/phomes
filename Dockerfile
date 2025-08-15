FROM python:3.12-slim

# Install uv
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/* \
  && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Cacheable deps layer
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project

# App code + quick project install
COPY . .
RUN uv sync

# Make the CLI available
ENV PATH="/app/.venv/bin:$PATH"

# Train all models during build
RUN mla train --all

# Train true baseline with original 7 features for presentation metrics
RUN mla train knn_baseline --include="bedrooms,bathrooms,sqft_living,sqft_lot,floors,sqft_above,sqft_basement" --note="True baseline - original 7 features from create_model.py"

EXPOSE 8000
ENTRYPOINT ["bash"]