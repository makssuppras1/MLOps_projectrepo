# Base image with Python and uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install build essentials
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy essential files
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY app/ app/
COPY data/ data/
COPY configs/ configs/

# Set working directory
WORKDIR /

# Install dependencies with cache mount for faster rebuilds
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-cache --no-install-project

# Expose port for API
EXPOSE 8000

# Entrypoint for API server
ENTRYPOINT ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
