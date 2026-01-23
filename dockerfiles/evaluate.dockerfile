# Base image with Python and uv pre-installed
# CRITICAL: For GCP deployment, build with: docker buildx build --platform linux/amd64 ...
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
COPY configs/ configs/
# Note: data/ and models/ are not copied - they should be mounted

# Set working directory
WORKDIR /

# Install dependencies with cache mount for faster rebuilds
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-cache --no-install-project

# Entrypoint for evaluation script
ENTRYPOINT ["uv", "run", "src/pname/evaluate.py"]
