# Base image with Python and uv pre-installed
# Supports multi-platform builds (ARM64 for local, AMD64 for GCP)
#
# Single platform build:
#   docker buildx build --platform linux/amd64 -f dockerfiles/api.dockerfile -t api:latest .
#   docker buildx build --platform linux/arm64 -f dockerfiles/api.dockerfile -t api:latest .
#
# Multi-platform build (recommended - works on both ARM64 Mac and AMD64 GCP):
#   docker buildx build --platform linux/amd64,linux/arm64 -f dockerfiles/api.dockerfile -t api:latest --push .
#   (or without --push for local use: docker buildx build --platform linux/amd64,linux/arm64 -f dockerfiles/api.dockerfile -t api:latest --load .)
#
# Docker automatically selects the native architecture when running the image.
# TARGETPLATFORM is automatically set by buildx when using --platform flag
ARG TARGETPLATFORM=linux/amd64
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install build essentials
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy essential files
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
# Note: data/ is not copied - it should be mounted if needed

# Set working directory
WORKDIR /

# Install dependencies with cache mount for faster rebuilds
# Use --no-dev to exclude dev dependencies (black, pytest, etc.) for smaller image
ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1
ENV UV_HTTP_TIMEOUT=600
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-cache --no-install-project --no-dev

# Expose port for API
EXPOSE 8000

# Health check for API container
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint for API server
ENTRYPOINT ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
