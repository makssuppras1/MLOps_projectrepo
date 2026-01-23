# Dockerfile for Drift Detection API Service
# Supports multi-platform builds (ARM64 for local, AMD64 for GCP)
#
# Single platform build:
#   docker buildx build --platform linux/amd64 -f monitoring_service/Dockerfile -t monitoring:latest .
#   docker buildx build --platform linux/arm64 -f monitoring_service/Dockerfile -t monitoring:latest .
#
# Multi-platform build (recommended - works on both ARM64 Mac and AMD64 GCP):
#   docker buildx build --platform linux/amd64,linux/arm64 -f monitoring_service/Dockerfile -t monitoring:latest --push .
#   (or without --push for local use: docker buildx build --platform linux/amd64,linux/arm64 -f monitoring_service/Dockerfile -t monitoring:latest --load .)
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
COPY monitoring/ monitoring/
COPY monitoring_service/ monitoring_service/
COPY configs/ configs/

# Set working directory
WORKDIR /

# Install dependencies
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-cache --no-install-project

# Expose port for API
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Entrypoint for drift detection API
ENTRYPOINT ["uv", "run", "uvicorn", "monitoring_service.app.main:app", "--host", "0.0.0.0", "--port", "8001"]
