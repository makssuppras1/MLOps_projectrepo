# Base image with Python and uv pre-installed
# Supports multi-platform builds (ARM64 for local, AMD64 for GCP)
#
# Single platform build:
#   docker buildx build --platform linux/amd64 -f dockerfiles/evaluate.dockerfile -t evaluate:latest .
#   docker buildx build --platform linux/arm64 -f dockerfiles/evaluate.dockerfile -t evaluate:latest .
#
# Multi-platform build (recommended - works on both ARM64 Mac and AMD64 GCP):
#   docker buildx build --platform linux/amd64,linux/arm64 -f dockerfiles/evaluate.dockerfile -t evaluate:latest --push .
#   (or without --push for local use: docker buildx build --platform linux/amd64,linux/arm64 -f dockerfiles/evaluate.dockerfile -t evaluate:latest --load .)
#
# Docker automatically selects the native architecture when running the image.
# TARGETPLATFORM is automatically set by buildx when using --platform flag
ARG TARGETPLATFORM=linux/amd64
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
