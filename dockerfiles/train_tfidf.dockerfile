# Base image with Python and uv pre-installed
# Supports multi-platform builds (ARM64 for local, AMD64 for GCP)
#
# Single platform build:
#   docker buildx build --platform linux/amd64 -f dockerfiles/train_tfidf.dockerfile -t train-tfidf:latest .
#   docker buildx build --platform linux/arm64 -f dockerfiles/train_tfidf.dockerfile -t train-tfidf:latest .
#
# Multi-platform build (recommended - works on both ARM64 Mac and AMD64 GCP):
#   docker buildx build --platform linux/amd64,linux/arm64 -f dockerfiles/train_tfidf.dockerfile -t train-tfidf:latest --push .
#   (or without --push for local use: docker buildx build --platform linux/amd64,linux/arm64 -f dockerfiles/train_tfidf.dockerfile -t train-tfidf:latest --load .)
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
# Note: data/ is not copied - it should be mounted or pulled on the instance

# Set working directory
WORKDIR /

# Create /outputs directory and make it a VOLUME
RUN mkdir -p /outputs
VOLUME ["/outputs"]

# Install dependencies with cache mount for faster rebuilds
# Use --no-dev to exclude dev dependencies (black, pytest, etc.) for smaller image
ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1
ENV UV_HTTP_TIMEOUT=600
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-cache --no-install-project --no-dev

# Entrypoint for TF-IDF training script
# Data will be accessed via /gcs/ mounted filesystem in Vertex AI
# PYTHONUNBUFFERED=1 is already set above for unbuffered output
# Using exec form ensures proper signal handling (Python process receives SIGTERM/SIGINT)
ENTRYPOINT ["uv", "run", "src/pname/train_tfidf.py"]
