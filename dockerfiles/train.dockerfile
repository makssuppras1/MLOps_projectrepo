# Base image with Python and uv pre-installed
# CRITICAL: Always build for linux/amd64 platform (required for GCP)
# Use: docker buildx build --platform linux/amd64 -f dockerfiles/train.dockerfile -t train:latest .
FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:python3.13-bookworm-slim

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

# Install dependencies
ENV UV_LINK_MODE=copy
RUN uv sync --frozen --no-cache --no-install-project

# Entrypoint for training script
# Data will be accessed via /gcs/ mounted filesystem in Vertex AI
ENTRYPOINT ["uv", "run", "src/pname/train.py"]
