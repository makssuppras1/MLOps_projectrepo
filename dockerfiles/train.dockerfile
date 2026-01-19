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
COPY configs/ configs/
# Note: data/ is not copied - it should be mounted or pulled on the instance

# Set working directory
WORKDIR /

# Install dependencies
ENV UV_LINK_MODE=copy
RUN uv sync --locked --no-cache --no-install-project

# Entrypoint for training script
ENTRYPOINT ["uv", "run", "src/pname/train.py"]
