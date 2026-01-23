# FastAPI Dockerfile for production deployment
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.cargo/bin/uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-dev

# Copy application code
COPY app/ ./app/
COPY src/ ./src/

# Copy model artifact (if available)
COPY artifacts/ ./artifacts/

# Set PYTHONPATH to include src directory for imports
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose port 80
EXPOSE 80

# Health check for production API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
