"""Locust load testing file for the ArXiv Paper Classifier API.

This file defines load tests that simulate multiple users making requests
to the API endpoints. It tests both the health endpoint and the prediction endpoint.

Usage:
    # IMPORTANT: If you see a blank page, kill any existing Locust processes first:
    #   lsof -ti:8089 | xargs kill  # Kill process on port 8089
    #   Or: pkill -f locust  # Kill all Locust processes

    # 1. Start your API server first (in a separate terminal):
    uv run invoke api

    # 2. Run Locust (choose one option):

    # Option A: Use --host flag (recommended)
    uv run locust -f tests/performancetests/locustfile.py --host=http://localhost:8000

    # Option B: Set the endpoint environment variable
    export MYENDPOINT=http://localhost:8000
    uv run locust -f tests/performancetests/locustfile.py

    # 3. Open http://localhost:8089 in your browser
    #    (If you see a blank page, make sure no other Locust instance is running)

    # 4. In the Locust web UI:
    #    - Set number of users (e.g., 10)
    #    - Set spawn rate (e.g., 2 users/second)
    #    - Click "Start Swarming"
"""

import os

from locust import HttpUser, task


class APIUser(HttpUser):
    """Locust user class that simulates API requests.

    This class defines the behavior of a simulated user making requests
    to the API. Each user will execute the tasks defined below.

    The host should be set via --host flag when running Locust, or via
    MYENDPOINT environment variable. If neither is set, Locust will fail
    with a clear error message.
    """

    # Host will be set from --host flag or MYENDPOINT env var
    # Setting to empty string allows --host flag to work properly
    host = os.getenv("MYENDPOINT", "").strip().rstrip("/") or ""

    def on_start(self):
        """Called when a simulated user starts.

        Validates and logs the endpoint being tested.
        """
        if not self.host:
            raise ValueError(
                "Host not set! Please either:\n"
                "  1. Set MYENDPOINT environment variable: export MYENDPOINT=http://localhost:8000\n"
                "  2. Use --host flag: locust --host=http://localhost:8000"
            )
        print(f"Testing endpoint: {self.host}")

    @task(3)
    def health_check(self):
        """Health check endpoint - weighted 3x more than predict.

        This simulates frequent health checks that monitoring systems might perform.
        """
        self.client.get("/health", name="health")

    @task(1)
    def root_endpoint(self):
        """Root endpoint check."""
        self.client.get("/", name="root")

    @task(5)
    def predict(self):
        """Prediction endpoint - weighted 5x (most common operation).

        Sends a valid prediction request with sample text.
        """
        payload = {
            "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms "
            "and statistical models to enable computer systems to improve their performance "
            "on a specific task through experience."
        }
        self.client.post("/predict", json=payload, name="predict")
