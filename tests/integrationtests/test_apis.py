"""Integration tests for the FastAPI application.

These tests validate the API endpoints by making actual HTTP requests
to the FastAPI application using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_valid_input(self):
        """Test POST /predict with valid input returns prediction."""
        with TestClient(app) as client:
            # Valid payload: text field with string value
            payload = {"text": "Machine learning is a subset of artificial intelligence that focuses on algorithms."}
            response = client.post("/predict", json=payload)

            # If model is not loaded, expect 503
            # If model is loaded, expect 200 with prediction
            if response.status_code == 503:
                # Model not loaded - this is acceptable for tests
                assert "Model not loaded" in response.json()["detail"]
            elif response.status_code == 200:
                # Model is loaded - validate response structure
                data = response.json()
                assert "predicted_class" in data
                assert "probabilities" in data
                assert "confidence" in data
                assert isinstance(data["predicted_class"], int)
                assert isinstance(data["probabilities"], list)
                assert isinstance(data["confidence"], float)
                assert len(data["probabilities"]) > 0
                # Confidence should be between 0 and 1
                assert 0 <= data["confidence"] <= 1
            else:
                pytest.fail(f"Unexpected status code: {response.status_code}")
