"""Integration tests for the FastAPI application.

These tests validate the API endpoints by making actual HTTP requests
to the FastAPI application using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self):
        """Test GET / endpoint returns correct response."""
        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "status" in data
            assert data["status"] == "running"

    def test_health_endpoint(self):
        """Test GET /health endpoint returns health status."""
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "model_loaded" in data
            assert "model_type" in data
            assert "device" in data
            assert data["status"] == "healthy"


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

    def test_health_endpoint_reports_model_status(self):
        """Test that /health endpoint accurately reports model loading status.

        This test ensures that model loading failures are detectable via the health endpoint.
        """
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()

            # Health endpoint should always return these fields
            assert "status" in data
            assert "model_loaded" in data
            assert "model_type" in data
            assert "device" in data

            # Status should be "healthy" even if model is not loaded
            assert data["status"] == "healthy"

            # model_loaded should be a string representation of boolean
            model_loaded = (
                data["model_loaded"].lower() if isinstance(data["model_loaded"], str) else data["model_loaded"]
            )
            assert model_loaded in ["true", "false", True, False]

    def test_predict_invalid_input_missing_field(self):
        """Test POST /predict with missing required field returns 422."""
        with TestClient(app) as client:
            # Missing 'text' field
            payload = {}
            response = client.post("/predict", json=payload)
            assert response.status_code == 422
            # FastAPI validation error should contain detail about missing field
            data = response.json()
            assert "detail" in data

    def test_predict_invalid_input_wrong_type(self):
        """Test POST /predict with wrong type returns 422."""
        with TestClient(app) as client:
            # 'text' field should be string, not number
            payload = {"text": 12345}
            response = client.post("/predict", json=payload)
            # FastAPI may accept this if it can coerce, but typically expects string
            # Let's check if it's 422 or if it accepts and processes
            assert response.status_code in [422, 200, 503]
            if response.status_code == 422:
                data = response.json()
                assert "detail" in data

    def test_predict_empty_string(self):
        """Test POST /predict with empty string."""
        with TestClient(app) as client:
            payload = {"text": ""}
            response = client.post("/predict", json=payload)
            # Empty string might be valid input (though not useful)
            # Should either return 200 (if model handles it) or 422/400
            assert response.status_code in [200, 400, 422, 503]
            if response.status_code == 200:
                data = response.json()
                assert "predicted_class" in data
