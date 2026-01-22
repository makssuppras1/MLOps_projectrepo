"""Tests for data drift monitoring."""

import json

import pandas as pd

from monitoring.drift_monitor import build_reference_data, extract_text_features, run_drift_monitoring


class TestExtractTextFeatures:
    """Tests for text feature extraction."""

    def test_extract_features_basic(self):
        """Test feature extraction from basic text."""
        text = "This is a test sentence. It has multiple words."
        features = extract_text_features(text)

        assert "text_length" in features
        assert "word_count" in features
        assert features["text_length"] == len(text)
        assert features["word_count"] == 10

    def test_extract_features_empty(self):
        """Test feature extraction with empty text."""
        features = extract_text_features("")

        assert features["text_length"] == 0
        assert features["word_count"] == 0


class TestBuildReferenceData:
    """Tests for building reference data."""

    def test_build_reference_data(self, tmp_path):
        """Test building reference data from training texts."""
        # Create mock training data
        train_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand text.",
        ]

        train_texts_path = tmp_path / "train_texts.json"
        with open(train_texts_path, "w", encoding="utf-8") as f:
            json.dump(train_texts, f)

        # Build reference data
        reference_df = build_reference_data(train_texts_path=str(train_texts_path))

        # Verify output
        assert len(reference_df) == 3
        assert "text_length" in reference_df.columns
        assert "word_count" in reference_df.columns


class TestDriftMonitoring:
    """Tests for drift monitoring."""

    def test_no_drift_sanity(self, tmp_path):
        """Test: No-drift sanity - current data is sample from reference data."""
        # Create reference data
        reference_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand text.",
        ]

        # Build reference data
        train_texts_path = tmp_path / "train_texts.json"
        with open(train_texts_path, "w", encoding="utf-8") as f:
            json.dump(reference_texts, f)

        # Temporarily override paths
        import monitoring.drift_monitor as dm

        original_ref_path = dm.REFERENCE_DATA_PATH
        original_curr_path = dm.CURRENT_DATA_PATH
        original_report_path = dm.DRIFT_REPORT_PATH
        original_results_path = dm.DRIFT_RESULTS_PATH

        try:
            dm.REFERENCE_DATA_PATH = tmp_path / "reference_data.parquet"
            dm.CURRENT_DATA_PATH = tmp_path / "current_data.parquet"
            dm.DRIFT_REPORT_PATH = tmp_path / "drift_report.html"
            dm.DRIFT_RESULTS_PATH = tmp_path / "drift_results.json"

            # Build reference
            build_reference_data(train_texts_path=str(train_texts_path))

            # Create current data as subset of reference (no drift expected)
            current_texts = reference_texts[:2]
            current_features = [extract_text_features(text) for text in current_texts]
            current_df = pd.DataFrame(current_features)
            current_df.to_parquet(dm.CURRENT_DATA_PATH, index=False)

            # Run drift monitoring
            result = run_drift_monitoring()

            # Verify outputs exist
            assert dm.DRIFT_REPORT_PATH.exists()
            assert dm.DRIFT_RESULTS_PATH.exists()

            # Verify structure
            assert "reference_rows" in result
            assert "current_rows" in result
            assert "drift_detected" in result
            assert result["reference_rows"] == 3
            assert result["current_rows"] == 2

        finally:
            # Restore original paths
            dm.REFERENCE_DATA_PATH = original_ref_path
            dm.CURRENT_DATA_PATH = original_curr_path
            dm.DRIFT_REPORT_PATH = original_report_path
            dm.DRIFT_RESULTS_PATH = original_results_path

    def test_drift_monitoring_output_contract(self, tmp_path):
        """Test that drift monitoring returns the required output contract."""
        # Create minimal reference and current data
        reference_texts = ["Reference text 1", "Reference text 2"]
        current_texts = ["Current text 1", "Current text 2"]

        # Build reference data
        train_texts_path = tmp_path / "train_texts.json"
        with open(train_texts_path, "w", encoding="utf-8") as f:
            json.dump(reference_texts, f)

        # Temporarily override paths
        import monitoring.drift_monitor as dm

        original_ref_path = dm.REFERENCE_DATA_PATH
        original_curr_path = dm.CURRENT_DATA_PATH
        original_report_path = dm.DRIFT_REPORT_PATH
        original_results_path = dm.DRIFT_RESULTS_PATH

        try:
            dm.REFERENCE_DATA_PATH = tmp_path / "reference_data.parquet"
            dm.CURRENT_DATA_PATH = tmp_path / "current_data.parquet"
            dm.DRIFT_REPORT_PATH = tmp_path / "drift_report.html"
            dm.DRIFT_RESULTS_PATH = tmp_path / "drift_results.json"

            build_reference_data(train_texts_path=str(train_texts_path))

            # Create current data
            current_features = [extract_text_features(text) for text in current_texts]
            current_df = pd.DataFrame(current_features)
            current_df.to_parquet(dm.CURRENT_DATA_PATH, index=False)

            result = run_drift_monitoring()

            # Verify required output contract fields
            required_fields = [
                "reference_rows",
                "current_rows",
                "report_path",
                "results_path",
                "drift_detected",
                "notes",
            ]

            for field in required_fields:
                assert field in result, f"Missing required field: {field}"

            # Verify types
            assert isinstance(result["reference_rows"], int)
            assert isinstance(result["current_rows"], int)
            assert isinstance(result["drift_detected"], bool)
            assert isinstance(result["report_path"], str)
            assert isinstance(result["results_path"], str)
            assert isinstance(result["notes"], str)

        finally:
            # Restore original paths
            dm.REFERENCE_DATA_PATH = original_ref_path
            dm.CURRENT_DATA_PATH = original_curr_path
            dm.DRIFT_REPORT_PATH = original_report_path
            dm.DRIFT_RESULTS_PATH = original_results_path
