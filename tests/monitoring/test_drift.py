"""Tests for data drift monitoring."""

import json

import pandas as pd

from monitoring.drift_monitor import build_reference_data, extract_text_features, run_drift_monitoring


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
