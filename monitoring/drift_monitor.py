"""Simple data drift monitoring using Evidently."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from loguru import logger

# Paths
MONITORING_DIR = Path(__file__).parent
REFERENCE_DATA_PATH = MONITORING_DIR / "reference_data.parquet"
CURRENT_DATA_PATH = MONITORING_DIR / "current_data.parquet"
DRIFT_REPORT_PATH = MONITORING_DIR / "drift_report.html"
DRIFT_RESULTS_PATH = MONITORING_DIR / "drift_results.json"
SCHEMA_PATH = MONITORING_DIR / "schema.json"


def extract_text_features(text: str) -> dict:
    """Extract numerical features from text for drift detection."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    return {
        "text_length": len(text),
        "word_count": len(text.split()),
    }


def build_reference_data(train_texts_path: Optional[str] = None) -> pd.DataFrame:
    """Build reference dataset from training data."""
    if train_texts_path is None:
        # Try common locations
        possible_paths = [
            Path("data/processed/train_texts.json"),
            Path(__file__).parent.parent / "data/processed/train_texts.json",
        ]
        for path in possible_paths:
            if path.exists():
                train_texts_path = str(path)
                break

    if train_texts_path is None:
        raise FileNotFoundError("Training data not found. Provide train_texts_path.")

    logger.info(f"Loading training data from: {train_texts_path}")
    with open(train_texts_path, "r", encoding="utf-8") as f:
        train_texts = json.load(f)

    # Extract features
    features = [extract_text_features(text) for text in train_texts]
    reference_df = pd.DataFrame(features)

    # Save
    REFERENCE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference_df.to_parquet(REFERENCE_DATA_PATH, index=False)
    logger.info(f"Reference data saved: {len(reference_df)} rows")

    return reference_df


def run_drift_monitoring() -> dict:
    """Run drift detection and generate reports.

    Returns:
        Dictionary with monitoring results.
    """
    # Load data
    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Reference data not found: {REFERENCE_DATA_PATH}. Run build_reference_data() first.")
    if not CURRENT_DATA_PATH.exists():
        raise FileNotFoundError(f"Current data not found: {CURRENT_DATA_PATH}. Collect current data first.")

    logger.info("Loading data...")
    reference_df = pd.read_parquet(REFERENCE_DATA_PATH)
    current_df = pd.read_parquet(CURRENT_DATA_PATH)

    # Ensure same columns
    feature_cols = ["text_length", "word_count"]
    reference_df = reference_df[feature_cols]
    current_df = current_df[feature_cols]

    # Prepare Evidently datasets
    data_definition = DataDefinition(numerical_columns=feature_cols)
    reference_dataset = Dataset.from_pandas(reference_df, data_definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition)

    # Generate report
    logger.info("Generating drift report...")
    report = Report([DataDriftPreset()])
    evaluation = report.run(current_data=current_dataset, reference_data=reference_dataset)

    # Save HTML report
    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    evaluation.save_html(str(DRIFT_REPORT_PATH))
    logger.info(f"HTML report saved: {DRIFT_REPORT_PATH}")

    # Extract drift status
    drift_detected = False
    try:
        if hasattr(evaluation, "metric_results") and evaluation.metric_results:
            for metric_result in evaluation.metric_results:
                if isinstance(metric_result, dict):
                    metric_id = str(metric_result.get("metric", {}).get("id", ""))
                    if "DriftedColumnsCount" in metric_id:
                        result_data = metric_result.get("result", {})
                        if isinstance(result_data, dict):
                            count = result_data.get("current", {}).get("value", 0)
                            if count > 0:
                                drift_detected = True
                                break
    except Exception as e:
        logger.warning(f"Could not extract drift status: {e}")

    # Save JSON results
    results = {
        "timestamp": datetime.now().isoformat(),
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "drift_detected": drift_detected,
        "notes": "Drift detected" if drift_detected else "No drift detected",
    }

    DRIFT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved: {DRIFT_RESULTS_PATH}")
    logger.info(f"Drift detected: {drift_detected}")

    return {
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "report_path": str(DRIFT_REPORT_PATH),
        "results_path": str(DRIFT_RESULTS_PATH),
        "drift_detected": drift_detected,
        "notes": results["notes"],
    }


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def build_reference(train_texts_path: Optional[str] = None) -> None:
        """Build reference dataset from training data."""
        build_reference_data(train_texts_path)

    @app.command()
    def monitor() -> None:
        """Run drift monitoring."""
        result = run_drift_monitoring()
        print(json.dumps(result, indent=2))

    app()
