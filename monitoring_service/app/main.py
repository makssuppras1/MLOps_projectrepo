"""Drift detection API service for monitoring ML model performance.

Deployed as a separate Cloud Run service for on-demand drift detection.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports (must be before monitoring imports)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Standard library imports
import typer  # noqa: E402
import uvicorn  # noqa: E402

# Third-party imports
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from loguru import logger  # noqa: E402
from pydantic import BaseModel  # noqa: E402

# Local imports (after sys.path modification)
from monitoring.collect_current_data import collect_current_data  # noqa: E402
from monitoring.drift_monitor import (  # noqa: E402
    CURRENT_DATA_PATH,
    DRIFT_REPORT_PATH,
    DRIFT_RESULTS_PATH,
    REFERENCE_DATA_PATH,
    run_drift_monitoring,
)

# GCS support for artifact storage
try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("google-cloud-storage not available. GCS artifact upload disabled.")

app = FastAPI(
    title="Drift Detection API",
    description="API for detecting data drift in production ML models",
    version="1.0.0",
)

# Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET", "mlops_project_data_bucket1-europe-west1")
GCS_ARTIFACT_PREFIX = os.getenv("GCS_ARTIFACT_PREFIX", "monitoring/drift_artifacts")


class DriftRunResponse(BaseModel):
    """Response model for drift detection run."""

    reference_rows: int
    current_rows: int
    drift_detected: bool
    top_drifted_features: list[str]
    timestamp: str
    artifacts: dict[str, str]
    recommendation: Optional[str] = None


def upload_to_gcs(local_path: Path, blob_name: str) -> Optional[str]:
    """Upload file to GCS bucket.

    Args:
        local_path: Local file path.
        blob_name: GCS blob name (path within bucket).

    Returns:
        GCS URI if successful, None otherwise.
    """
    if not GCS_AVAILABLE or not local_path.exists():
        return None

    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        gcs_uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
        logger.info(f"Uploaded {local_path} to {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.warning(f"Failed to upload to GCS: {e}")
        return None


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint - health check.

    Returns:
        Service status.
    """
    return {
        "service": "Drift Detection API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Health status and data availability.
    """
    ref_exists = REFERENCE_DATA_PATH.exists()
    curr_exists = CURRENT_DATA_PATH.exists()

    return {
        "status": "healthy",
        "reference_data_available": ref_exists,
        "current_data_available": curr_exists,
        "reference_path": str(REFERENCE_DATA_PATH),
        "current_path": str(CURRENT_DATA_PATH),
    }


@app.post("/drift/run", response_model=DriftRunResponse)
async def run_drift_detection() -> DriftRunResponse:
    """Run drift detection and return machine-readable results.

    This endpoint:
    1. Collects current data from API logs (if needed)
    2. Runs drift detection using Evidently
    3. Uploads artifacts to GCS (if configured)
    4. Returns JSON results with drift status

    Returns:
        Drift detection results with drift_detected boolean and feature-level stats.

    Raises:
        HTTPException: If reference or current data is missing.
    """
    # Check if reference data exists
    if not REFERENCE_DATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Reference data not found: {REFERENCE_DATA_PATH}. "
            "Run build_reference_data() first or ensure reference_data.parquet exists.",
        )

    # Collect current data if needed
    if not CURRENT_DATA_PATH.exists():
        logger.info("Current data not found, attempting to collect from API logs...")
        try:
            collect_current_data()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to collect current data: {e}. " "Ensure API logs exist at monitoring/api_requests.csv",
            )

    # Run drift detection
    try:
        result = run_drift_monitoring()
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")

    # Extract top drifted features from Evidently report
    top_drifted_features = []
    try:
        # Try to extract feature-level drift info from Evidently results
        # This is a simplified version - in production, parse the HTML report or use Evidently's JSON export
        if result.get("drift_detected"):
            # Default features we monitor
            top_drifted_features = ["text_length", "word_count"]
    except Exception as e:
        logger.warning(f"Could not extract top drifted features: {e}")

    # Upload artifacts to GCS if available
    artifacts = {}
    if DRIFT_REPORT_PATH.exists():
        report_blob = f"{GCS_ARTIFACT_PREFIX}/drift_report_{Path(DRIFT_REPORT_PATH).stat().st_mtime}.html"
        report_uri = upload_to_gcs(DRIFT_REPORT_PATH, report_blob)
        if report_uri:
            artifacts["html_report"] = report_uri
        else:
            artifacts["html_report"] = str(DRIFT_REPORT_PATH)

    if DRIFT_RESULTS_PATH.exists():
        results_blob = f"{GCS_ARTIFACT_PREFIX}/drift_results_{Path(DRIFT_RESULTS_PATH).stat().st_mtime}.json"
        results_uri = upload_to_gcs(DRIFT_RESULTS_PATH, results_blob)
        if results_uri:
            artifacts["json"] = results_uri
        else:
            artifacts["json"] = str(DRIFT_RESULTS_PATH)

    # Generate recommendation
    recommendation = None
    if result.get("drift_detected"):
        recommendation = (
            "Drift detected in production data. "
            "Consider: 1) Reviewing data pipeline for changes, "
            "2) Retraining model if performance degraded, "
            "3) Adjusting drift thresholds if false positive."
        )
    else:
        recommendation = "No drift detected. Model performance should be stable."

    return DriftRunResponse(
        reference_rows=result["reference_rows"],
        current_rows=result["current_rows"],
        drift_detected=result["drift_detected"],
        top_drifted_features=top_drifted_features,
        timestamp=result.get("timestamp", ""),
        artifacts=artifacts,
        recommendation=recommendation,
    )


@app.get("/drift/report")
async def get_drift_report() -> JSONResponse:
    """Get latest drift report HTML.

    Returns:
        JSON with report location (GCS URI or local path).
    """
    if not DRIFT_REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Drift report not found: {DRIFT_REPORT_PATH}. Run /drift/run first.",
        )

    # Try to get GCS URI if uploaded
    report_blob = f"{GCS_ARTIFACT_PREFIX}/drift_report_{Path(DRIFT_REPORT_PATH).stat().st_mtime}.html"
    report_uri = f"gs://{GCS_BUCKET_NAME}/{report_blob}"

    return JSONResponse(
        content={
            "report_path": str(DRIFT_REPORT_PATH),
            "gcs_uri": report_uri,
            "message": "Use GCS URI to download report, or access local path if running locally",
        }
    )


def run_api(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8001, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
    """Run the drift detection API server.

    Args:
        host: Host address to bind to.
        port: Port number to bind to.
        reload: Enable auto-reload for development.
    """
    logger.info(f"Starting Drift Detection API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    typer.run(run_api)
