"""FastAPI application for model inference."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import typer
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel
from transformers import DistilBertTokenizer

from pname.model import MyAwesomeModel

try:
    from pname.model_tfidf import TFIDFXGBoostModel

    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False
    logger.warning("TF-IDF model not available. Only PyTorch models will be supported.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Global model and tokenizer (supports both PyTorch and TF-IDF models)
model: Optional[MyAwesomeModel] = None  # Can also be TFIDFXGBoostModel if TFIDF_AVAILABLE
tokenizer: Optional[DistilBertTokenizer] = None
model_type: Optional[str] = None  # "pytorch" or "tfidf"
class_names: Optional[dict[int, str]] = None  # Mapping of class index to category name

# Request logging for drift monitoring (CSV format)
REQUEST_LOG_PATH = Path(os.getenv("REQUEST_LOG_PATH", "monitoring/api_requests.csv"))
REQUEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="ArXiv Paper Classifier API",
    description="API for classifying ArXiv papers using DistilBERT or TF-IDF + XGBoost",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    """Request model for prediction.

    Attributes:
        text: Text to classify (title + abstract).
    """

    text: str


class PredictionResponse(BaseModel):
    """Response model for prediction.

    Attributes:
        predicted_class: Predicted class index (0-4).
        predicted_class_name: Name of the predicted category (if available).
        probabilities: List of probabilities for each class.
        class_probabilities: Dictionary mapping class indices to probabilities.
        confidence: Confidence score (max probability).
    """

    predicted_class: int
    predicted_class_name: Optional[str] = None
    probabilities: list[float]
    class_probabilities: Optional[dict[str, float]] = None
    confidence: float


def load_model(model_path: str, model_name: str = "distilbert-base-uncased") -> None:
    """Load model and tokenizer.

    Supports both PyTorch (.pt) and TF-IDF (.pkl) models.

    Args:
        model_path: Path to the saved model checkpoint (.pt or .pkl).
        model_name: Name of the pretrained model to use for tokenizer (only for PyTorch models).
    """
    global model, tokenizer, model_type

    logger.info(f"Loading model from: {model_path}")

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Detect model type by file extension
    if model_path.endswith(".pkl"):
        # Load TF-IDF + XGBoost model
        if not TFIDF_AVAILABLE:
            raise ImportError("TF-IDF model support not available. Install required dependencies.")

        logger.info("Detected TF-IDF model (.pkl), loading...")
        model = TFIDFXGBoostModel.load(str(model_path_obj))
        model_type = "tfidf"
        tokenizer = None  # TF-IDF doesn't use a tokenizer
        logger.info("TF-IDF model loaded successfully")
    else:
        # Load PyTorch model
        logger.info("Detected PyTorch model (.pt), loading...")
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = MyAwesomeModel().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        model_type = "pytorch"
        logger.info("PyTorch model and tokenizer loaded successfully")


def load_class_names() -> Optional[dict[int, str]]:
    """Load class name mapping from processed data.

    Returns:
        Dictionary mapping class index to category name, or None if not found.
    """
    global class_names

    project_root = Path(__file__).parent.parent

    # Try to load category mapping from processed data
    mapping_paths = [
        project_root / "data/processed/category_mapping.json",
        Path("data/processed/category_mapping.json"),
        Path("/gcs/mlops_project_data_bucket1-europe-west1/data/processed/category_mapping.json"),
        Path("/gcs/mlops_project_data_bucket1/data/processed/category_mapping.json"),
    ]

    for mapping_path in mapping_paths:
        if mapping_path.exists():
            try:
                with open(mapping_path, "r", encoding="utf-8") as f:
                    category_to_label = json.load(f)
                # Reverse mapping: label -> category
                label_to_category = {int(v): k for k, v in category_to_label.items()}
                logger.info(f"Loaded class names from: {mapping_path}")
                return label_to_category
            except Exception as e:
                logger.warning(f"Failed to load class names from {mapping_path}: {e}")
                continue

    # Fallback: use generic class names if mapping not found
    logger.info("Category mapping not found, using generic class names")
    return {i: f"Class {i}" for i in range(5)}


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize model on startup."""
    global class_names

    project_root = Path(__file__).parent.parent

    # Load class names first
    class_names = load_class_names()

    # Try to load model from common locations (check both .pt and .pkl)
    model_paths = [
        # Current directory
        Path("trained_model.pt"),
        Path("trained_model.pkl"),
        # Project root
        project_root / "trained_model.pt",
        project_root / "trained_model.pkl",
        # WandB artifacts (TF-IDF model)
        project_root / "artifacts" / "model-tfidf-fbomdu0l:v0" / "trained_model.pkl",
        # WandB artifacts (PyTorch model)
        project_root / "artifacts" / "model-a5yv5hjp:v0" / "trained_model.pt",
        # Outputs directory
        Path("/outputs/trained_model.pt"),
        Path("/outputs/trained_model.pkl"),
        # GCS mounts (if available)
        Path("/gcs/mlops_project_data_bucket1-europe-west1/experiments/tfidf_xgboost/model/trained_model.pkl"),
        Path("/gcs/mlops_project_data_bucket1-europe-west1/trained_model.pt"),
    ]

    # Also check for any model in artifacts folder
    artifacts_dir = project_root / "artifacts"
    if artifacts_dir.exists():
        for artifact_path in artifacts_dir.rglob("trained_model.*"):
            if artifact_path.suffix in [".pt", ".pkl"]:
                if artifact_path not in model_paths:
                    model_paths.append(artifact_path)

    logger.info(f"Checking {len(model_paths)} locations for model auto-loading...")
    for model_path in model_paths:
        path_obj = Path(model_path) if isinstance(model_path, str) else model_path
        if path_obj.exists():
            try:
                load_model(str(path_obj))
                logger.info(f"✅ Auto-loaded model from: {path_obj}")
                return
            except Exception as e:
                logger.warning(f"Failed to load model from {path_obj}: {e}")
                continue

    logger.info("No model found in default locations. Load model manually via /load endpoint")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint.

    Returns:
        Welcome message.
    """
    return {"message": "ArXiv Paper Classifier API", "status": "running"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Health status and model availability.
    """
    return {
        "status": "healthy",
        "model_loaded": str(model is not None),
        "model_type": model_type or "none",
        "device": str(DEVICE),
    }


@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring() -> HTMLResponse:
    """Monitoring endpoint returning Evidently HTML report.

    Returns:
        HTML report showing data drift analysis.
    """
    try:
        from monitoring.drift_monitor import run_drift_monitoring

        # Run drift monitoring
        result = run_drift_monitoring()

        # Read and return HTML report
        report_path = Path(result["report_path"])
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content, status_code=200)
        else:
            return HTMLResponse(
                content=f"<h1>Monitoring Report Not Found</h1><p>Report path: {report_path}</p>",
                status_code=404,
            )
    except Exception as e:
        logger.error(f"Monitoring endpoint error: {e}")
        return HTMLResponse(
            content=f"<h1>Monitoring Error</h1><p>{str(e)}</p>",
            status_code=500,
        )


@app.post("/load")
async def load_model_endpoint(model_path: str) -> dict[str, str]:
    """Load a model checkpoint.

    Args:
        model_path: Path to the model checkpoint file.

    Returns:
        Success message.
    """
    try:
        load_model(model_path)
        return {"status": "success", "message": f"Model loaded from {model_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict the class of a text.

    Supports both PyTorch (DistilBERT) and TF-IDF + XGBoost models.

    Args:
        request: Prediction request containing the text to classify.

    Returns:
        Prediction response with class, probabilities, and confidence.

    Raises:
        HTTPException: If model is not loaded.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please load a model first using /load endpoint.")

    # Log will be done after prediction to include outputs

    try:
        if model_type == "tfidf":
            # TF-IDF + XGBoost prediction
            probs = model.predict_proba([request.text])[0]
            pred_class = int(model.predict([request.text])[0])
            confidence = float(max(probs))
            probs_list = probs.tolist()

            # Get class name if available
            pred_class_name = class_names.get(pred_class, f"Class {pred_class}") if class_names else None

            # Create class probabilities dictionary
            class_probs = {}
            if class_names:
                for idx, prob in enumerate(probs_list):
                    class_name = class_names.get(idx, f"Class {idx}")
                    class_probs[class_name] = float(prob)
            else:
                for idx, prob in enumerate(probs_list):
                    class_probs[f"Class {idx}"] = float(prob)

            response = PredictionResponse(
                predicted_class=pred_class,
                predicted_class_name=pred_class_name,
                probabilities=probs_list,
                class_probabilities=class_probs,
                confidence=confidence,
            )

            # Log input-output data for monitoring
            try:
                import csv

                log_exists = REQUEST_LOG_PATH.exists()
                with open(REQUEST_LOG_PATH, "a", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    if not log_exists:
                        writer.writerow(["time", "text_length", "word_count", "predicted_class", "confidence"])
                    text_len = len(request.text)
                    word_count = len(request.text.split())
                    writer.writerow([datetime.now().isoformat(), text_len, word_count, pred_class, confidence])
            except Exception as e:
                logger.warning(f"Failed to log for monitoring: {e}")

            return response
        else:
            # PyTorch (DistilBERT) prediction
            if tokenizer is None:
                raise HTTPException(status_code=503, detail="Tokenizer not loaded for PyTorch model.")

            # Tokenize input
            encoded = tokenizer(
                request.text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(DEVICE)
            attention_mask = encoded["attention_mask"].to(DEVICE)

            # Get prediction
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = outputs["probs"]
                pred = outputs["preds"]

            # Convert to CPU and extract values
            probs_list = probs[0].cpu().numpy().tolist()
            pred_class = int(pred[0].cpu().item())
            confidence = float(probs[0].max().cpu().item())

            # Get class name if available
            pred_class_name = class_names.get(pred_class, f"Class {pred_class}") if class_names else None

            # Create class probabilities dictionary
            class_probs = {}
            if class_names:
                for idx, prob in enumerate(probs_list):
                    class_name = class_names.get(idx, f"Class {idx}")
                    class_probs[class_name] = float(prob)
            else:
                for idx, prob in enumerate(probs_list):
                    class_probs[f"Class {idx}"] = float(prob)

            response = PredictionResponse(
                predicted_class=pred_class,
                predicted_class_name=pred_class_name,
                probabilities=probs_list,
                class_probabilities=class_probs,
                confidence=confidence,
            )

            # Log input-output data for monitoring
            try:
                import csv

                log_exists = REQUEST_LOG_PATH.exists()
                with open(REQUEST_LOG_PATH, "a", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    if not log_exists:
                        writer.writerow(["time", "text_length", "word_count", "predicted_class", "confidence"])
                    text_len = len(request.text)
                    word_count = len(request.text.split())
                    writer.writerow([datetime.now().isoformat(), text_len, word_count, pred_class, confidence])
            except Exception as e:
                logger.warning(f"Failed to log for monitoring: {e}")

            return response
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def run_api(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    model_path: Optional[str] = typer.Option(None, help="Path to model checkpoint"),
    model_name: str = typer.Option("distilbert-base-uncased", help="Pretrained model name"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
    """Run the FastAPI server.

    Args:
        host: Host address to bind to.
        port: Port number to bind to.
        model_path: Optional path to model checkpoint (will be loaded on startup).
        model_name: Name of the pretrained model.
        reload: Enable auto-reload for development.
    """
    if model_path:
        # Load model before starting server
        try:
            load_model(model_path, model_name)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Starting server without model. Load model via /load endpoint.")

    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    typer.run(run_api)
