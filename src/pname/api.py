"""FastAPI application for model inference."""
from pathlib import Path
from typing import Optional

import torch
import typer
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from transformers import DistilBertTokenizer

from pname.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Global model and tokenizer
model: Optional[MyAwesomeModel] = None
tokenizer: Optional[DistilBertTokenizer] = None

app = FastAPI(
    title="ArXiv Paper Classifier API",
    description="API for classifying ArXiv papers using DistilBERT",
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
        predicted_class: Predicted class index.
        probabilities: List of probabilities for each class.
        confidence: Confidence score (max probability).
    """
    predicted_class: int
    probabilities: list[float]
    confidence: float


def load_model(model_path: str, model_name: str = "distilbert-base-uncased") -> None:
    """Load model and tokenizer.

    Args:
        model_path: Path to the saved model checkpoint.
        model_name: Name of the pretrained model to use for tokenizer.
    """
    global model, tokenizer

    logger.info(f"Loading model from: {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Load model
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    logger.info("Model and tokenizer loaded successfully")


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize model on startup."""
    # Default model path - can be overridden via environment variable or CLI
    default_model_path = "trained_model.pt"
    if Path(default_model_path).exists():
        try:
            load_model(default_model_path)
        except Exception as e:
            logger.warning(f"Failed to load default model: {e}")
            logger.info("Model will need to be loaded manually via /load endpoint")


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
        "device": str(DEVICE),
    }


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

    Args:
        request: Prediction request containing the text to classify.

    Returns:
        Prediction response with class, probabilities, and confidence.

    Raises:
        HTTPException: If model is not loaded.
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please load a model first using /load endpoint."
        )

    try:
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
            logits = outputs["logits"]
            probs = outputs["probs"]
            pred = outputs["preds"]

        # Convert to CPU and extract values
        probs_cpu = probs[0].cpu().numpy().tolist()
        pred_class = int(pred[0].cpu().item())
        confidence = float(probs[0].max().cpu().item())

        return PredictionResponse(
            predicted_class=pred_class,
            probabilities=probs_cpu,
            confidence=confidence,
        )
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
