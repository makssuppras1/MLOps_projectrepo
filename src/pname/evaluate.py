"""Model evaluation script for ArXiv text classification."""

from typing import Dict

import torch
import typer
from loguru import logger
from transformers import DistilBertTokenizer

from pname.data import arxiv_dataset
from pname.model import MyAwesomeModel
from pname.train import collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(
    model_checkpoint: str,
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 32,
    max_length: int = 512,
) -> Dict[str, float]:
    """
    Evaluate a trained model on the test set.

    Loads a trained model checkpoint and evaluates it on the ArXiv test dataset.
    Computes and logs test accuracy.

    Args:
        model_checkpoint: Path to the saved model checkpoint file.
        model_name: Name of the pretrained model for tokenizer. Defaults to "distilbert-base-uncased".
        batch_size: Batch size for evaluation. Defaults to 32.
        max_length: Maximum sequence length for tokenization. Defaults to 512.

    Returns:
        Dictionary containing evaluation metrics (e.g., 'accuracy').

    Raises:
        FileNotFoundError: If model checkpoint file is not found.
    """
    logger.info("Evaluating like my life depended on it")
    logger.info(f"Loading model from: {model_checkpoint}")

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Initialize model (will load weights from checkpoint)
    model = MyAwesomeModel().to(DEVICE)

    # Load model checkpoint, mapping to CPU first to handle cross-platform compatibility
    checkpoint = torch.load(model_checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    logger.info("Model loaded successfully")

    # Load test dataset
    _, test_set = arxiv_dataset()

    # Create collate function with tokenizer
    def collate(batch):
        return collate_fn(batch, tokenizer, max_length)

    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
    )
    logger.info(f"Test dataset size: {len(test_set)}")

    model.eval()
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, targets) in enumerate(test_dataloader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            targets = targets.long().to(DEVICE)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs["preds"]

            correct += (preds == targets).float().sum().item()
            total += targets.size(0)

            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Processed {batch_idx + 1}/{len(test_dataloader)} batches")

    test_accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Test accuracy: {test_accuracy:.4f} ({correct}/{total})")

    return {"accuracy": test_accuracy}


if __name__ == "__main__":
    typer.run(evaluate)
