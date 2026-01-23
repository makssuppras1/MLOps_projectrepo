from typing import Tuple

import torch
import typer
from loguru import logger
from transformers import AutoTokenizer

from pname.data import arxiv_dataset
from pname.model import MyAwesomeModel

try:
    from pname.model_tfidf import TFIDFXGBoostModel

    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def collate_fn_eval(
    batch: list[Tuple[str, torch.Tensor]], tokenizer, max_length: int = 512
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function to tokenize text batches for evaluation.

    Args:
        batch: List of (text, label) tuples.
        tokenizer: Tokenizer to use for encoding texts.
        max_length: Maximum sequence length for tokenization.

    Returns:
        Tuple of (input_ids, attention_mask, labels) tensors.
    """
    texts, labels = zip(*batch)

    # Tokenize texts
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    labels = torch.stack([label.long() for label in labels])

    return encoded["input_ids"], encoded["attention_mask"], labels


def evaluate(
    model_checkpoint: str = typer.Argument(..., help="Path to model checkpoint file (.pt or .pkl)"),
    model_name: str = typer.Option("distilbert-base-uncased", help="Pretrained model name (for PyTorch models only)"),
    batch_size: int = typer.Option(32, help="Batch size for evaluation (PyTorch models only)"),
    max_length: int = typer.Option(512, help="Maximum sequence length (PyTorch models only)"),
    max_samples: int = typer.Option(None, help="Maximum number of samples to evaluate (None = all)"),
) -> None:
    """Evaluate a trained model on the test set.

    Supports both PyTorch (.pt) and TF-IDF (.pkl) models.

    Args:
        model_checkpoint: Path to the saved model checkpoint (.pt or .pkl).
        model_name: Name of the pretrained model to use for tokenizer (PyTorch models only).
        batch_size: Batch size for evaluation (PyTorch models only).
        max_length: Maximum sequence length for tokenization (PyTorch models only).
        max_samples: Maximum number of samples to evaluate (None = all).
    """
    logger.info("Evaluating like my life depended on it")
    logger.info(f"Loading model from: {model_checkpoint}")

    # Detect model type by file extension
    if model_checkpoint.endswith(".pkl"):
        # Evaluate TF-IDF model
        if not TFIDF_AVAILABLE:
            raise ImportError("TF-IDF model support not available. Install required dependencies.")

        logger.info("Detected TF-IDF model (.pkl), loading...")
        model = TFIDFXGBoostModel.load(model_checkpoint)
        logger.info("TF-IDF model loaded successfully")

        # Load test data with same number of categories as model
        _, _, test_set = arxiv_dataset(max_categories=model.num_labels)

        # Optionally use a subset for faster evaluation
        if max_samples is not None and max_samples < len(test_set):
            import random

            random.seed(42)  # For reproducibility
            indices = random.sample(range(len(test_set)), max_samples)
            test_set = torch.utils.data.Subset(test_set, indices)
            logger.info(f"Using subset of {max_samples} samples for evaluation")
        elif max_samples is not None:
            logger.info(f"max_samples ({max_samples}) >= dataset size ({len(test_set)}), using full dataset")

        logger.info(f"Test dataset size: {len(test_set)}")

        # Evaluate TF-IDF model
        texts = [test_set[i][0] for i in range(len(test_set))]
        labels = [test_set[i][1].item() for i in range(len(test_set))]

        predictions = model.predict(texts)
        correct = sum(pred == label for pred, label in zip(predictions, labels))
        total = len(labels)

        test_accuracy = correct / total
        logger.info(f"Test accuracy: {test_accuracy:.4f} ({correct}/{total})")

    else:
        # Evaluate PyTorch model
        logger.info("Detected PyTorch model (.pt), loading...")

        # Initialize tokenizer (AutoTokenizer works with DistilBERT, TinyBERT, BERT, etc.)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with default config (5 labels for ultra-fast training)
        model = MyAwesomeModel().to(DEVICE)

        # Load model checkpoint, mapping to CPU first to handle cross-platform compatibility
        # Use weights_only=False for PyTorch models (they're from trusted source)
        checkpoint = torch.load(model_checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint)
        logger.info("PyTorch model loaded successfully")

        # Use same category reduction as training (5 categories)
        _, _, test_set = arxiv_dataset(max_categories=model.num_labels)

        # Optionally use a subset for faster evaluation
        if max_samples is not None and max_samples < len(test_set):
            import random

            random.seed(42)  # For reproducibility
            indices = random.sample(range(len(test_set)), max_samples)
            test_set = torch.utils.data.Subset(test_set, indices)
            logger.info(f"Using subset of {max_samples} samples for evaluation")
        elif max_samples is not None:
            logger.info(f"max_samples ({max_samples}) >= dataset size ({len(test_set)}), using full dataset")

        def collate(batch):
            return collate_fn_eval(batch, tokenizer, max_length)

        test_dataloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=0,  # Set to 0 to avoid pickling issues with tokenizer
        )
        logger.info(f"Test dataset size: {len(test_set)}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(test_dataloader):
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                preds = logits.argmax(dim=1)

                correct += (preds == labels).float().sum().item()
                total += labels.size(0)

                if (batch_idx + 1) % 10 == 0:
                    logger.debug(f"Processed {batch_idx + 1}/{len(test_dataloader)} batches")

        test_accuracy = correct / total
        logger.info(f"Test accuracy: {test_accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    typer.run(evaluate)
