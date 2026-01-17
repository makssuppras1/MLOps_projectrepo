from typing import Tuple

import torch
import typer
from loguru import logger
from transformers import DistilBertTokenizer

from pname.data import arxiv_dataset
from pname.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def collate_fn_eval(
    batch: list[Tuple[str, torch.Tensor]], tokenizer: DistilBertTokenizer, max_length: int = 512
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
    model_checkpoint: str = typer.Argument(..., help="Path to model checkpoint file"),
    model_name: str = typer.Option("distilbert-base-uncased", help="Pretrained model name"),
    batch_size: int = typer.Option(32, help="Batch size for evaluation"),
    max_length: int = typer.Option(512, help="Maximum sequence length"),
) -> None:
    """Evaluate a trained model on the test set.

    Args:
        model_checkpoint: Path to the saved model checkpoint.
        model_name: Name of the pretrained model to use for tokenizer.
        batch_size: Batch size for evaluation.
        max_length: Maximum sequence length for tokenization.
    """
    logger.info("Evaluating like my life depended on it")
    logger.info(f"Loading model from: {model_checkpoint}")

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Load model with default config (will be overridden by checkpoint if it contains config)
    model = MyAwesomeModel().to(DEVICE)

    # Load model checkpoint, mapping to CPU first to handle cross-platform compatibility
    checkpoint = torch.load(model_checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    logger.info("Model loaded successfully")

    _, test_set = arxiv_dataset()

    def collate(batch):
        return collate_fn_eval(batch, tokenizer, max_length)

    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=collate,
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
