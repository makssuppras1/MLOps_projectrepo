from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import DistilBertTokenizer

from pname.data import arxiv_dataset
from pname.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def collate_fn_vis(
    batch: list[tuple[str, torch.Tensor]],
    tokenizer: DistilBertTokenizer,
    max_length: int = 512
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function to tokenize text batches for visualization.

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


def visualize(
    model_checkpoint: str = typer.Argument(..., help="Path to model checkpoint file"),
    figure_name: str = typer.Option("embeddings.png", help="Name of the output figure file"),
    model_name: str = typer.Option("distilbert-base-uncased", help="Pretrained model name"),
    batch_size: int = typer.Option(32, help="Batch size for processing"),
    max_length: int = typer.Option(512, help="Maximum sequence length"),
    max_samples: int = typer.Option(1000, help="Maximum number of samples to visualize"),
) -> None:
    """Visualize model embeddings using t-SNE.

    Extracts embeddings from the model's encoder and visualizes them in 2D using t-SNE.

    Args:
        model_checkpoint: Path to the saved model checkpoint.
        figure_name: Name of the output figure file (saved in reports/figures/).
        model_name: Name of the pretrained model to use for tokenizer.
        batch_size: Batch size for processing embeddings.
        max_length: Maximum sequence length for tokenization.
        max_samples: Maximum number of samples to visualize (for faster processing).
    """
    logger.info(f"Loading model from: {model_checkpoint}")

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Load model
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model.eval()

    # Get encoder outputs (before classifier)
    def get_embeddings(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from the encoder."""
        with torch.no_grad():
            outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token representation
            return outputs.last_hidden_state[:, 0]  # [B, hidden_size]

    # Load test dataset
    _, test_set = arxiv_dataset()

    # Limit samples if needed
    if len(test_set) > max_samples:
        indices = torch.randperm(len(test_set))[:max_samples]
        test_set = torch.utils.data.Subset(test_set, indices.tolist())
        logger.info(f"Using {max_samples} samples for visualization")

    def collate(batch):
        return collate_fn_vis(batch, tokenizer, max_length)

    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=collate,
    )

    embeddings, targets = [], []
    logger.info("Extracting embeddings...")
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(test_dataloader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            # Get embeddings from encoder
            batch_embeddings = get_embeddings(input_ids, attention_mask)
            embeddings.append(batch_embeddings.cpu())
            targets.append(labels.cpu())

            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Processed {batch_idx + 1}/{len(test_dataloader)} batches")

    # Concatenate and convert to numpy
    embeddings = torch.cat(embeddings).numpy()
    targets = torch.cat(targets).numpy()

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Reduce dimensionality if needed
    if embeddings.shape[1] > 500:
        logger.info("Reducing dimensionality with PCA...")
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)

    logger.info("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Get unique labels
    unique_labels = sorted(set(targets.tolist()))
    num_classes = len(unique_labels)

    logger.info(f"Visualizing {num_classes} classes")

    # Create figure
    plt.figure(figsize=(12, 10))
    for label in unique_labels:
        mask = targets == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f"Class {label}", alpha=0.6)
    plt.legend()
    plt.title("t-SNE Visualization of Model Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Save figure
    figure_path = Path("reports/figures") / figure_name
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figure_path)
    logger.info(f"Figure saved to {figure_path}")


if __name__ == "__main__":
    typer.run(visualize)
