"""Visualization utilities for model embeddings and predictions."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import DistilBertTokenizer

from pname.data import arxiv_dataset
from pname.model import MyAwesomeModel
from pname.train import collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(
    model_checkpoint: str,
    figure_name: str = "embeddings.png",
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 32,
    max_length: int = 512,
    max_samples: Optional[int] = None,
) -> None:
    """
    Visualize model embeddings using t-SNE dimensionality reduction.

    Loads a trained model, extracts embeddings from the encoder's [CLS] token,
    reduces dimensionality using PCA (if needed) and t-SNE, and creates a scatter
    plot colored by class labels.

    Args:
        model_checkpoint: Path to the saved model checkpoint file.
        figure_name: Name of the output figure file. Defaults to "embeddings.png".
        model_name: Name of the pretrained model for tokenizer. Defaults to "distilbert-base-uncased".
        batch_size: Batch size for processing. Defaults to 32.
        max_length: Maximum sequence length for tokenization. Defaults to 512.
        max_samples: Maximum number of samples to visualize. If None, uses all test samples.

    Raises:
        FileNotFoundError: If model checkpoint or data files are not found.
    """
    logger.info(f"Loading model from: {model_checkpoint}")

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Load model
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model.eval()
    logger.info("Model loaded successfully")

    # Load test dataset
    _, test_set = arxiv_dataset()

    # Limit samples if specified
    if max_samples is not None and max_samples < len(test_set):
        indices = torch.randperm(len(test_set))[:max_samples]
        test_set = torch.utils.data.Subset(test_set, indices.tolist())

    # Create collate function with tokenizer
    def collate(batch):
        return collate_fn(batch, tokenizer, max_length)

    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
    )

    # Extract embeddings from encoder
    embeddings_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []

    with torch.inference_mode():
        for input_ids, attention_mask, targets in test_dataloader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            # Get encoder outputs and extract [CLS] token embeddings
            encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = encoder_outputs.last_hidden_state[:, 0]  # [B, hidden_size]

            embeddings_list.append(cls_embeddings.cpu())
            targets_list.append(targets)

    # Concatenate all embeddings and targets
    embeddings = torch.cat(embeddings_list).numpy()
    targets = torch.cat(targets_list).numpy()

    logger.info(f"Extracted embeddings shape: {embeddings.shape}")

    # Reduce dimensionality if needed
    if embeddings.shape[1] > 500:
        logger.info("Applying PCA to reduce dimensionality...")
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
        logger.info(f"After PCA: {embeddings.shape}")

    # Apply t-SNE
    logger.info("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    logger.info("t-SNE complete")

    # Create visualization
    num_classes = len(np.unique(targets))
    fig, ax = plt.subplots(figsize=(10, 10))

    for class_idx in range(num_classes):
        mask = targets == class_idx
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=f"Class {class_idx}",
            alpha=0.6,
        )

    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_title("Model Embeddings Visualization (t-SNE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save figure
    figure_path = Path(f"reports/figures/{figure_name}")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved to {figure_path}")


if __name__ == "__main__":
    typer.run(visualize)
