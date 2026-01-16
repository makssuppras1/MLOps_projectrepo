"""Training script for DistilBERT-based text classification model."""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import hydra
import wandb
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import DistilBertTokenizer

from pname.data import arxiv_dataset
from pname.model import MyAwesomeModel

# Load environment variables
load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all random number generators.

    Sets seeds for PyTorch (CPU, CUDA, MPS), Python's random module, and NumPy
    to ensure reproducible results across runs.

    Args:
        seed: Integer seed value to use for all random number generators.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        # MPS doesn't have a direct seed function, but we can set the generator
        torch.mps.manual_seed(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def collate_fn(
    batch: List[Tuple[str, torch.Tensor]],
    tokenizer: DistilBertTokenizer,
    max_length: int = 512
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to tokenize text batches for DataLoader.

    Takes a batch of (text, label) tuples and tokenizes the texts using the
    provided tokenizer. Returns tokenized input_ids, attention_mask, and labels
    as tensors ready for model input.

    Args:
        batch: List of (text, label) tuples from the dataset.
        tokenizer: DistilBERT tokenizer instance for encoding text.
        max_length: Maximum sequence length for tokenization. Defaults to 512.

    Returns:
        Tuple containing:
            - input_ids: Token IDs tensor of shape [B, L]
            - attention_mask: Attention mask tensor of shape [B, L]
            - labels: Label tensor of shape [B]
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


def train(cfg: DictConfig) -> None:
    """
    Train a DistilBERT-based text classifier on ArXiv dataset.

    This function handles the complete training pipeline:
    - Initializes tokenizer and model
    - Loads and prepares the dataset
    - Sets up optimizer from config
    - Trains the model for specified epochs
    - Saves model checkpoint and training statistics
    - Logs results to wandb if initialized

    Args:
        cfg: Hydra configuration object containing model, training, and other parameters.
    """
    logger.info("Training day and night")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed for reproducibility
    set_seed(cfg.training.seed)
    logger.info(f"Random seed set to: {cfg.training.seed}")

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(cfg.model.model_name)
    logger.info(f"Tokenizer loaded: {cfg.model.model_name}")

    # Initialize model with config
    model = MyAwesomeModel(model_cfg=cfg.model).to(DEVICE)
    logger.info(f"Model initialized with {model.num_trainable_params():,} trainable parameters")

    # Load dataset
    train_set, _ = arxiv_dataset()
    logger.info(f"Dataset loaded: {len(train_set)} training samples")

    # Create collate function with tokenizer
    max_length = cfg.training.get("max_length", 512)

    def collate(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Wrapper collate function that includes tokenizer."""
        return collate_fn(batch, tokenizer, max_length)

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        collate_fn=collate,
        shuffle=True,
    )

    # Instantiate optimizer from config (loss is computed in model forward)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Batch size: {cfg.training.batch_size}, Epochs: {cfg.training.epochs}")

    # Track training statistics
    statistics: Dict[str, List[float]] = {"train_loss": [], "train_accuracy": []}
    for epoch in range(cfg.training.epochs):
        model.train()
        for i, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
            # Move to device
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass (model computes loss internally when labels provided)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            # Compute accuracy
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % cfg.training.log_interval == 0:
                logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, accuracy: {accuracy:.4f}")

    logger.info("Training complete")

    # Save model
    model_save_path = Path(cfg.training.model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    # Save training statistics plot
    fig, axs = plt.subplots(1, 2, figsize=tuple(cfg.training.figure_size))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Accuracy")
    plt.tight_layout()

    figure_save_path = Path(cfg.training.figure_save_path)
    figure_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_save_path)
    logger.info(f"Training statistics saved to {figure_save_path}")

    # Check if wandb is initialized and log plot
    wandb_initialized = wandb.run is not None
    if wandb_initialized:
        wandb.log({"train/training_statistics": wandb.Image(fig)})
        plt.close(fig)

    # Log model as artifact
    if wandb_initialized:
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description=f"Trained model from run {wandb.run.id}",
            metadata={
                "epochs": cfg.training.epochs,
                "batch_size": cfg.training.batch_size,
                "final_loss": statistics["train_loss"][-1] if statistics["train_loss"] else None,
                "final_accuracy": statistics["train_accuracy"][-1] if statistics["train_accuracy"] else None,
            }
        )
        artifact.add_file(str(model_save_path))
        wandb.log_artifact(artifact)
        logger.info("Model logged as wandb artifact")

        # Finish wandb run
        wandb.finish()
        logger.info("Wandb run finished")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training script.

    Uses Hydra to load configuration and delegates to train() function.

    Args:
        cfg: Hydra configuration object loaded from config files.
    """
    train(cfg)


if __name__ == "__main__":
    main()
