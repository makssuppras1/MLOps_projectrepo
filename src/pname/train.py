import os
from pathlib import Path
from typing import Tuple

import hydra
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import DistilBertTokenizer

import wandb
from pname.data import arxiv_dataset
from pname.model import MyAwesomeModel

# Load environment variables
load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value to set for all random number generators.
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
    batch: list[Tuple[str, torch.Tensor]], tokenizer: DistilBertTokenizer, max_length: int = 512
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function to tokenize text batches.

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


def train(cfg: DictConfig) -> None:
    """Train a DistilBERT text classifier.

    Args:
        cfg: Hydra configuration dictionary containing model, training, and experiment settings.
    """
    logger.info("Training day and night")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize W&B if API key is available
    wandb_initialized = False
    if os.getenv("WANDB_API_KEY"):
        try:
            hydra_cfg = HydraConfig.get()
            run_name = f"{hydra_cfg.job.name}_{hydra_cfg.job.override_dirname}"
            wandb.init(
                project=cfg.get("wandb_project", "arxiv-classifier"),
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            wandb_initialized = True
            logger.info("W&B initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
    else:
        logger.info("WANDB_API_KEY not found, skipping W&B logging")

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

    def collate(batch):
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

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

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

            epoch_loss += loss.item()
            epoch_accuracy += accuracy
            num_batches += 1

            if i % cfg.training.log_interval == 0:
                logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, accuracy: {accuracy:.4f}")

            # Log to W&B
            if wandb_initialized:
                wandb.log(
                    {
                        "train/step_loss": loss.item(),
                        "train/step_accuracy": accuracy,
                        "epoch": epoch,
                        "iteration": i,
                    }
                )

        # Log epoch-level metrics
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_epoch_accuracy = epoch_accuracy / num_batches if num_batches > 0 else 0.0

        if wandb_initialized:
            wandb.log(
                {
                    "train/epoch_loss": avg_epoch_loss,
                    "train/epoch_accuracy": avg_epoch_accuracy,
                    "epoch": epoch,
                }
            )

        logger.info(f"Epoch {epoch} complete - Avg loss: {avg_epoch_loss:.4f}, Avg accuracy: {avg_epoch_accuracy:.4f}")

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

    # Log plot to wandb
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
            },
        )
        artifact.add_file(str(model_save_path))
        wandb.log_artifact(artifact)
        logger.info("Model logged as wandb artifact")

        # Finish wandb run
        wandb.finish()
        logger.info("Wandb run finished")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training script.

    Args:
        cfg: Hydra configuration dictionary (automatically loaded).
    """
    train(cfg)


if __name__ == "__main__":
    main()
