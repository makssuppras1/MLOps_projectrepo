"""
Training script for ArXiv paper classification.

⚠️ CRITICAL REMINDER FOR VERTEX AI JOBS:
Before submitting any Vertex AI job, ALWAYS run:
    ./scripts/preflight_check.sh

This checks platform architecture, permissions, image regions, configs, and quotas.
See docs/PRE_FLIGHT_CHECKLIST.md for details.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import hydra
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig

# instantiate removed - using direct optimizer creation for param groups
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

import wandb
from pname.data import arxiv_dataset
from pname.model import MyAwesomeModel

# Load environment variables
load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Global variables for graceful shutdown handling
_model = None
_model_save_path = None
_training_start_time = None


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
    batch: list[Tuple[str, torch.Tensor]], tokenizer, max_length: int = 512
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


def save_model_checkpoint(
    model: torch.nn.Module, save_path: Path, checkpoint_name: str = "checkpoint_preempted.pt"
) -> None:
    """Save model checkpoint, handling both local and GCS paths.

    Args:
        model: The model to save.
        save_path: Base path for saving (directory).
        checkpoint_name: Name of the checkpoint file.
    """
    checkpoint_path = save_path / checkpoint_name
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        # Also try to copy to GCS if AIP_MODEL_DIR is set
        if os.getenv("AIP_MODEL_DIR"):
            try:
                gcs_checkpoint = Path(os.getenv("AIP_MODEL_DIR")) / checkpoint_name
                shutil.copy2(checkpoint_path, gcs_checkpoint)
                logger.info(f"Checkpoint also saved to GCS: {gcs_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint to GCS: {e}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def signal_handler(signum, frame):
    """Handle preemption signals gracefully."""
    global _model, _model_save_path
    logger.warning(f"Received signal {signum}. Attempting to save checkpoint before shutdown...")
    if _model is not None and _model_save_path is not None:
        save_model_checkpoint(_model, _model_save_path.parent, "checkpoint_preempted.pt")
    sys.exit(1)


def train(cfg: DictConfig) -> None:
    """Train a DistilBERT text classifier.

    Args:
        cfg: Hydra configuration dictionary containing model, training, and experiment settings.
    """
    global _model, _model_save_path, _training_start_time

    logger.info("Training day and night")

    # Set up signal handlers for graceful shutdown on preemption
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Log model save location early so it's clear where the model will be saved
    aip_model_dir = os.getenv("AIP_MODEL_DIR")
    if aip_model_dir:
        expected_model_path = Path(aip_model_dir) / "trained_model.pt"
        logger.info("=" * 70)
        logger.info("MODEL SAVE LOCATION (Vertex AI):")
        logger.info(f"  AIP_MODEL_DIR: {aip_model_dir}")
        logger.info(f"  Final model path: {expected_model_path}")
        # Convert /gcs/ path to GCS URI format
        if "/gcs/" in aip_model_dir:
            gcs_path = aip_model_dir.replace("/gcs/", "")
            logger.info(f"  GCS URI: gs://{gcs_path}/trained_model.pt")
        logger.info("=" * 70)
    else:
        fallback_path = (
            Path("/outputs/trained_model.pt") if Path("/outputs").exists() else Path("outputs/trained_model.pt")
        )
        logger.info("=" * 70)
        logger.info("MODEL SAVE LOCATION (Local/Fallback):")
        logger.info(f"  Final model path: {fallback_path.resolve()}")
        logger.info("=" * 70)

    # Merge experiment config into training config if experiment is specified
    if hasattr(cfg, "experiment") and cfg.experiment is not None and hasattr(cfg.experiment, "training"):
        # Temporarily disable struct mode to allow merging new keys (like max_samples, max_runtime_hours)
        OmegaConf.set_struct(cfg.training, False)
        # Merge experiment training config into base training config
        cfg.training = OmegaConf.merge(cfg.training, cfg.experiment.training)
        logger.info("Merged experiment config into training config")

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Determine experiment name for WandB
    experiment_name = "balanced"
    if hasattr(cfg, "experiment") and cfg.experiment is not None:
        # Extract experiment name from config path or override
        experiment_name = str(cfg.experiment).split(".")[-1] if "." in str(cfg.experiment) else str(cfg.experiment)

    # Get subset size for run name
    max_samples = cfg.training.get("max_samples", None)
    subset_suffix = f"-subset-{max_samples}" if max_samples else ""

    # Initialize W&B if API key is available
    wandb_initialized = False
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project = os.getenv("WANDB_PROJECT", "pname-arxiv-classifier")

    if wandb_api_key:
        try:
            hydra_cfg = HydraConfig.get()
            run_name = f"{experiment_name}{subset_suffix}-{hydra_cfg.job.override_dirname}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            wandb_initialized = True
            logger.info(f"W&B initialized successfully: project={wandb_project}, run={run_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
    else:
        logger.info("WANDB_API_KEY not found, skipping W&B logging")

    # Set random seed for reproducibility
    set_seed(cfg.training.seed)
    logger.info(f"Random seed set to: {cfg.training.seed}")

    # Initialize tokenizer (AutoTokenizer works with DistilBERT, TinyBERT, BERT, etc.)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    logger.info(f"Tokenizer loaded: {cfg.model.model_name}")

    # Initialize model with config
    model = MyAwesomeModel(model_cfg=cfg.model).to(DEVICE)
    _model = model  # Store globally for signal handler

    # Start with encoder frozen (will be unfrozen gradually)
    if cfg.model.get("freeze_encoder", True):
        model.freeze_encoder(freeze=True)
        logger.info(f"Model initialized with encoder FROZEN - {model.num_trainable_params():,} trainable parameters")
    else:
        logger.info(f"Model initialized with {model.num_trainable_params():,} trainable parameters")

    # Load dataset (with category reduction if needed)
    max_categories = cfg.model.get("num_labels", 5)
    train_set, val_set, test_set = arxiv_dataset(max_categories=max_categories)
    logger.info(f"Dataset loaded: {len(train_set)} training, {len(val_set)} validation, {len(test_set)} test samples")

    # Optionally use a subset of data for faster training
    max_samples = cfg.training.get("max_samples", None)
    if max_samples is not None:
        if max_samples > len(train_set):
            logger.warning(
                f"max_samples ({max_samples}) is larger than dataset size ({len(train_set)}). Using full dataset."
            )
        elif max_samples < len(train_set):
            # Use generator with seed for reproducible subset selection
            generator = torch.Generator()
            generator.manual_seed(cfg.training.seed)
            indices = torch.randperm(len(train_set), generator=generator)[:max_samples].tolist()
            train_set = torch.utils.data.Subset(train_set, indices)
            logger.info(f"Using subset of {max_samples} samples for faster training")

    logger.info(f"Dataset loaded: {len(train_set)} training samples")

    # Create collate function with tokenizer
    max_length = cfg.training.get("max_length", 512)

    def collate(batch):
        return collate_fn(batch, tokenizer, max_length)

    # Simplified data loading: remove non-essential optimizations for faster startup
    num_workers = cfg.training.get("num_workers", 0)
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        collate_fn=collate,
        shuffle=True,
        num_workers=num_workers,
        # Simplified: removed pin_memory and persistent_workers for simplicity
        # These optimizations are non-essential for small datasets and add overhead
    )

    # Set up differential learning rates with gradual unfreezing
    # Strategy:
    # - Epoch 0: Only classifier head (encoder frozen) with LR 1e-3
    # - Epoch 1+: Unfreeze encoder, use LR 1e-5 for encoder, 1e-3 for classifier

    # Get learning rates from config
    encoder_lr = cfg.training.get("encoder_lr", 1e-5)  # Lower LR for encoder
    classifier_lr = cfg.training.get("classifier_lr", 1e-3)  # Higher LR for classifier

    # Separate parameters for encoder and classifier
    encoder_params = list(model.encoder.parameters())
    classifier_params = list(model.classifier.parameters()) + list(model.dropout.parameters())

    # Create parameter groups for differential learning rates
    # Start with only classifier trainable (encoder frozen)
    param_groups = [{"params": classifier_params, "lr": classifier_lr, "name": "classifier"}]

    # Create optimizer with initial parameter groups (only classifier)
    optimizer_type = cfg.training.optimizer.get("_target_", "torch.optim.Adam")

    # Extract optimizer config (excluding _target_ and lr)
    optimizer_kwargs = {}
    for key, value in cfg.training.optimizer.items():
        if key not in ["_target_", "lr"]:
            optimizer_kwargs[key] = value

    if optimizer_type == "torch.optim.Adam":
        optimizer = torch.optim.Adam(param_groups, **optimizer_kwargs)
    else:
        # For other optimizers, try to instantiate with param groups
        # Fallback to Adam if instantiate fails
        # For other optimizers, use Adam with param groups as fallback
        # (Most optimizers support param groups similarly)
        optimizer = torch.optim.Adam(param_groups, **optimizer_kwargs)

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Batch size: {cfg.training.batch_size}, Epochs: {cfg.training.epochs}")
    logger.info(f"Learning rates - Encoder: {encoder_lr}, Classifier: {classifier_lr}")
    logger.info("Gradual unfreezing: Epoch 0 = classifier only, Epoch 1+ = full model")

    # Budget/time limit tracking
    max_runtime_hours = cfg.training.get("max_runtime_hours", None)
    training_start_time = time.time()  # Track total training time
    _training_start_time = training_start_time  # Store globally
    start_time = time.time()
    if max_runtime_hours:
        max_runtime_seconds = max_runtime_hours * 3600
        logger.info(f"Maximum runtime limit: {max_runtime_hours} hours ({max_runtime_seconds}s)")

    statistics = {"train_loss": [], "train_accuracy": []}
    training_stopped_early = False
    max_steps = cfg.training.get("max_steps", None)
    global_step = 0

    # Wrap training loop in try/except for graceful preemption handling
    try:
        for epoch in range(cfg.training.epochs):
            # Gradual unfreezing: After epoch 0, unfreeze encoder and use differential LRs
            if epoch == 1 and cfg.model.get("freeze_encoder", True):
                logger.info("=" * 70)
                logger.info("GRADUAL UNFREEZING: Unfreezing encoder for epoch 1+")
                logger.info("=" * 70)
                model.freeze_encoder(freeze=False)

                # Recreate optimizer with both encoder and classifier parameters
                encoder_params = list(model.encoder.parameters())
                classifier_params = list(model.classifier.parameters()) + list(model.dropout.parameters())

                param_groups = [
                    {"params": encoder_params, "lr": encoder_lr, "name": "encoder"},
                    {"params": classifier_params, "lr": classifier_lr, "name": "classifier"},
                ]

                # Extract optimizer config (excluding _target_ and lr)
                optimizer_kwargs = {}
                for key, value in cfg.training.optimizer.items():
                    if key not in ["_target_", "lr"]:
                        optimizer_kwargs[key] = value

                optimizer_type = cfg.training.optimizer.get("_target_", "torch.optim.Adam")
                if optimizer_type == "torch.optim.Adam":
                    optimizer = torch.optim.Adam(param_groups, **optimizer_kwargs)
                else:
                    # For other optimizers, fallback to Adam with param groups
                    optimizer = torch.optim.Adam(param_groups, **optimizer_kwargs)

                logger.info("Optimizer recreated with differential learning rates")
                logger.info(f"  Encoder LR: {encoder_lr}, Classifier LR: {classifier_lr}")
                logger.info(f"  Trainable params: {model.num_trainable_params():,}")

            # Check runtime limit before starting epoch
            if max_runtime_hours:
                elapsed = time.time() - start_time
                if elapsed >= max_runtime_seconds:
                    logger.warning(f"Runtime limit reached ({elapsed/3600:.2f} hours). Stopping training early.")
                    logger.info(f"Completed {epoch} out of {cfg.training.epochs} epochs")
                    training_stopped_early = True
                    break

            # Check max_steps before starting epoch
            if max_steps is not None and global_step >= max_steps:
                logger.info(f"Max steps ({max_steps}) reached. Stopping training.")
                training_stopped_early = True
                break

            model.train()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0

            for i, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
                # Check max_steps limit before processing batch
                if max_steps is not None and global_step >= max_steps:
                    logger.info(f"Max steps ({max_steps}) reached. Stopping training.")
                    training_stopped_early = True
                    break

                # Check runtime limit periodically during training
                if max_runtime_hours and (i % 100 == 0):  # Check every 100 iterations
                    elapsed = time.time() - start_time
                    if elapsed >= max_runtime_seconds:
                        logger.warning(f"Runtime limit reached ({elapsed/3600:.2f} hours). Stopping training early.")
                        logger.info(f"Completed {epoch} out of {cfg.training.epochs} epochs")
                        training_stopped_early = True
                        break

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
                global_step += 1

                if i % cfg.training.log_interval == 0:
                    logger.info(
                        f"Epoch {epoch}, iter {i}, step {global_step}, loss: {loss.item():.4f}, accuracy: {accuracy:.4f}"
                    )

                # Log to W&B (less frequently to reduce overhead)
                if wandb_initialized and (global_step % cfg.training.log_interval == 0):
                    wandb.log(
                        {
                            "train/step_loss": loss.item(),
                            "train/step_accuracy": accuracy,
                            "epoch": epoch,
                            "iteration": i,
                            "global_step": global_step,
                        }
                    )

            # Break out of outer loop if training was stopped early
            if training_stopped_early or (max_steps is not None and global_step >= max_steps):
                break

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

            logger.info(
                f"Epoch {epoch} complete - Avg loss: {avg_epoch_loss:.4f}, Avg accuracy: {avg_epoch_accuracy:.4f}"
            )

    except (KeyboardInterrupt, SystemExit, Exception) as e:
        # Handle preemption or other exceptions gracefully
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            logger.warning(f"Training interrupted: {e}")
        else:
            logger.error(f"Training error: {e}", exc_info=True)

        # Try to save checkpoint before exiting
        if model is not None:
            try:
                # Determine checkpoint path
                aip_model_dir = os.getenv("AIP_MODEL_DIR")
                if aip_model_dir:
                    checkpoint_dir = Path(aip_model_dir)
                elif Path("/outputs").exists():
                    checkpoint_dir = Path("/outputs")
                else:
                    checkpoint_dir = Path("outputs")

                save_model_checkpoint(model, checkpoint_dir, "checkpoint_preempted.pt")
                logger.info("Checkpoint saved before exit")
            except Exception as checkpoint_error:
                logger.error(f"Failed to save checkpoint: {checkpoint_error}")

        # Re-raise to exit
        raise

    logger.info("Training complete")

    # Calculate total training time
    total_training_time = time.time() - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)

    # Determine model save path using AIP_MODEL_DIR (Vertex AI) or fallback to local paths
    aip_model_dir = os.getenv("AIP_MODEL_DIR")
    if aip_model_dir:
        # Vertex AI provides AIP_MODEL_DIR pointing to GCS bucket
        model_save_path = Path(aip_model_dir) / "trained_model.pt"
        logger.info(f"Using AIP_MODEL_DIR for model save: {aip_model_dir}")
    else:
        # Fallback: try /outputs (Docker container) or local outputs/
        if Path("/outputs").exists():
            model_save_path = Path("/outputs/trained_model.pt")
        else:
            model_save_path = Path("outputs/trained_model.pt")
        logger.info(f"Using fallback path for model save: {model_save_path}")

    _model_save_path = model_save_path  # Store globally for signal handler

    # Create directory if it doesn't exist
    os.makedirs(model_save_path.parent, exist_ok=True)

    # Save model
    try:
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model saved to: {model_save_path}")

        # If using AIP_MODEL_DIR, the file is already in GCS (via FUSE mount)
        # Otherwise, try to upload to GCS as backup
        if not aip_model_dir:
            # Try to upload to GCS bucket if available
            gcs_regional_uri = "gs://mlops_project_data_bucket1-europe-west1"
            gcs_bucket_uri = "gs://mlops_project_data_bucket1"

            try:
                gcs_model_uri = f"{gcs_regional_uri}/trained_model.pt"
                result = subprocess.run(
                    ["gsutil", "cp", str(model_save_path), gcs_model_uri],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    logger.info(f"Model uploaded to GCS via gsutil: {gcs_model_uri}")
                else:
                    # Fall back to multi-region bucket
                    gcs_model_uri = f"{gcs_bucket_uri}/trained_model.pt"
                    result = subprocess.run(
                        ["gsutil", "cp", str(model_save_path), gcs_model_uri],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if result.returncode == 0:
                        logger.info(f"Model uploaded to GCS via gsutil: {gcs_model_uri}")
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                logger.debug(f"GCS upload skipped: {e}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

    # Verification: Print absolute path and file size
    absolute_path = model_save_path.resolve()
    file_size = absolute_path.stat().st_size
    print("Model saved successfully!")
    print(f"Absolute path: {absolute_path}")
    print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")

    # Print training time and final model path (required output)
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_training_time:.2f} seconds)")
    print(f"Final model path: {absolute_path}")
    print(f"{'='*60}\n")

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
