import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import hydra
import wandb
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from pname.data import corrupt_mnist
from pname.model import MyAwesomeModel
from pname.profiler import profile_training

# Load environment variables
load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
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


def train(cfg: DictConfig) -> None:
    """Train a model on MNIST."""
    # Configure loguru to save logs to hydra output directory
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    log_file = hydra_output_dir / "train.log"

    # Remove default handler and add custom handlers
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        rotation="100 MB",
        retention="10 days",
        compression="zip"
    )

    logger.info("Training day and night")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize wandb
    wandb_config = {
        "project": os.getenv("WANDB_PROJECT", "corrupt-mnist"),
        "entity": os.getenv("WANDB_ENTITY", None),
        "job_type": "training",
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    # Only initialize wandb if API key is available
    wandb_initialized = False
    if os.getenv("WANDB_API_KEY"):
        wandb.init(**wandb_config)
        wandb_initialized = True
        logger.info("Weights & Biases initialized")

        # If running a sweep, update config with wandb.config values
        if wandb.config:
            logger.info("Updating config with wandb sweep parameters")
            # Update training hyperparameters from wandb.config if they exist
            if "training.batch_size" in wandb.config:
                cfg.training.batch_size = wandb.config["training.batch_size"]
            if "training.optimizer.lr" in wandb.config:
                cfg.training.optimizer.lr = wandb.config["training.optimizer.lr"]
            if "training.epochs" in wandb.config:
                cfg.training.epochs = wandb.config["training.epochs"]

            # Update model hyperparameters from wandb.config if they exist
            if "model.dropout" in wandb.config:
                cfg.model.dropout = wandb.config["model.dropout"]
            if "model.conv1.out_channels" in wandb.config:
                cfg.model.conv1.out_channels = wandb.config["model.conv1.out_channels"]
            if "model.conv2.out_channels" in wandb.config:
                cfg.model.conv2.out_channels = wandb.config["model.conv2.out_channels"]
            if "model.conv3.out_channels" in wandb.config:
                cfg.model.conv3.out_channels = wandb.config["model.conv3.out_channels"]

            # Update wandb config with the full updated config
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    else:
        logger.warning("WANDB_API_KEY not found. Wandb logging disabled.")

    # Set random seed for reproducibility
    set_seed(cfg.training.seed)
    logger.info(f"Random seed set to: {cfg.training.seed}")

    # Initialize model with config
    model = MyAwesomeModel(model_cfg=cfg.model).to(DEVICE)
    # Ensure model is in float32 (required for NNPack on MPS)
    model = model.float()

    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.training.batch_size
    )

    # Instantiate loss function and optimizer from config
    loss_fn = instantiate(cfg.training.loss_fn)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())

    logger.info(f"Model: {model}")
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Loss function: {loss_fn}")
    logger.info(f"Batch size: {cfg.training.batch_size}, Epochs: {cfg.training.epochs}")

    # Log model architecture to wandb
    if wandb_initialized:
        wandb.config.update({
            "model_params": sum(p.numel() for p in model.parameters()),
            "model_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })

    statistics = {"train_loss": [], "train_accuracy": []}

    # Use profiling if enabled
    profile_enabled = cfg.training.get("profile", False)

    if profile_enabled:
        logger.info("Profiling enabled")
        profiler_ctx = profile_training()
        pytorch_profiler = profiler_ctx.__enter__()
    else:
        profiler_ctx = None
        pytorch_profiler = None

    try:
        global_step = 0
        for epoch in range(cfg.training.epochs):
            model.train()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0

            for i, (img, target) in enumerate(train_dataloader):
                # Ensure images are float32 and on correct device
                img = img.float().to(DEVICE)
                target = target.long().to(DEVICE)
                optimizer.zero_grad()
                y_pred = model(img)
                loss = loss_fn(y_pred, target)
                loss.backward()
                optimizer.step()

                # Step PyTorch profiler if enabled
                if pytorch_profiler:
                    pytorch_profiler.step()

                statistics["train_loss"].append(loss.item())
                accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
                statistics["train_accuracy"].append(accuracy)

                epoch_loss += loss.item()
                epoch_accuracy += accuracy
                num_batches += 1

                # Log to wandb at each step
                if wandb_initialized:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/accuracy": accuracy,
                        "train/epoch": epoch,
                        "global_step": global_step,
                    }, step=global_step)

                global_step += 1

                if i % cfg.training.log_interval == 0:
                    logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, accuracy: {accuracy:.4f}")

            # Log epoch-level metrics
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_accuracy = epoch_accuracy / num_batches
            logger.info(f"Epoch {epoch} completed - Avg Loss: {avg_epoch_loss:.4f}, Avg Accuracy: {avg_epoch_accuracy:.4f}")

            if wandb_initialized:
                wandb.log({
                    "train/epoch_loss": avg_epoch_loss,
                    "train/epoch_accuracy": avg_epoch_accuracy,
                }, step=epoch)

            # Log sample images from the first batch of the first epoch
            if epoch == 0 and i == 0 and wandb_initialized:
                # Log a few sample images with predictions
                sample_images = img[:8].cpu()
                sample_targets = target[:8].cpu()
                sample_preds = y_pred[:8].argmax(dim=1).cpu()

                # Create a grid of images
                fig, axes = plt.subplots(2, 4, figsize=(12, 6))
                for idx, ax in enumerate(axes.flat):
                    if idx < len(sample_images):
                        ax.imshow(sample_images[idx].squeeze(), cmap="gray")
                        ax.set_title(f"True: {sample_targets[idx].item()}, Pred: {sample_preds[idx].item()}")
                        ax.axis("off")
                plt.tight_layout()
                wandb.log({"train/sample_predictions": wandb.Image(fig)})
                plt.close(fig)

    finally:
        if profiler_ctx:
            profiler_ctx.__exit__(None, None, None)

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
    train(cfg)


if __name__ == "__main__":
    main()
