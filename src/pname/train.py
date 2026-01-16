import logging
import matplotlib.pyplot as plt
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pname.data import corrupt_mnist
from pname.model import MyAwesomeModel

log = logging.getLogger(__name__)

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
    log.info("Training day and night")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed for reproducibility
    set_seed(cfg.training.seed)
    log.info(f"Random seed set to: {cfg.training.seed}")

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

    log.info(f"Model: {model}")
    log.info(f"Optimizer: {optimizer}")
    log.info(f"Loss function: {loss_fn}")
    log.info(f"Batch size: {cfg.training.batch_size}, Epochs: {cfg.training.epochs}")

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(cfg.training.epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            # Ensure images are float32 and on correct device
            img = img.float().to(DEVICE)
            target = target.long().to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % cfg.training.log_interval == 0:
                log.info(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, accuracy: {accuracy:.4f}")

    log.info("Training complete")

    # Save model
    torch.save(model.state_dict(), cfg.training.model_save_path)
    log.info(f"Model saved to {cfg.training.model_save_path}")

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
    fig.savefig(cfg.training.figure_save_path)
    log.info(f"Training statistics saved to {cfg.training.figure_save_path}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
