import logging
import matplotlib.pyplot as plt
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import DistilBertTokenizer

from pname.data import arxiv_dataset
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


def collate_fn(batch, tokenizer: DistilBertTokenizer, max_length: int = 512):
    """Collate function to tokenize text batches."""
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
    """Train a DistilBERT text classifier."""
    log.info("Training day and night")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed for reproducibility
    set_seed(cfg.training.seed)
    log.info(f"Random seed set to: {cfg.training.seed}")

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(cfg.model.model_name)
    log.info(f"Tokenizer loaded: {cfg.model.model_name}")

    # Initialize model with config
    model = MyAwesomeModel(model_cfg=cfg.model).to(DEVICE)
    log.info(f"Model initialized with {model.num_trainable_params():,} trainable parameters")

    # Load dataset
    train_set, _ = arxiv_dataset()
    log.info(f"Dataset loaded: {len(train_set)} training samples")

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

    log.info(f"Optimizer: {optimizer}")
    log.info(f"Batch size: {cfg.training.batch_size}, Epochs: {cfg.training.epochs}")

    statistics = {"train_loss": [], "train_accuracy": []}
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
