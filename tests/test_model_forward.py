"""Simple tests for DistilBERT classifier forward pass."""

import torch
from omegaconf import DictConfig

from pname.model import MyAwesomeModel


def test_model_forward():
    """Test forward pass with and without labels."""
    model_cfg = DictConfig({
        'num_labels': 5,
        'model_name': 'distilbert-base-uncased',
        'dropout': 0.1,
    })
    model = MyAwesomeModel(model_cfg=model_cfg)
    model.eval()

    # Create dummy batch: B=2, L=8
    input_ids = torch.randint(100, 1000, (2, 8), dtype=torch.long)
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    labels = torch.randint(0, 5, (2,), dtype=torch.long)

    # Forward pass without labels
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs["logits"].shape == (2, 5)
    assert outputs["probs"].shape == (2, 5)
    assert outputs["preds"].shape == (2,)
    assert "loss" not in outputs

    # Forward pass with labels
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert outputs["logits"].shape == (2, 5)
    assert "loss" in outputs
    assert outputs["loss"].item() >= 0
