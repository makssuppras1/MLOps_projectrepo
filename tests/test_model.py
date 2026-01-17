import pytest
import torch

from pname.model import MyAwesomeModel


def test_model():
    """Test that model forward pass works with text inputs."""
    model = MyAwesomeModel()
    batch_size = 2
    seq_length = 128

    # Create dummy tokenized inputs (input_ids and attention_mask)
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    output = model(input_ids=input_ids, attention_mask=attention_mask)

    # Check output structure
    assert "logits" in output
    assert output["logits"].shape == (batch_size, 5)  # 5 classes for ArXiv categories


def test_error_on_wrong_shape():
    """Test that model raises error on wrong input shape."""
    model = MyAwesomeModel()
    # Wrong shape - missing attention_mask
    with pytest.raises(TypeError):
        model(input_ids=torch.randn(1, 2, 3))
