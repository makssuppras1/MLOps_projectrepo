import pytest
import torch

from pname.model import MyAwesomeModel


def test_model():
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)


def test_error_on_wrong_shape():
    """Test that model raises error on wrong input shape."""
    model = MyAwesomeModel()
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 2, 3))
