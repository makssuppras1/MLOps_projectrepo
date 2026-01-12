import pytest
import torch
from torch.utils.data import TensorDataset

from pname.model import MyAwesomeModel


@pytest.mark.parametrize("batch_size", [16, 32, 64])
def test_training(batch_size: int) -> None:
    """Test training with different batch sizes."""
    dummy_images = torch.randn(8, 1, 28, 28)
    dummy_targets = torch.randint(0, 10, (8,))
    dataloader = torch.utils.data.DataLoader(
        TensorDataset(dummy_images, dummy_targets), batch_size=batch_size
    )
    
    model = MyAwesomeModel()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for img, target in dataloader:
        optimizer.zero_grad()
        y_pred = model(img.float())
        assert y_pred.shape == (len(target), 10)
        loss = loss_fn(y_pred, target.long())
        loss.backward()
        optimizer.step()
