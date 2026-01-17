import pytest
import torch
from torch.utils.data import Dataset

from pname.model import MyAwesomeModel


class DummyTextDataset(Dataset):
    """Dummy dataset for testing text model."""

    def __init__(self, num_samples=8, seq_length=128, num_classes=5):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.input_ids = torch.randint(0, 1000, (num_samples, seq_length))
        self.attention_mask = torch.ones(num_samples, seq_length, dtype=torch.long)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.labels[idx],
        )


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_training(batch_size: int) -> None:
    """Test training with different batch sizes."""
    dataset = DummyTextDataset(num_samples=8)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    model = MyAwesomeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for input_ids, attention_mask, target in dataloader:
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=target.long())

        # Check output structure
        assert "logits" in output
        assert output["logits"].shape == (len(target), 5)  # 5 classes
        assert "loss" in output

        loss = output["loss"]
        loss.backward()
        optimizer.step()
