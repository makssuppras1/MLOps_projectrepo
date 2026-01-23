import torch

from pname.model import MyAwesomeModel
from pname.train import set_seed


class TestSetSeed:
    """Test suite for set_seed function."""

    def test_set_seed_produces_reproducible_model_outputs(self):
        """Test that set_seed produces identical model outputs for reproducible training."""
        batch_size = 2
        seq_length = 8
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        # First run with seed 42
        set_seed(42)
        model1 = MyAwesomeModel()
        model1.eval()
        with torch.no_grad():
            output1 = model1(input_ids=input_ids, attention_mask=attention_mask)

        # Second run with seed 42
        set_seed(42)
        model2 = MyAwesomeModel()
        model2.eval()
        with torch.no_grad():
            output2 = model2(input_ids=input_ids, attention_mask=attention_mask)

        assert torch.allclose(
            output1["logits"], output2["logits"], atol=1e-6
        ), "Different seeds produced different model outputs. Reproducibility test failed."


class TestTrainingIntegration:
    """Integration tests for training pipeline."""

    def test_training_step_reduces_loss(self):
        """Test that loss decreases over multiple training steps (accounting for dropout variance)."""
        set_seed(42)
        model = MyAwesomeModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        batch_size = 4
        seq_length = 16
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        labels = torch.randint(0, 5, (batch_size,), dtype=torch.long)

        model.train()
        losses = []

        # Run multiple training steps to see overall trend
        for step in range(10):
            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output["loss"].item()
            losses.append(loss)
            output["loss"].backward()
            optimizer.step()

        # Compare average loss from first 3 steps vs last 3 steps
        avg_first_steps = sum(losses[:3]) / 3
        avg_last_steps = sum(losses[-3:]) / 3

        assert (
            avg_last_steps < avg_first_steps
        ), f"Average loss should decrease over training steps. First 3 steps avg: {avg_first_steps:.6f}, Last 3 steps avg: {avg_last_steps:.6f}. Losses: {[f'{loss:.6f}' for loss in losses]}"
