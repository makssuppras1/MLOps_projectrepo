import pytest
import torch
from transformers import DistilBertTokenizer

from pname.model import MyAwesomeModel
from pname.train import collate_fn, set_seed


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

    def test_set_seed_makes_training_deterministic(self):
        """Test that set_seed makes a training step deterministic."""
        batch_size = 2
        seq_length = 8
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        labels = torch.tensor([0, 1], dtype=torch.long)

        # First training step with seed 42
        set_seed(42)
        model1 = MyAwesomeModel()
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
        output1 = model1(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss1 = output1["loss"]
        loss1.backward()
        initial_weight_1 = model1.classifier.weight.data.clone()
        optimizer1.step()

        # Second training step with seed 42
        set_seed(42)
        model2 = MyAwesomeModel()
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        output2 = model2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss2 = output2["loss"]
        loss2.backward()
        initial_weight_2 = model2.classifier.weight.data.clone()
        optimizer2.step()

        assert torch.allclose(
            loss1, loss2, atol=1e-6
        ), f"Loss values differ between runs: {loss1.item()} vs {loss2.item()}. Seed not working correctly."
        assert torch.allclose(
            initial_weight_1, initial_weight_2, atol=1e-6
        ), "Initial weights differ between seeded runs. Reproducibility failed."


class TestCollateFn:
    """Test suite for collate_fn function."""

    @pytest.fixture
    def tokenizer(self):
        """Fixture to load DistilBERT tokenizer."""
        return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def test_collate_fn_returns_correct_shapes(self, tokenizer):
        """Test that collate_fn returns tensors with correct shapes."""
        batch_size = 4
        batch = [
            ("This is a sample text", torch.tensor(0)),
            ("Another example here", torch.tensor(1)),
            ("More test data", torch.tensor(2)),
            ("Final sample text", torch.tensor(1)),
        ]

        input_ids, attention_mask, labels = collate_fn(batch, tokenizer, max_length=512)

        assert (
            input_ids.shape[0] == batch_size
        ), f"Input IDs batch dimension incorrect: expected {batch_size}, got {input_ids.shape[0]}"
        assert (
            attention_mask.shape[0] == batch_size
        ), f"Attention mask batch dimension incorrect: expected {batch_size}, got {attention_mask.shape[0]}"
        assert labels.shape == (batch_size,), f"Labels shape incorrect: expected ({batch_size},), got {labels.shape}"
        assert (
            input_ids.shape == attention_mask.shape
        ), f"Input IDs and attention mask shapes don't match: {input_ids.shape} vs {attention_mask.shape}"


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
