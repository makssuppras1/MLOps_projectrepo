import pytest
import torch
from transformers import DistilBertTokenizer

from pname.model import MyAwesomeModel
from pname.train import set_seed, collate_fn


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

    def test_collate_fn_pads_to_same_length(self, tokenizer):
        """Test that collate_fn pads all sequences to the same length."""
        batch = [
            ("Short text", torch.tensor(0)),
            ("This is a much longer piece of text with more words", torch.tensor(1)),
        ]

        input_ids, attention_mask, labels = collate_fn(batch, tokenizer, max_length=512)

        # All sequences should have same length after collation
        assert input_ids.shape[0] == input_ids.shape[0], "All sequences should have the same padded length"
        assert (
            input_ids[0].shape == input_ids[1].shape
        ), f"Sequence lengths don't match after padding: {input_ids[0].shape} vs {input_ids[1].shape}"

    def test_collate_fn_returns_correct_dtypes(self, tokenizer):
        """Test that collate_fn returns tensors with correct data types."""
        batch = [
            ("Sample text one", torch.tensor(0)),
            ("Sample text two", torch.tensor(1)),
        ]

        input_ids, attention_mask, labels = collate_fn(batch, tokenizer, max_length=512)

        assert input_ids.dtype == torch.long, f"Input IDs should be torch.long, got {input_ids.dtype}"
        assert attention_mask.dtype == torch.long, f"Attention mask should be torch.long, got {attention_mask.dtype}"
        assert labels.dtype == torch.long, f"Labels should be torch.long, got {labels.dtype}"

    def test_collate_fn_attention_mask_values(self, tokenizer):
        """Test that attention mask contains only 0s and 1s."""
        batch = [
            ("Text one", torch.tensor(0)),
            ("Text two", torch.tensor(1)),
        ]

        input_ids, attention_mask, labels = collate_fn(batch, tokenizer, max_length=512)

        unique_values = torch.unique(attention_mask)
        assert torch.all(
            (unique_values == 0) | (unique_values == 1)
        ), f"Attention mask should contain only 0s and 1s, got values: {unique_values}"

    def test_collate_fn_respects_max_length(self, tokenizer):
        """Test that collate_fn respects the max_length parameter."""
        max_length = 32
        batch = [
            (
                "This is a long text that will be truncated if the max length is reached during tokenization",
                torch.tensor(0),
            ),
            ("Another long text with many words to test truncation behavior", torch.tensor(1)),
        ]

        input_ids, attention_mask, labels = collate_fn(batch, tokenizer, max_length=max_length)

        assert input_ids.shape[1] <= max_length, f"Sequence length {input_ids.shape[1]} exceeds max_length {max_length}"
        assert (
            attention_mask.shape[1] <= max_length
        ), f"Attention mask length {attention_mask.shape[1]} exceeds max_length {max_length}"

    def test_collate_fn_labels_match_input_order(self, tokenizer):
        """Test that labels are preserved in the same order as input batch."""
        batch = [
            ("Text one", torch.tensor(2)),
            ("Text two", torch.tensor(0)),
            ("Text three", torch.tensor(4)),
        ]

        input_ids, attention_mask, labels = collate_fn(batch, tokenizer, max_length=512)

        expected_labels = torch.tensor([2, 0, 4], dtype=torch.long)
        assert torch.equal(
            labels, expected_labels
        ), f"Labels order not preserved: expected {expected_labels}, got {labels}"


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

    def test_model_parameters_updated_after_training_step(self):
        """Test that model parameters are updated after a training step."""
        model = MyAwesomeModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Store initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        batch_size = 4
        seq_length = 16
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        labels = torch.randint(0, 5, (batch_size,), dtype=torch.long)

        model.train()

        # Training step
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output["loss"].backward()
        optimizer.step()

        # Check that parameters changed
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(initial_params[name], param, atol=1e-6):
                params_changed = True
                break

        assert params_changed, "Model parameters should be updated after training step. No parameters changed."

    def test_gradients_flow_through_complete_model(self):
        """Test that gradients flow through the entire model during backward pass."""
        model = MyAwesomeModel()

        batch_size = 4
        seq_length = 16
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        labels = torch.randint(0, 5, (batch_size,), dtype=torch.long)

        model.train()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output["loss"]
        loss.backward()

        # Check that both encoder and classifier have gradients
        encoder_has_grads = False
        classifier_has_grads = False

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if "encoder" in name:
                    encoder_has_grads = True
                if "classifier" in name:
                    classifier_has_grads = True

        assert encoder_has_grads, "Encoder parameters should have gradients after backward pass"
        assert classifier_has_grads, "Classifier parameters should have gradients after backward pass"
