import pytest
import torch
from omegaconf import DictConfig

from pname.model import MyAwesomeModel


class TestMyAwesomeModel:
    """Test suite for MyAwesomeModel class."""

    def test_model_initialization_default_config(self):
        """Test model initialization with default configuration."""
        model = MyAwesomeModel()

        assert model.num_labels == 5, f"Expected 5 labels by default, got {model.num_labels}"
        assert (
            model.model_name == "distilbert-base-uncased"
        ), f"Expected 'distilbert-base-uncased' model name, got {model.model_name}"
        assert isinstance(model.encoder, torch.nn.Module), "Encoder should be a torch.nn.Module"
        assert isinstance(model.classifier, torch.nn.Linear), "Classifier should be a Linear layer"
        assert isinstance(model.criterion, torch.nn.CrossEntropyLoss), "Loss criterion should be CrossEntropyLoss"

    def test_model_initialization_custom_config(self):
        """Test model initialization with custom configuration."""
        config = DictConfig(
            {
                "num_labels": 10,
                "model_name": "distilbert-base-uncased",
                "dropout": 0.2,
                "freeze_encoder": False,
            }
        )

        model = MyAwesomeModel(config)

        assert model.num_labels == 10, f"Custom config num_labels not respected: expected 10, got {model.num_labels}"

    def test_model_forward_pass_without_labels(self):
        """Test forward pass without labels returns dict with expected keys."""
        model = MyAwesomeModel()

        batch_size = 2
        seq_length = 8
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        assert isinstance(output, dict), f"Expected dict output with return_dict=True, got {type(output)}"
        assert "logits" in output, "Output missing 'logits' key"
        assert "probs" in output, "Output missing 'probs' key"
        assert "preds" in output, "Output missing 'preds' key"
        assert "loss" not in output, "Loss should not be in output when labels are not provided"

    def test_model_forward_pass_with_labels(self):
        """Test forward pass with labels returns dict including loss."""
        model = MyAwesomeModel()

        batch_size = 2
        seq_length = 8
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        labels = torch.tensor([0, 1], dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

        assert isinstance(output, dict), f"Expected dict output, got {type(output)}"
        assert "logits" in output, "Output missing 'logits' key when labels provided"
        assert "probs" in output, "Output missing 'probs' key when labels provided"
        assert "preds" in output, "Output missing 'preds' key when labels provided"
        assert "loss" in output, "Loss should be in output when labels are provided"
        assert output["loss"].item() > 0, f"Loss should be positive, got {output['loss'].item()}"

    def test_model_forward_pass_return_tuple(self):
        """Test forward pass with return_dict=False returns tuple."""
        model = MyAwesomeModel()

        batch_size = 2
        seq_length = 8
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        assert isinstance(output, tuple), f"Expected tuple with return_dict=False, got {type(output)}"
        assert len(output) == 1, f"Expected tuple of length 1 (no labels), got length {len(output)}"
        assert output[0].shape == (batch_size, 5), f"Expected logits shape ({batch_size}, 5), got {output[0].shape}"

    def test_model_forward_pass_return_tuple_with_labels(self):
        """Test forward pass returns tuple with loss when labels provided."""
        model = MyAwesomeModel()

        batch_size = 2
        seq_length = 8
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        labels = torch.tensor([0, 1], dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)

        assert isinstance(output, tuple), f"Expected tuple with return_dict=False, got {type(output)}"
        assert len(output) == 2, f"Expected tuple of length 2 (logits, loss), got length {len(output)}"
        logits, loss = output
        assert logits.shape == (batch_size, 5), f"Expected logits shape ({batch_size}, 5), got {logits.shape}"
        assert loss.item() > 0, f"Expected positive loss value, got {loss.item()}"

    def test_model_output_shapes(self):
        """Test that output tensors have correct shapes."""
        model = MyAwesomeModel()

        batch_size = 4
        seq_length = 16
        num_labels = 5

        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        labels = torch.randint(0, num_labels, (batch_size,), dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        assert output["logits"].shape == (
            batch_size,
            num_labels,
        ), f"Logits shape mismatch: expected ({batch_size}, {num_labels}), got {output['logits'].shape}"
        assert output["probs"].shape == (
            batch_size,
            num_labels,
        ), f"Probs shape mismatch: expected ({batch_size}, {num_labels}), got {output['probs'].shape}"
        assert output["preds"].shape == (
            batch_size,
        ), f"Preds shape mismatch: expected ({batch_size},), got {output['preds'].shape}"
        assert output["loss"].shape == (), f"Loss shape should be scalar (), got {output['loss'].shape}"

    def test_model_probabilities_sum_to_one(self):
        """Test that softmax probabilities sum to 1."""
        model = MyAwesomeModel()

        batch_size = 2
        seq_length = 8
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask)

        probs_sum = output["probs"].sum(dim=-1)
        assert torch.allclose(
            probs_sum, torch.ones(batch_size), atol=1e-6
        ), f"Probabilities do not sum to 1 across class dimension. Got sums: {probs_sum}"

    def test_model_predictions_in_range(self):
        """Test that predictions are within valid label range."""
        model = MyAwesomeModel()

        batch_size = 5
        seq_length = 10
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask)

        assert torch.all(output["preds"] >= 0), f"Predictions contain values < 0. Min: {output['preds'].min()}"
        assert torch.all(
            output["preds"] < model.num_labels
        ), f"Predictions exceed num_labels ({model.num_labels}). Max: {output['preds'].max()}"

    def test_model_freeze_encoder(self):
        """Test encoder freezing functionality."""
        model = MyAwesomeModel()

        # Get initial trainable params
        trainable_before = model.num_trainable_params()

        # Freeze encoder
        model.freeze_encoder(freeze=True)
        trainable_after_freeze = model.num_trainable_params()

        # Trainable params should decrease
        assert (
            trainable_after_freeze < trainable_before
        ), f"Freezing encoder should reduce trainable params. Before: {trainable_before}, After: {trainable_after_freeze}"

        # Encoder params should have requires_grad=False
        for param in model.encoder.parameters():
            assert param.requires_grad is False, "Encoder parameters should have requires_grad=False after freezing"

        # Classifier should still be trainable
        for param in model.classifier.parameters():
            assert param.requires_grad is True, "Classifier parameters should remain trainable when encoder is frozen"

    def test_model_unfreeze_encoder(self):
        """Test encoder unfreezing functionality."""
        model = MyAwesomeModel()

        # Freeze then unfreeze
        model.freeze_encoder(freeze=True)
        model.freeze_encoder(freeze=False)

        # All params should be trainable again
        for param in model.encoder.parameters():
            assert param.requires_grad is True, "Encoder parameters should have requires_grad=True after unfreezing"

    def test_model_num_trainable_params(self):
        """Test trainable parameter counting."""
        config = DictConfig(
            {
                "num_labels": 10,
                "model_name": "distilbert-base-uncased",
                "dropout": 0.1,
                "freeze_encoder": False,
            }
        )

        model = MyAwesomeModel(config)

        trainable_params = model.num_trainable_params()
        total_params = sum(p.numel() for p in model.parameters())

        # All params should be trainable
        assert (
            trainable_params == total_params
        ), f"With freeze_encoder=False, all params should be trainable. Trainable: {trainable_params}, Total: {total_params}"
        assert trainable_params > 0, f"Model should have trainable parameters, got {trainable_params}"

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = MyAwesomeModel()

        batch_size = 2
        seq_length = 8
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        labels = torch.tensor([0, 1], dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output["loss"]

        # Backward pass
        loss.backward()

        # Check that gradients exist
        params_with_grads = 0
        params_without_grads = 0
        for param in model.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    params_with_grads += 1
                else:
                    params_without_grads += 1

        assert params_with_grads > 0, "No trainable parameters received gradients during backward pass"
        assert (
            params_without_grads == 0
        ), f"Some trainable parameters did not receive gradients: {params_without_grads} params without grads"

    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
    def test_model_different_batch_sizes(self, batch_size):
        """Test model with different batch sizes."""
        model = MyAwesomeModel()
        seq_length = 10

        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask)

        assert (
            output["logits"].shape == (batch_size, 5)
        ), f"Logits shape incorrect for batch_size={batch_size}: expected ({batch_size}, 5), got {output['logits'].shape}"
        assert output["preds"].shape == (
            batch_size,
        ), f"Preds shape incorrect for batch_size={batch_size}: expected ({batch_size},), got {output['preds'].shape}"

    @pytest.mark.parametrize("seq_length", [4, 8, 16, 32, 64, 128])
    def test_model_different_sequence_lengths(self, seq_length):
        """Test model with different sequence lengths."""
        model = MyAwesomeModel()
        batch_size = 2

        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask)

        assert (
            output["logits"].shape == (batch_size, 5)
        ), f"Logits shape incorrect for seq_length={seq_length}: expected ({batch_size}, 5), got {output['logits'].shape}"

    def test_model_eval_mode(self):
        """Test model in evaluation mode (no dropout)."""
        model = MyAwesomeModel()
        model.eval()

        batch_size = 2
        seq_length = 8
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        # Run twice and check outputs are identical (no dropout variance)
        with torch.no_grad():
            output1 = model(input_ids=input_ids, attention_mask=attention_mask)
            output2 = model(input_ids=input_ids, attention_mask=attention_mask)

        assert torch.allclose(
            output1["logits"], output2["logits"], atol=1e-6
        ), "Model outputs differ in eval mode (dropout should be disabled). This indicates dropout is still active."

    def test_model_invalid_label_shape(self):
        """Test that model handles invalid label shapes."""
        model = MyAwesomeModel()

        batch_size = 2
        seq_length = 8
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        # Wrong shape: should be (batch_size,) not (batch_size, 1)
        labels = torch.tensor([[0], [1]], dtype=torch.long)

        with pytest.raises((ValueError, RuntimeError)):
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # Trigger the loss computation
            _ = output["loss"]

    def test_model_no_grad_context(self):
        """Test model works in no_grad context."""
        model = MyAwesomeModel()
        model.eval()

        batch_size = 2
        seq_length = 8
        input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        assert (
            output["logits"].requires_grad is False
        ), "Output logits should not require grad in torch.no_grad() context"
