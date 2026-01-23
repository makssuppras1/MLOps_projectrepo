import torch

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
