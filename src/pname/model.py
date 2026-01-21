from typing import Dict, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from torch import nn
from transformers import AutoModel, BertModel, DistilBertModel


class MyAwesomeModel(nn.Module):
    """Transformer-based multi-class text classifier.

    Supports DistilBERT, TinyBERT, and other BERT-based models.
    """

    def __init__(self, model_cfg: DictConfig = None) -> None:
        super().__init__()
        if model_cfg is None:
            # Default values for backward compatibility
            model_cfg = DictConfig(
                {
                    "num_labels": 5,
                    "model_name": "distilbert-base-uncased",
                    "dropout": 0.1,
                    "freeze_encoder": False,
                }
            )

        self.num_labels = model_cfg.num_labels
        self.model_name = model_cfg.model_name

        # Load pretrained encoder (supports DistilBERT, TinyBERT, BERT, etc.)
        # Use AutoModel for flexibility with different architectures
        try:
            # Try AutoModel first (works with most models)
            self.encoder = AutoModel.from_pretrained(model_cfg.model_name)
        except Exception:
            # Fallback to specific models if needed
            if "distilbert" in model_cfg.model_name.lower():
                self.encoder = DistilBertModel.from_pretrained(model_cfg.model_name)
            elif "tinybert" in model_cfg.model_name.lower() or "bert" in model_cfg.model_name.lower():
                self.encoder = BertModel.from_pretrained(model_cfg.model_name)
            else:
                self.encoder = AutoModel.from_pretrained(model_cfg.model_name)

        # Get hidden size (different models use different attribute names)
        if hasattr(self.encoder.config, "dim"):
            hidden_size = self.encoder.config.dim  # DistilBERT
        elif hasattr(self.encoder.config, "hidden_size"):
            hidden_size = self.encoder.config.hidden_size  # BERT, TinyBERT
        else:
            # Fallback: try to infer from first layer
            hidden_size = self.encoder.config.hidden_size if hasattr(self.encoder.config, "hidden_size") else 768

        # Classification head
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.classifier = nn.Linear(hidden_size, model_cfg.num_labels)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Freeze encoder if specified
        if model_cfg.get("freeze_encoder", False):
            self.freeze_encoder(freeze=True)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs tensor of shape [B, L].
            attention_mask: Attention mask tensor of shape [B, L].
            labels: Optional class labels tensor of shape [B].
            return_dict: If True, return dict; if False, return tuple.

        Returns:
            If return_dict=True: dict with 'logits', 'loss' (if labels provided),
                'probs', and 'preds' keys.
            If return_dict=False: tuple of (logits, loss) or (logits,) if no labels.
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0]  # [B, hidden_size]

        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [B, num_labels]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        # Compute probabilities and predictions
        probs = torch.softmax(logits, dim=-1)  # [B, num_labels]
        preds = torch.argmax(logits, dim=-1)  # [B]

        if not return_dict:
            if loss is not None:
                return (logits, loss)
            return (logits,)

        output_dict: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probs": probs,
            "preds": preds,
        }
        if loss is not None:
            output_dict["loss"] = loss

        return output_dict

    def freeze_encoder(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze encoder parameters.

        Args:
            freeze: If True, freeze encoder; if False, unfreeze.
        """
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def num_trainable_params(self) -> int:
        """
        Count the number of trainable parameters in the model.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of trainable parameters: {model.num_trainable_params()}")

    # Create dummy batch: B=2, L=8
    batch_size = 2
    seq_length = 8
    dummy_input_ids = torch.randint(100, 1000, (batch_size, seq_length), dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    output = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Output probs shape: {output['probs'].shape}")
    print(f"Output preds shape: {output['preds'].shape}")
