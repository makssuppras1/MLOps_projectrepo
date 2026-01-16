"""Pure PyTorch implementations of classification metrics."""

import torch


def accuracy(preds: torch.LongTensor, targets: torch.LongTensor) -> float:
    """
    Compute top-1 accuracy.

    Args:
        preds: Predicted class indices of shape [B].
        targets: True class indices of shape [B].

    Returns:
        Accuracy score as a float between 0 and 1.
    """
    assert preds.shape == targets.shape, "preds and targets must have the same shape"
    correct = (preds == targets).sum().item()
    total = preds.numel()
    return correct / total if total > 0 else 0.0


def macro_f1(
    preds: torch.LongTensor,
    targets: torch.LongTensor,
    num_classes: int,
) -> float:
    """
    Compute macro-averaged F1 score.

    Args:
        preds: Predicted class indices of shape [B].
        targets: True class indices of shape [B].
        num_classes: Number of classes.

    Returns:
        Macro F1 score as a float between 0 and 1.
    """
    assert preds.shape == targets.shape, "preds and targets must have the same shape"

    f1_scores = []

    for class_idx in range(num_classes):
        # True positives: predicted as class_idx and actually class_idx
        tp = ((preds == class_idx) & (targets == class_idx)).sum().item()

        # False positives: predicted as class_idx but not actually class_idx
        fp = ((preds == class_idx) & (targets != class_idx)).sum().item()

        # False negatives: not predicted as class_idx but actually class_idx
        fn = ((preds != class_idx) & (targets == class_idx)).sum().item()

        # Compute precision and recall with division-by-zero protection
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Compute F1 score with division-by-zero protection
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_scores.append(f1)

    # Macro average: mean of per-class F1 scores
    macro_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return macro_f1_score
