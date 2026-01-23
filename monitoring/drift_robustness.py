"""Drift robustness testing for text classification models.

Generates drifted versions of validation data and evaluates model performance
across key drift scenarios (vocabulary shift, domain shift) to quantify
sensitivity to data distribution shifts. Simplified to 2 scenarios × 2 severities = 4 test runs.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from pname.data import arxiv_dataset
from pname.model import MyAwesomeModel
from pname.model_tfidf import TFIDFXGBoostModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def apply_vocabulary_shift(texts: List[str], severity: float) -> List[str]:
    """Apply vocabulary shift by replacing words with OOV tokens.

    Args:
        texts: List of input texts.
        severity: Severity level (0.0-1.0), fraction of words to replace with OOV.

    Returns:
        List of texts with vocabulary shift applied.
    """
    oov_token = "<UNK>"
    shifted_texts = []

    for text in texts:
        words = text.split()
        num_replace = max(1, int(len(words) * severity))
        indices_to_replace = random.sample(range(len(words)), min(num_replace, len(words)))

        shifted_words = words.copy()
        for idx in indices_to_replace:
            shifted_words[idx] = oov_token

        shifted_texts.append(" ".join(shifted_words))

    return shifted_texts


def apply_length_shift(texts: List[str], severity: float, direction: str = "shorter") -> List[str]:
    """Apply text length distribution shift.

    Args:
        texts: List of input texts.
        severity: Severity level (0.0-1.0), fraction of text to remove/add.
        direction: "shorter" or "longer".

    Returns:
        List of texts with length shift applied.
    """
    shifted_texts = []

    for text in texts:
        words = text.split()
        if direction == "shorter":
            # Remove words from the end
            num_keep = max(1, int(len(words) * (1 - severity)))
            shifted_texts.append(" ".join(words[:num_keep]))
        else:  # longer
            # Repeat words to make longer
            num_repeat = int(len(words) * severity)
            shifted_texts.append(text + " " + " ".join(words[:num_repeat]))

    return shifted_texts


def apply_domain_shift(texts: List[str], severity: float) -> List[str]:
    """Apply domain shift by adding domain-specific terminology.

    Args:
        texts: List of input texts.
        severity: Severity level (0.0-1.0), fraction of texts to modify.

    Returns:
        List of texts with domain shift applied.
    """
    # Add technical jargon that might not be in training data
    domain_terms = [
        "quantum computing",
        "neural architecture search",
        "federated learning",
        "transformer attention",
        "gradient descent optimization",
        "backpropagation algorithm",
        "convolutional neural network",
        "recurrent neural network",
        "generative adversarial network",
    ]

    shifted_texts = []
    num_modify = max(1, int(len(texts) * severity))
    indices_to_modify = random.sample(range(len(texts)), min(num_modify, len(texts)))

    for idx, text in enumerate(texts):
        if idx in indices_to_modify:
            # Insert domain term at random position
            term = random.choice(domain_terms)
            words = text.split()
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, term)
            shifted_texts.append(" ".join(words))
        else:
            shifted_texts.append(text)

    return shifted_texts


def evaluate_model_pytorch(
    model: MyAwesomeModel,
    tokenizer: Any,
    texts: List[str],
    labels: torch.Tensor,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Evaluate PyTorch model on texts.

    Returns:
        Dictionary with accuracy and macro_f1 metrics.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(DEVICE)
            attention_mask = encoded["attention_mask"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs["preds"].cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    labels_np = labels.cpu().numpy()

    accuracy = accuracy_score(labels_np, all_preds)
    macro_f1 = f1_score(labels_np, all_preds, average="macro")

    return {"accuracy": float(accuracy), "macro_f1": float(macro_f1)}


def evaluate_model_tfidf(
    model: TFIDFXGBoostModel,
    texts: List[str],
    labels: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate TF-IDF model on texts.

    Returns:
        Dictionary with accuracy and macro_f1 metrics.
    """
    preds = model.predict(texts)
    labels_np = labels.cpu().numpy()

    accuracy = accuracy_score(labels_np, preds)
    macro_f1 = f1_score(labels_np, preds, average="macro")

    return {"accuracy": float(accuracy), "macro_f1": float(macro_f1)}


def run_drift_robustness_test(
    model_path: str,
    model_type: str = "pytorch",  # or "tfidf"
    model_name: str = "distilbert-base-uncased",
    val_set_size: Optional[int] = 300,
    severities: List[float] = [0.3, 0.5],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run drift robustness test across multiple scenarios.

    Args:
        model_path: Path to model checkpoint.
        model_type: "pytorch" or "tfidf".
        model_name: Pretrained model name (for PyTorch).
        val_set_size: Number of validation samples to use (None = all).
        severities: List of severity levels to test.
        output_path: Path to save JSON report.

    Returns:
        Dictionary with baseline metrics and scenario results.
    """
    if output_path is None:
        output_path = Path("monitoring/drift_robustness_report.json")

    logger.info("Loading validation dataset...")
    _, val_set, _ = arxiv_dataset(max_categories=5)

    if len(val_set) == 0:
        raise ValueError("Validation set is empty. Cannot run robustness test.")

    # Use subset if specified
    if val_set_size and val_set_size < len(val_set):
        indices = random.sample(range(len(val_set)), val_set_size)
        val_set = torch.utils.data.Subset(val_set, indices)
        logger.info(f"Using subset of {val_set_size} samples")

    # Extract texts and labels
    val_texts = [val_set[i][0] for i in range(len(val_set))]
    val_labels = torch.stack([torch.tensor(val_set[i][1]) for i in range(len(val_set))])

    logger.info(f"Loaded {len(val_texts)} validation samples")

    # Load model
    logger.info(f"Loading {model_type} model from {model_path}...")
    if model_type == "pytorch":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MyAwesomeModel().to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()

        def evaluate_fn(texts, labels):
            return evaluate_model_pytorch(model, tokenizer, texts, labels)

    else:  # tfidf
        model = TFIDFXGBoostModel.load(model_path)

        def evaluate_fn(texts, labels):
            return evaluate_model_tfidf(model, texts, labels)

    # Baseline evaluation (no drift)
    logger.info("Evaluating baseline (no drift)...")
    baseline_metrics = evaluate_fn(val_texts, val_labels)
    logger.info(f"Baseline accuracy: {baseline_metrics['accuracy']:.4f}, F1: {baseline_metrics['macro_f1']:.4f}")

    # Test scenarios (simplified: 2 most important scenarios)
    scenarios = []

    # Scenario 1: Vocabulary shift (OOV handling)
    logger.info("Testing vocabulary shift scenario...")
    for severity in severities:
        drifted_texts = apply_vocabulary_shift(val_texts, severity)
        metrics = evaluate_fn(drifted_texts, val_labels)
        delta = {
            "accuracy": metrics["accuracy"] - baseline_metrics["accuracy"],
            "macro_f1": metrics["macro_f1"] - baseline_metrics["macro_f1"],
        }
        scenarios.append(
            {
                "name": "vocabulary_shift",
                "severity": severity,
                "metrics": metrics,
                "delta": delta,
            }
        )
        logger.info(f"  Severity {severity}: accuracy={metrics['accuracy']:.4f}, delta={delta['accuracy']:.+.4f}")

    # Scenario 2: Domain shift (realistic production scenario)
    logger.info("Testing domain shift scenario...")
    for severity in severities:
        drifted_texts = apply_domain_shift(val_texts, severity)
        metrics = evaluate_fn(drifted_texts, val_labels)
        delta = {
            "accuracy": metrics["accuracy"] - baseline_metrics["accuracy"],
            "macro_f1": metrics["macro_f1"] - baseline_metrics["macro_f1"],
        }
        scenarios.append(
            {
                "name": "domain_shift",
                "severity": severity,
                "metrics": metrics,
                "delta": delta,
            }
        )
        logger.info(f"  Severity {severity}: accuracy={metrics['accuracy']:.4f}, delta={delta['accuracy']:.+.4f}")

    # Generate conclusion
    worst_delta = min(s["delta"]["accuracy"] for s in scenarios)
    worst_scenario = next(s for s in scenarios if s["delta"]["accuracy"] == worst_delta)

    conclusion = (
        f"Model tested across {len(scenarios)} drift scenarios. "
        f"Baseline accuracy: {baseline_metrics['accuracy']:.4f}. "
        f"Worst performance drop: {worst_delta:.4f} accuracy in {worst_scenario['name']} "
        f"(severity {worst_scenario['severity']}). "
        f"Model shows {'high' if abs(worst_delta) > 0.2 else 'moderate' if abs(worst_delta) > 0.1 else 'low'} "
        f"sensitivity to data drift."
    )

    # Compile report
    report = {
        "baseline_metrics": baseline_metrics,
        "scenarios": scenarios,
        "conclusion": conclusion,
        "model_path": model_path,
        "model_type": model_type,
        "val_set_size": len(val_texts),
    }

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Drift robustness report saved to: {output_path}")
    logger.info(f"Conclusion: {conclusion}")

    return report


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def test(
        model_path: str = typer.Argument(..., help="Path to model checkpoint"),
        model_type: str = typer.Option("pytorch", help="Model type: pytorch or tfidf"),
        model_name: str = typer.Option("distilbert-base-uncased", help="Pretrained model name (PyTorch only)"),
        val_set_size: Optional[int] = typer.Option(300, help="Validation set size to use"),
        output_path: str = typer.Option("monitoring/drift_robustness_report.json", help="Output JSON report path"),
    ) -> None:
        """Run drift robustness test."""
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        report = run_drift_robustness_test(
            model_path=model_path,
            model_type=model_type,
            model_name=model_name,
            val_set_size=val_set_size,
            output_path=Path(output_path),
        )
        print(json.dumps(report, indent=2))

    app()
