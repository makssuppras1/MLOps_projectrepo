"""Training script for TF-IDF + XGBoost model.

Fast, explainable, and perfect for MLOps with hyperparameter tuning.
"""

import os
import time
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import wandb
from pname.data import arxiv_dataset
from pname.model_tfidf import TFIDFXGBoostModel

# Load environment variables
load_dotenv()

DEVICE = "cpu"  # XGBoost runs on CPU


def train(cfg: DictConfig) -> None:
    """Train a TF-IDF + XGBoost text classifier.

    Args:
        cfg: Hydra configuration dictionary containing model and training settings.
    """
    logger.info("Training TF-IDF + XGBoost model")

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Determine experiment name for WandB
    experiment_name = "tfidf_xgboost"
    if hasattr(cfg, "experiment") and cfg.experiment is not None:
        experiment_name = str(cfg.experiment).split(".")[-1] if "." in str(cfg.experiment) else str(cfg.experiment)

    # Initialize W&B if API key is available
    wandb_initialized = False
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project = os.getenv("WANDB_PROJECT", "pname-arxiv-classifier")

    if wandb_api_key:
        try:
            hydra_cfg = HydraConfig.get()
            run_name = f"{experiment_name}-{hydra_cfg.job.override_dirname}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            wandb_initialized = True
            logger.info(f"W&B initialized successfully: project={wandb_project}, run={run_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
    else:
        logger.info("WANDB_API_KEY not found, skipping W&B logging")

    # Load dataset
    max_categories = cfg.model.get("num_labels", 5)
    train_set, val_set, test_set = arxiv_dataset(max_categories=max_categories)
    logger.info(f"Dataset loaded: {len(train_set)} training, {len(val_set)} validation, {len(test_set)} test samples")

    # Extract texts and labels as lists (for sklearn/XGBoost)
    train_texts = [train_set[i][0] for i in range(len(train_set))]
    train_labels = [int(train_set[i][1].item()) for i in range(len(train_set))]

    val_texts = [val_set[i][0] for i in range(len(val_set))] if len(val_set) > 0 else []
    val_labels = [int(val_set[i][1].item()) for i in range(len(val_set))] if len(val_set) > 0 else []

    test_texts = [test_set[i][0] for i in range(len(test_set))]
    test_labels = [int(test_set[i][1].item()) for i in range(len(test_set))]

    # Optionally use a subset for faster training
    max_samples = cfg.training.get("max_samples", None)
    if max_samples is not None and max_samples < len(train_texts):
        import random

        random.seed(cfg.training.get("seed", 42))
        indices = random.sample(range(len(train_texts)), max_samples)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        logger.info(f"Using subset of {max_samples} samples for faster training")

    logger.info(f"Training on {len(train_texts)} samples")

    # Initialize model
    model = TFIDFXGBoostModel(model_cfg=cfg.model)
    logger.info("Model initialized")

    # Train model with early stopping if validation set is available
    training_start_time = time.time()
    logger.info("Starting training...")

    try:
        if len(val_texts) > 0:
            # Use validation set for early stopping
            logger.info(f"Training with validation set ({len(val_texts)} samples) for early stopping")
            model.fit(train_texts, train_labels, val_texts=val_texts, val_labels=val_labels)
            logger.info("Training completed with early stopping on validation set")
        else:
            # No validation set, train normally
            logger.info("Training without validation set (no early stopping)")
            model.fit(train_texts, train_labels)
            logger.info("Training completed (no validation set for early stopping)")
    except Exception as e:
        logger.error(f"ERROR during training: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise

    training_time = time.time() - training_start_time
    logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Evaluate on validation set
    if len(val_texts) > 0:
        logger.info("Evaluating on validation set...")
        val_preds = model.predict(val_texts)
        val_accuracy = accuracy_score(val_labels, val_preds)
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")

        if wandb_initialized:
            wandb.log({"val/accuracy": val_accuracy})

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_preds = model.predict(test_texts)
    test_accuracy = accuracy_score(test_labels, test_preds)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")

    # Log metrics
    if wandb_initialized:
        wandb.log({"test/accuracy": test_accuracy})
        wandb.log({"training_time_seconds": training_time})

    # Print classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(test_labels, test_preds))

    # Get feature importance for explainability
    top_features = model.get_feature_importance(top_n=20)
    logger.info("\nTop 20 Feature Importances:")
    for feature, importance in top_features:
        logger.info(f"  {feature}: {importance:.4f}")

    if wandb_initialized:
        # Log feature importance as table
        wandb.log(
            {
                "feature_importance": wandb.Table(
                    columns=["feature", "importance"], data=[[feat, imp] for feat, imp in top_features]
                )
            }
        )

    # Save model
    aip_model_dir = os.getenv("AIP_MODEL_DIR")
    if aip_model_dir:
        model_save_path = Path(aip_model_dir) / "trained_model.pkl"
        logger.info(f"Using AIP_MODEL_DIR for model save: {aip_model_dir}")
    else:
        if Path("/outputs").exists():
            model_save_path = Path("/outputs/trained_model.pkl")
        else:
            model_save_path = Path("outputs/trained_model.pkl")
        logger.info(f"Using fallback path for model save: {model_save_path}")

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_save_path))
    logger.info(f"Model saved to: {model_save_path}")

    # Create confusion matrix visualization
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.tight_layout()

    figure_save_path = Path(cfg.training.get("figure_save_path", "reports/figures/confusion_matrix.png"))
    figure_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_save_path)
    logger.info(f"Confusion matrix saved to {figure_save_path}")

    if wandb_initialized:
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

    # Log model as artifact
    if wandb_initialized:
        artifact = wandb.Artifact(
            name=f"model-tfidf-{wandb.run.id}",
            type="model",
            description=f"TF-IDF + XGBoost model from run {wandb.run.id}",
            metadata={
                "test_accuracy": float(test_accuracy),
                "training_time_seconds": float(training_time),
                "num_features": len(model.vectorizer.get_feature_names_out()),
            },
        )
        artifact.add_file(str(model_save_path))
        wandb.log_artifact(artifact)
        logger.info("Model logged as wandb artifact")

        wandb.finish()
        logger.info("Wandb run finished")

    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {model_save_path.resolve()}")
    print(f"{'='*60}\n")


@hydra.main(version_base=None, config_path="../../configs", config_name="config_tfidf")
def main(cfg: DictConfig) -> None:
    """Main entry point for training script.

    Args:
        cfg: Hydra configuration dictionary (automatically loaded).
    """
    train(cfg)


if __name__ == "__main__":
    main()
