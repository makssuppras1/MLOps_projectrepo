import pytest
import torch
from omegaconf import DictConfig

from pname.data import ArXivDataset
from pname.train_tfidf import train as train_tfidf


@pytest.fixture
def minimal_model_cfg():
    return DictConfig(
        {
            "model": {
                "num_labels": 3,
                "max_features": 200,
                "stop_words": "english",
                "ngram_range_min": 1,
                "ngram_range_max": 1,
                "min_df": 1,
                "max_df": 0.95,
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_estimators": 20,
                "random_state": 42,
                "early_stopping_rounds": 5,
            },
            "training": {
                "seed": 42,
                "max_samples": None,
                # Will be overridden per-test to a temp location
                "figure_save_path": "reports/figures/confusion_matrix_tfidf.png",
                "epochs": None,
            },
        }
    )


def build_synthetic_dataset():
    train_texts = [
        "machine learning is great",
        "deep learning neural networks",
        "natural language processing",
        "computer vision image recognition",
        "reinforcement learning agents",
        "statistical methods data analysis",
        "optimization algorithms gradient descent",
        "supervised learning classification",
    ]
    train_labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1], dtype=torch.long)

    val_texts = ["neural network architecture", "text classification model"]
    val_labels = torch.tensor([0, 1], dtype=torch.long)

    test_texts = ["image recognition", "language model"]
    test_labels = torch.tensor([1, 0], dtype=torch.long)

    return (
        ArXivDataset(train_texts, train_labels),
        ArXivDataset(val_texts, val_labels),
        ArXivDataset(test_texts, test_labels),
    )


def build_no_val_dataset():
    # Same as above but with empty validation set
    train_texts = [
        "machine learning is great",
        "deep learning neural networks",
        "natural language processing",
        "computer vision image recognition",
    ]
    train_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    val_texts = []
    val_labels = torch.tensor([], dtype=torch.long)

    test_texts = ["image recognition", "language model"]
    test_labels = torch.tensor([1, 0], dtype=torch.long)

    return (
        ArXivDataset(train_texts, train_labels),
        ArXivDataset(val_texts, val_labels),
        ArXivDataset(test_texts, test_labels),
    )


def test_train_tfidf_saves_artifacts(tmp_path, monkeypatch, minimal_model_cfg):
    # Patch dataset source used inside train_tfidf
    monkeypatch.setattr(
        "pname.train_tfidf.arxiv_dataset",
        lambda max_categories=3: build_synthetic_dataset(),
    )

    # Redirect model and figure outputs to temp locations
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AIP_MODEL_DIR", str(model_dir))

    fig_path = tmp_path / "confusion.png"
    minimal_model_cfg.training.figure_save_path = str(fig_path)

    # Run training
    train_tfidf(minimal_model_cfg)

    # Verify artifacts
    model_path = model_dir / "trained_model.pkl"
    assert model_path.exists(), f"Model file not saved at {model_path}"
    assert fig_path.exists(), f"Confusion matrix not saved at {fig_path}"


def test_train_tfidf_maps_epochs_to_estimators(tmp_path, monkeypatch, minimal_model_cfg):
    from pname.model_tfidf import TFIDFXGBoostModel

    monkeypatch.setattr(
        "pname.train_tfidf.arxiv_dataset",
        lambda max_categories=3: build_synthetic_dataset(),
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AIP_MODEL_DIR", str(model_dir))

    fig_path = tmp_path / "confusion.png"
    minimal_model_cfg.training.figure_save_path = str(fig_path)

    # Map epochs to n_estimators and train
    minimal_model_cfg.training.epochs = 10
    train_tfidf(minimal_model_cfg)

    # Load model and verify estimator count
    model_path = model_dir / "trained_model.pkl"
    loaded = TFIDFXGBoostModel.load(str(model_path))
    assert loaded.classifier.get_params()["n_estimators"] == 10


def test_train_tfidf_no_val_set(tmp_path, monkeypatch, minimal_model_cfg):
    monkeypatch.setattr(
        "pname.train_tfidf.arxiv_dataset",
        lambda max_categories=3: build_no_val_dataset(),
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AIP_MODEL_DIR", str(model_dir))

    fig_path = tmp_path / "confusion.png"
    minimal_model_cfg.training.figure_save_path = str(fig_path)

    train_tfidf(minimal_model_cfg)

    model_path = model_dir / "trained_model.pkl"
    assert model_path.exists()
    assert fig_path.exists()
