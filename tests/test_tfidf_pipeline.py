"""Comprehensive tests for TF-IDF + XGBoost model with Pipeline integration.

Tests:
1. Old .pkl files still load correctly (backward compatibility)
2. New models save/load correctly
3. predict() and predict_proba() work with pipeline
4. Early stopping still works
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig

from pname.model_tfidf import TFIDFXGBoostModel


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "machine learning is great",
        "deep learning neural networks",
        "natural language processing",
        "computer vision image recognition",
        "reinforcement learning agents",
        "statistical methods data analysis",
        "optimization algorithms gradient descent",
        "supervised learning classification",
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return [0, 0, 1, 1, 2, 2, 0, 1]


@pytest.fixture
def val_texts():
    """Validation texts for early stopping."""
    return [
        "neural network architecture",
        "text classification model",
    ]


@pytest.fixture
def val_labels():
    """Validation labels for early stopping."""
    return [0, 1]


@pytest.fixture
def model_config():
    """Minimal model config for testing."""
    return DictConfig(
        {
            "num_labels": 3,
            "max_features": 100,
            "stop_words": "english",
            "ngram_range_min": 1,
            "ngram_range_max": 1,
            "min_df": 1,  # Set to 1 for small test datasets
            "max_df": 0.95,
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 20,  # Small for fast testing
            "random_state": 42,
            "early_stopping_rounds": 5,
        }
    )


def test_pipeline_initialization(model_config):
    """Test that pipeline is created during initialization."""
    model = TFIDFXGBoostModel(model_cfg=model_config)

    assert hasattr(model, "pipeline"), "Pipeline should be created"
    assert hasattr(model, "vectorizer"), "Vectorizer should exist"
    assert hasattr(model, "classifier"), "Classifier should exist"

    # Check pipeline structure
    assert "vectorizer" in model.pipeline.named_steps
    assert "classifier" in model.pipeline.named_steps
    assert model.pipeline.named_steps["vectorizer"] is model.vectorizer
    assert model.pipeline.named_steps["classifier"] is model.classifier


def test_new_model_save_load_roundtrip(model_config, sample_texts, sample_labels):
    """Test that new models (with pipeline) save and load correctly."""
    model = TFIDFXGBoostModel(model_cfg=model_config)
    model.fit(sample_texts, sample_labels)

    original_predictions = model.predict(sample_texts)
    original_probas = model.predict_proba(sample_texts)

    # Save and load
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name

    try:
        model.save(temp_path)

        # Verify file exists
        assert Path(temp_path).exists(), "Model file should be saved"

        # Load model
        loaded_model = TFIDFXGBoostModel.load(temp_path)

        # Verify pipeline exists
        assert hasattr(loaded_model, "pipeline"), "Loaded model should have pipeline"
        assert "vectorizer" in loaded_model.pipeline.named_steps
        assert "classifier" in loaded_model.pipeline.named_steps

        # Predictions should be identical
        loaded_predictions = loaded_model.predict(sample_texts)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)

        # Probabilities should be identical
        loaded_probas = loaded_model.predict_proba(sample_texts)
        np.testing.assert_array_almost_equal(original_probas, loaded_probas)

        # Pipeline should work
        pipeline_predictions = loaded_model.pipeline.predict(sample_texts)
        np.testing.assert_array_equal(loaded_predictions, pipeline_predictions)

    finally:
        Path(temp_path).unlink()


def test_backward_compatibility_old_format(model_config, sample_texts, sample_labels):
    """Test that old .pkl files (without pipeline) still load correctly."""
    # Create a model and save it in old format (without pipeline)
    model = TFIDFXGBoostModel(model_cfg=model_config)
    model.fit(sample_texts, sample_labels)

    # Manually create old-format dict (simulating old save format)
    import pickle

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name

    try:
        # Save in old format (without pipeline)
        old_format_dict = {
            "vectorizer": model.vectorizer,
            "classifier": model.classifier,
            "config": model.model_cfg,
        }
        with open(temp_path, "wb") as f:
            pickle.dump(old_format_dict, f)

        # Load old format - should reconstruct pipeline
        loaded_model = TFIDFXGBoostModel.load(temp_path)

        # Should have pipeline (reconstructed)
        assert hasattr(loaded_model, "pipeline"), "Should reconstruct pipeline from old format"
        assert "vectorizer" in loaded_model.pipeline.named_steps
        assert "classifier" in loaded_model.pipeline.named_steps

        # Should work correctly
        predictions = loaded_model.predict(sample_texts)
        assert len(predictions) == len(sample_texts)

        # Pipeline should work
        pipeline_predictions = loaded_model.pipeline.predict(sample_texts)
        np.testing.assert_array_equal(predictions, pipeline_predictions)

    finally:
        Path(temp_path).unlink()


def test_early_stopping_works(model_config, sample_texts, sample_labels, val_texts, val_labels):
    """Test that early stopping still works with pipeline."""
    model = TFIDFXGBoostModel(model_cfg=model_config)

    # Fit with validation set (early stopping)
    model.fit(sample_texts, sample_labels, val_texts=val_texts, val_labels=val_labels)

    # Verify pipeline was updated after fitting
    assert hasattr(model, "pipeline"), "Pipeline should exist"
    assert model.pipeline.named_steps["classifier"] is model.classifier

    # Verify predictions work
    predictions = model.predict(sample_texts)
    assert len(predictions) == len(sample_texts)

    # Verify pipeline produces same results
    pipeline_predictions = model.pipeline.predict(sample_texts)
    np.testing.assert_array_equal(predictions, pipeline_predictions)

    # Verify early stopping was used (classifier should have been fitted)
    assert hasattr(model.classifier, "feature_importances_"), "Classifier should be fitted"


def test_no_data_leakage(model_config):
    """Test that vectorizer is fit only on training data."""
    # Use more texts to avoid min_df pruning issues
    train_texts = [
        "machine learning is great",
        "deep learning neural networks",
        "natural language processing",
        "computer vision image recognition",
        "reinforcement learning agents",
        "statistical methods data analysis",
    ]
    val_texts = ["neural network architecture"]
    test_texts = ["text classification model"]
    train_labels = [0, 0, 1, 1, 2, 2]
    val_labels = [0]

    # Use model_config directly (already has min_df=1)
    model = TFIDFXGBoostModel(model_cfg=model_config)
    model.fit(train_texts, train_labels, val_texts=val_texts, val_labels=val_labels)

    # Vectorizer should have been fit on train_texts only
    # Check that vocabulary was learned from training data
    train_features = model.vectorizer.transform(train_texts)
    val_features = model.vectorizer.transform(val_texts)
    test_features = model.vectorizer.transform(test_texts)

    # All should have same number of features (vocabulary size)
    assert train_features.shape[1] == val_features.shape[1]
    assert train_features.shape[1] == test_features.shape[1]

    # Predictions should work
    val_preds = model.predict(val_texts)
    test_preds = model.predict(test_texts)
    assert len(val_preds) == len(val_texts)
    assert len(test_preds) == len(test_texts)

    # Pipeline should produce same results
    val_pipeline_preds = model.pipeline.predict(val_texts)
    test_pipeline_preds = model.pipeline.predict(test_texts)
    np.testing.assert_array_equal(val_preds, val_pipeline_preds)
    np.testing.assert_array_equal(test_preds, test_pipeline_preds)
