#!/usr/bin/env python3
"""Standalone test script for TF-IDF Pipeline integration.

Tests:
1. Old .pkl files still load correctly (backward compatibility)
2. New models save/load correctly
3. predict() and predict_proba() work with pipeline
4. Early stopping still works
"""

import pickle
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from pname.model_tfidf import TFIDFXGBoostModel  # noqa: E402


def test_pipeline_initialization():
    """Test that pipeline is created during initialization."""
    print("Test 1: Pipeline initialization...")
    model_config = DictConfig(
        {
            "num_labels": 3,
            "max_features": 100,
            "stop_words": "english",
            "ngram_range_min": 1,
            "ngram_range_max": 1,
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 20,
            "random_state": 42,
        }
    )

    model = TFIDFXGBoostModel(model_cfg=model_config)

    assert hasattr(model, "pipeline"), "Pipeline should be created"
    assert hasattr(model, "vectorizer"), "Vectorizer should exist"
    assert hasattr(model, "classifier"), "Classifier should exist"
    assert "vectorizer" in model.pipeline.named_steps
    assert "classifier" in model.pipeline.named_steps
    print("  ✓ Pipeline initialized correctly")


def test_predict_uses_pipeline():
    """Test that predict() uses pipeline."""
    print("\nTest 2: predict() uses pipeline...")
    sample_texts = ["machine learning", "deep learning", "natural language"]
    sample_labels = [0, 0, 1]

    model_config = DictConfig(
        {
            "num_labels": 2,
            "max_features": 50,
            "stop_words": "english",
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_estimators": 10,
            "random_state": 42,
        }
    )

    model = TFIDFXGBoostModel(model_cfg=model_config)
    model.fit(sample_texts, sample_labels)

    predictions = model.predict(sample_texts)
    pipeline_predictions = model.pipeline.predict(sample_texts)

    np.testing.assert_array_equal(predictions, pipeline_predictions)
    print("  ✓ predict() uses pipeline correctly")


def test_predict_proba_uses_pipeline():
    """Test that predict_proba() uses pipeline."""
    print("\nTest 3: predict_proba() uses pipeline...")
    sample_texts = ["machine learning", "deep learning", "natural language"]
    sample_labels = [0, 0, 1]

    model_config = DictConfig(
        {
            "num_labels": 2,
            "max_features": 50,
            "stop_words": "english",
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_estimators": 10,
            "random_state": 42,
        }
    )

    model = TFIDFXGBoostModel(model_cfg=model_config)
    model.fit(sample_texts, sample_labels)

    probas = model.predict_proba(sample_texts)
    pipeline_probas = model.pipeline.predict_proba(sample_texts)

    np.testing.assert_array_almost_equal(probas, pipeline_probas)
    assert np.allclose(probas.sum(axis=1), 1.0), "Probabilities should sum to 1"
    print("  ✓ predict_proba() uses pipeline correctly")


def test_new_model_save_load():
    """Test that new models (with pipeline) save and load correctly."""
    print("\nTest 4: New model save/load roundtrip...")
    sample_texts = ["machine learning", "deep learning", "natural language", "computer vision"]
    sample_labels = [0, 0, 1, 1]

    model_config = DictConfig(
        {
            "num_labels": 2,
            "max_features": 50,
            "stop_words": "english",
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_estimators": 10,
            "random_state": 42,
        }
    )

    model = TFIDFXGBoostModel(model_cfg=model_config)
    model.fit(sample_texts, sample_labels)

    original_predictions = model.predict(sample_texts)
    original_probas = model.predict_proba(sample_texts)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name

    try:
        model.save(temp_path)
        assert Path(temp_path).exists(), "Model file should be saved"

        loaded_model = TFIDFXGBoostModel.load(temp_path)
        assert hasattr(loaded_model, "pipeline"), "Loaded model should have pipeline"

        loaded_predictions = loaded_model.predict(sample_texts)
        loaded_probas = loaded_model.predict_proba(sample_texts)

        np.testing.assert_array_equal(original_predictions, loaded_predictions)
        np.testing.assert_array_almost_equal(original_probas, loaded_probas)

        # Pipeline should work
        pipeline_predictions = loaded_model.pipeline.predict(sample_texts)
        np.testing.assert_array_equal(loaded_predictions, pipeline_predictions)

        print("  ✓ New model save/load works correctly")
    finally:
        Path(temp_path).unlink()


def test_backward_compatibility():
    """Test that old .pkl files (without pipeline) still load correctly."""
    print("\nTest 5: Backward compatibility with old format...")
    sample_texts = ["machine learning", "deep learning", "natural language"]
    sample_labels = [0, 0, 1]

    model_config = DictConfig(
        {
            "num_labels": 2,
            "max_features": 50,
            "stop_words": "english",
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_estimators": 10,
            "random_state": 42,
        }
    )

    model = TFIDFXGBoostModel(model_cfg=model_config)
    model.fit(sample_texts, sample_labels)

    # Create old-format dict (simulating old save format)
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

        assert hasattr(loaded_model, "pipeline"), "Should reconstruct pipeline from old format"
        assert "vectorizer" in loaded_model.pipeline.named_steps
        assert "classifier" in loaded_model.pipeline.named_steps

        predictions = loaded_model.predict(sample_texts)
        assert len(predictions) == len(sample_texts)

        pipeline_predictions = loaded_model.pipeline.predict(sample_texts)
        np.testing.assert_array_equal(predictions, pipeline_predictions)

        print("  ✓ Backward compatibility works correctly")
    finally:
        Path(temp_path).unlink()


def test_early_stopping():
    """Test that early stopping still works with pipeline."""
    print("\nTest 6: Early stopping with pipeline...")
    train_texts = ["machine learning", "deep learning", "natural language", "computer vision"]
    train_labels = [0, 0, 1, 1]
    val_texts = ["neural network", "text classification"]
    val_labels = [0, 1]

    model_config = DictConfig(
        {
            "num_labels": 2,
            "max_features": 50,
            "stop_words": "english",
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_estimators": 20,
            "random_state": 42,
            "early_stopping_rounds": 5,
        }
    )

    model = TFIDFXGBoostModel(model_cfg=model_config)

    # Fit with validation set (early stopping)
    model.fit(train_texts, train_labels, val_texts=val_texts, val_labels=val_labels)

    assert hasattr(model, "pipeline"), "Pipeline should exist"
    assert model.pipeline.named_steps["classifier"] is model.classifier

    predictions = model.predict(train_texts)
    assert len(predictions) == len(train_texts)

    pipeline_predictions = model.pipeline.predict(train_texts)
    np.testing.assert_array_equal(predictions, pipeline_predictions)

    assert hasattr(model.classifier, "feature_importances_"), "Classifier should be fitted"

    print("  ✓ Early stopping works correctly")


def test_random_state_from_config():
    """Test that random_state comes from config."""
    print("\nTest 7: Random state from config...")

    model_config = DictConfig(
        {
            "num_labels": 2,
            "max_features": 50,
            "stop_words": "english",
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_estimators": 10,
            "random_state": 123,
        }
    )

    model = TFIDFXGBoostModel(model_cfg=model_config)
    assert model.classifier.get_params()["random_state"] == 123

    print("  ✓ Random state from config works correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TF-IDF Pipeline Integration Tests")
    print("=" * 60)

    try:
        test_pipeline_initialization()
        test_predict_uses_pipeline()
        test_predict_proba_uses_pipeline()
        test_new_model_save_load()
        test_backward_compatibility()
        test_early_stopping()
        test_random_state_from_config()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
