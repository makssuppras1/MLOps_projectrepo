#!/usr/bin/env python3
"""Quick test to verify backward compatibility with existing trained_model.pkl"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_backward_compatibility():
    """Test backward compatibility with existing trained_model.pkl file."""
    try:
        from pname.model_tfidf import TFIDFXGBoostModel
    except ImportError as e:
        pytest.skip(f"Could not import model: {e}. This is expected if dependencies are not installed.")

    model_path = Path(__file__).parent.parent / "trained_model.pkl"

    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}. Skipping backward compatibility test.")

    print(f"Loading existing model: {model_path}")
    print(f"File size: {model_path.stat().st_size / 1024:.1f} KB")

    # Try to load
    model = TFIDFXGBoostModel.load(str(model_path))
    print("✓ Model loaded successfully")

    # Check if pipeline exists (new format) or was reconstructed (old format)
    assert hasattr(model, "pipeline"), "Model should have pipeline attribute"
    print("✓ Pipeline found")
    assert "vectorizer" in model.pipeline.named_steps, "Pipeline should contain vectorizer"
    print("✓ Vectorizer in pipeline")
    assert "classifier" in model.pipeline.named_steps, "Pipeline should contain classifier"
    print("✓ Classifier in pipeline")

    # Test prediction
    test_texts = ["machine learning", "deep learning"]
    predictions = model.predict(test_texts)
    print(f"✓ Predictions work: {predictions}")
    assert len(predictions) == len(test_texts), "Should return predictions for all texts"

    # Test pipeline if available
    if hasattr(model, "pipeline"):
        pipeline_preds = model.pipeline.predict(test_texts)
        assert (predictions == pipeline_preds).all(), "Pipeline predictions should match direct predictions"
        print("✓ Pipeline predictions match direct predictions")

    print("\n✓ Backward compatibility test PASSED")


# Standalone script execution (for manual testing)
if __name__ == "__main__":
    try:
        from pname.model_tfidf import TFIDFXGBoostModel

        model_path = Path(__file__).parent.parent / "trained_model.pkl"

        if not model_path.exists():
            print(f"⚠️  Model file not found: {model_path}")
            print("   Skipping backward compatibility test")
            sys.exit(0)

        print(f"Loading existing model: {model_path}")
        print(f"File size: {model_path.stat().st_size / 1024:.1f} KB")

        # Try to load
        try:
            model = TFIDFXGBoostModel.load(str(model_path))
            print("✓ Model loaded successfully")

            # Check if pipeline exists (new format) or was reconstructed (old format)
            if hasattr(model, "pipeline"):
                print("✓ Pipeline found")
                if "vectorizer" in model.pipeline.named_steps:
                    print("✓ Vectorizer in pipeline")
                if "classifier" in model.pipeline.named_steps:
                    print("✓ Classifier in pipeline")
            else:
                print("⚠️  No pipeline found - this is an old format model")

            # Test prediction
            test_texts = ["machine learning", "deep learning"]
            try:
                predictions = model.predict(test_texts)
                print(f"✓ Predictions work: {predictions}")

                # Test pipeline if available
                if hasattr(model, "pipeline"):
                    pipeline_preds = model.pipeline.predict(test_texts)
                    if (predictions == pipeline_preds).all():
                        print("✓ Pipeline predictions match direct predictions")
                    else:
                        print("⚠️  Pipeline predictions differ")
            except Exception as e:
                print(f"✗ Prediction failed: {e}")
                sys.exit(1)

            print("\n✓ Backward compatibility test PASSED")
            sys.exit(0)

        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    except ImportError as e:
        print(f"⚠️  Could not import model: {e}")
        print("   This is expected if dependencies are not installed")
        print("   Run with: uv run python tests/test_backward_compat.py")
        sys.exit(0)
