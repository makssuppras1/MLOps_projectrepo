#!/usr/bin/env python3
"""Code verification script - checks implementation without running XGBoost.

This script verifies that all the required changes are in place by:
1. Parsing the Python AST to check for Pipeline usage
2. Verifying imports
3. Checking function implementations
4. Validating config file changes
"""

from pathlib import Path


def check_file_contains(file_path: Path, patterns: list[str]) -> dict[str, bool]:
    """Check if file contains all required patterns."""
    content = file_path.read_text()
    results = {}
    for pattern in patterns:
        results[pattern] = pattern in content
    return results


def verify_model_tfidf():
    """Verify model_tfidf.py has all required changes."""
    file_path = Path("src/pname/model_tfidf.py")
    content = file_path.read_text()

    checks = {
        "Pipeline import": "from sklearn.pipeline import Pipeline" in content,
        "Pipeline creation": "self.pipeline = Pipeline([" in content,
        "random_state from config": 'random_state=model_cfg.get("random_state"' in content,
        "predict uses pipeline": "return self.pipeline.predict(texts)" in content,
        "predict_proba uses pipeline": "return self.pipeline.predict_proba(texts)" in content,
        "save pipeline": '"pipeline": self.pipeline' in content,
        "load pipeline check": '"pipeline" in model_dict' in content,
        "backward compatibility": "reconstruct pipeline" in content.lower() or "Backward compatibility" in content,
    }

    return checks


def verify_config():
    """Verify config file has random_state."""
    file_path = Path("configs/model_tfidf.yaml")
    content = file_path.read_text()

    checks = {
        "random_state in config": "random_state:" in content,
    }

    return checks


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("TF-IDF Pipeline Integration - Code Verification")
    print("=" * 60)

    all_passed = True

    # Check model_tfidf.py
    print("\n1. Verifying model_tfidf.py...")
    model_checks = verify_model_tfidf()
    for check, passed in model_checks.items():
        status = "✓" if passed else "✗"
        print(f"   {status} {check}")
        if not passed:
            all_passed = False

    # Check config file
    print("\n2. Verifying configs/model_tfidf.yaml...")
    config_checks = verify_config()
    for check, passed in config_checks.items():
        status = "✓" if passed else "✗"
        print(f"   {status} {check}")
        if not passed:
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CODE CHANGES VERIFIED")
        print("\nNote: To run full tests, install OpenMP:")
        print("  brew install libomp")
        print("\nThen run:")
        print("  uv run pytest tests/test_tfidf_pipeline.py -v")
    else:
        print("✗ SOME CHECKS FAILED")
        print("   Please review the implementation")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
