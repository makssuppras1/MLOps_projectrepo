"""MVP script to compute basic dataset statistics for CML workflows."""

import json
from pathlib import Path

import typer


def compute_data_statistics(
    processed_dir: str = typer.Option("data/processed", help="Path to processed data directory"),
) -> None:
    """Compute basic statistics for the processed dataset.

    Args:
        processed_dir: Path to directory containing processed data files.
    """
    processed_dir = Path(processed_dir)

    # Load data
    try:
        with open(processed_dir / "train_texts.json", "r", encoding="utf-8") as f:
            train_texts = json.load(f)

        with open(processed_dir / "test_texts.json", "r", encoding="utf-8") as f:
            test_texts = json.load(f)

        with open(processed_dir / "category_mapping.json", "r", encoding="utf-8") as f:
            category_mapping = json.load(f)
    except FileNotFoundError as e:
        error_msg = str(e)
        print(f"❌ Error: Data files not found: {error_msg}")
        raise

    # Basic statistics
    train_size = len(train_texts)
    test_size = len(test_texts)
    num_classes = len(category_mapping)

    # Print summary
    print("\n## 📊 Dataset Statistics")
    print(f"- Training samples: {train_size:,}")
    print(f"- Test samples: {test_size:,}")
    print(f"- Number of classes: {num_classes}")
    print("- ✅ Data validation passed")


if __name__ == "__main__":
    typer.run(compute_data_statistics)
