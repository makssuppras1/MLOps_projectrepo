import tempfile
from pathlib import Path

import torch

from pname.data import preprocess_data


class TestPreprocessData:
    """Test suite for preprocess_data function."""

    def test_preprocess_data_creates_output_files(self):
        """Test that preprocess_data creates all expected output files."""
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as processed_dir:
            # Create sample CSV file
            raw_path = Path(raw_dir)
            csv_path = raw_path / "sample.csv"

            csv_data = "title,abstract,categories\n"
            csv_data += "Paper 1,Abstract 1,cs.AI\n"
            csv_data += "Paper 2,Abstract 2,cs.LG\n"
            csv_data += "Paper 3,Abstract 3,cs.AI\n"
            csv_data += "Paper 4,Abstract 4,cs.NE\n"

            csv_path.write_text(csv_data)

            # Run preprocessing
            preprocess_data(
                str(raw_path), str(processed_dir), train_split=0.75, val_split=0.0, test_split=0.25, seed=42
            )

            # Check that all expected files exist
            processed_path = Path(processed_dir)
            assert (processed_path / "train_texts.json").exists()
            assert (processed_path / "train_labels.pt").exists()
            assert (processed_path / "test_texts.json").exists()
            assert (processed_path / "test_labels.pt").exists()
            assert (processed_path / "category_mapping.json").exists()

    def test_preprocess_data_correct_split(self):
        """Test that preprocess_data splits data correctly."""
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as processed_dir:
            raw_path = Path(raw_dir)
            csv_path = raw_path / "sample.csv"

            # Create CSV with 100 samples
            csv_data = "title,abstract,categories\n"
            for i in range(100):
                category = ["cs.AI", "cs.LG", "cs.NE"][i % 3]
                csv_data += f"Paper {i},Abstract {i},{category}\n"

            csv_path.write_text(csv_data)

            preprocess_data(str(raw_path), str(processed_dir), train_split=0.8, val_split=0.0, test_split=0.2, seed=42)

            processed_path = Path(processed_dir)
            train_labels = torch.load(processed_path / "train_labels.pt")
            test_labels = torch.load(processed_path / "test_labels.pt")

            # Check split ratio (80/20)
            assert len(train_labels) == 80
            assert len(test_labels) == 20
