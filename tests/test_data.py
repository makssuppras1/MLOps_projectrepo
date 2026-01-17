import json
import tempfile
from pathlib import Path

import pytest
import torch

from pname.data import ArXivDataset, preprocess_data


class TestArXivDataset:
    """Test suite for ArXivDataset class."""

    def test_dataset_initialization(self):
        """Test ArXivDataset initialization with sample data."""
        texts = ["paper 1 abstract here", "paper 2 abstract here", "paper 3 abstract here"]
        labels = torch.tensor([0, 1, 2])
        
        dataset = ArXivDataset(texts, labels)
        
        assert dataset.texts == texts
        assert torch.equal(dataset.labels, labels)

    def test_dataset_len(self):
        """Test dataset length."""
        texts = ["text 1", "text 2", "text 3"]
        labels = torch.tensor([0, 1, 0])
        
        dataset = ArXivDataset(texts, labels)
        
        assert len(dataset) == 3

    def test_dataset_getitem(self):
        """Test dataset __getitem__ returns correct samples."""
        texts = ["paper A", "paper B", "paper C"]
        labels = torch.tensor([0, 1, 2])
        
        dataset = ArXivDataset(texts, labels)
        
        text, label = dataset[0]
        assert text == "paper A"
        assert label == 0
        
        text, label = dataset[1]
        assert text == "paper B"
        assert label == 1

    def test_dataset_getitem_out_of_bounds(self):
        """Test dataset indexing raises error for out of bounds."""
        texts = ["text 1"]
        labels = torch.tensor([0])
        
        dataset = ArXivDataset(texts, labels)
        
        with pytest.raises(IndexError):
            _ = dataset[5]

    def test_dataset_empty(self):
        """Test dataset with empty texts and labels."""
        texts = []
        labels = torch.tensor([])
        
        dataset = ArXivDataset(texts, labels)
        
        assert len(dataset) == 0


class TestPreprocessData:
    """Test suite for preprocess_data function."""

    def test_preprocess_data_creates_output_files(self):
        """Test that preprocess_data creates all expected output files."""
        with tempfile.TemporaryDirectory() as raw_dir, \
             tempfile.TemporaryDirectory() as processed_dir:
            
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
            preprocess_data(str(raw_path), str(processed_dir), test_split=0.25, seed=42)
            
            # Check that all expected files exist
            processed_path = Path(processed_dir)
            assert (processed_path / "train_texts.json").exists()
            assert (processed_path / "train_labels.pt").exists()
            assert (processed_path / "test_texts.json").exists()
            assert (processed_path / "test_labels.pt").exists()
            assert (processed_path / "category_mapping.json").exists()

    def test_preprocess_data_correct_split(self):
        """Test that preprocess_data splits data correctly."""
        with tempfile.TemporaryDirectory() as raw_dir, \
             tempfile.TemporaryDirectory() as processed_dir:
            
            raw_path = Path(raw_dir)
            csv_path = raw_path / "sample.csv"
            
            # Create CSV with 100 samples
            csv_data = "title,abstract,categories\n"
            for i in range(100):
                category = ["cs.AI", "cs.LG", "cs.NE"][i % 3]
                csv_data += f"Paper {i},Abstract {i},{category}\n"
            
            csv_path.write_text(csv_data)
            
            preprocess_data(str(raw_path), str(processed_dir), test_split=0.2, seed=42)
            
            processed_path = Path(processed_dir)
            train_labels = torch.load(processed_path / "train_labels.pt")
            test_labels = torch.load(processed_path / "test_labels.pt")
            
            # Check split ratio (80/20)
            assert len(train_labels) == 80
            assert len(test_labels) == 20

    def test_preprocess_data_category_mapping(self):
        """Test that category mapping is created correctly."""
        with tempfile.TemporaryDirectory() as raw_dir, \
             tempfile.TemporaryDirectory() as processed_dir:
            
            raw_path = Path(raw_dir)
            csv_path = raw_path / "sample.csv"
            
            csv_data = "title,abstract,categories\n"
            csv_data += "Paper 1,Abstract 1,cs.AI\n"
            csv_data += "Paper 2,Abstract 2,cs.LG\n"
            csv_data += "Paper 3,Abstract 3,cs.AI\n"
            
            csv_path.write_text(csv_data)
            
            preprocess_data(str(raw_path), str(processed_dir), test_split=0.25, seed=42)
            
            processed_path = Path(processed_dir)
            with open(processed_path / "category_mapping.json", "r") as f:
                mapping = json.load(f)
            
            assert "cs.AI" in mapping
            assert "cs.LG" in mapping
            assert mapping["cs.AI"] == 0
            assert mapping["cs.LG"] == 1

    def test_preprocess_data_no_csv_raises_error(self):
        """Test that FileNotFoundError is raised when no CSV found."""
        with tempfile.TemporaryDirectory() as raw_dir, \
             tempfile.TemporaryDirectory() as processed_dir:
            
            # Don't create any CSV file
            with pytest.raises(FileNotFoundError):
                preprocess_data(str(raw_dir), str(processed_dir))

    def test_preprocess_data_skips_rows_without_category(self):
        """Test that rows without categories are skipped."""
        with tempfile.TemporaryDirectory() as raw_dir, \
             tempfile.TemporaryDirectory() as processed_dir:
            
            raw_path = Path(raw_dir)
            csv_path = raw_path / "sample.csv"
            
            csv_data = "title,abstract,categories\n"
            csv_data += "Paper 1,Abstract 1,cs.AI\n"
            csv_data += "Paper 2,Abstract 2,\n"  # Missing category
            csv_data += "Paper 3,Abstract 3,cs.LG\n"
            
            csv_path.write_text(csv_data)
            
            preprocess_data(str(raw_path), str(processed_dir), test_split=0.25, seed=42)
            
            processed_path = Path(processed_dir)
            with open(processed_path / "train_texts.json", "r") as f:
                train_texts = json.load(f)
            train_labels = torch.load(processed_path / "train_labels.pt")
            
            # Should only have 2 samples (Paper 2 skipped)
            assert len(train_texts) + len(torch.load(processed_path / "test_labels.pt")) == 2

    def test_preprocess_data_reproducibility(self):
        """Test that same seed produces same split."""
        with tempfile.TemporaryDirectory() as raw_dir1, \
             tempfile.TemporaryDirectory() as processed_dir1, \
             tempfile.TemporaryDirectory() as processed_dir2:
            
            raw_path = Path(raw_dir1)
            csv_path = raw_path / "sample.csv"
            
            csv_data = "title,abstract,categories\n"
            for i in range(50):
                csv_data += f"Paper {i},Abstract {i},cs.AI\n"
            
            csv_path.write_text(csv_data)
            
            # Run twice with same seed
            preprocess_data(str(raw_path), str(processed_dir1), test_split=0.2, seed=42)
            preprocess_data(str(raw_path), str(processed_dir2), test_split=0.2, seed=42)
            
            # Check that results are identical
            labels1 = torch.load(Path(processed_dir1) / "train_labels.pt")
            labels2 = torch.load(Path(processed_dir2) / "train_labels.pt")
            
            assert torch.equal(labels1, labels2)