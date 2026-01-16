"""Data loading and preprocessing utilities for ArXiv dataset."""

import csv
import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
import typer


def preprocess_data(
    raw_dir: str = typer.Argument(..., help="Path to raw data directory"),
    processed_dir: str = typer.Argument(..., help="Path to processed data directory"),
    test_split: float = typer.Option(0.2, help="Fraction of data to use for testing"),
    seed: int = typer.Option(42, help="Random seed for train/test split"),
) -> None:
    """
    Process raw ArXiv data and save it to processed directory.

    Loads ArXiv dataset from raw_dir, processes text (title + abstract),
    encodes categories as integer labels, splits into train/test sets, and saves
    processed data to processed_dir. Creates the following files:
    - train_texts.json: Training text samples
    - train_labels.pt: Training labels as PyTorch tensor
    - test_texts.json: Test text samples
    - test_labels.pt: Test labels as PyTorch tensor
    - category_mapping.json: Mapping from category strings to integer labels

    Args:
        raw_dir: Path to directory containing raw CSV data files.
        processed_dir: Path to directory where processed data will be saved.
        test_split: Fraction of data to use for testing (0.0 to 1.0). Defaults to 0.2.
        seed: Random seed for reproducible train/test split. Defaults to 42.

    Raises:
        FileNotFoundError: If no CSV file is found in raw_dir.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    # Create processed directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Find CSV file in raw directory
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {raw_dir}")
    csv_file = csv_files[0]  # Use first CSV file found

    # Load and process data
    texts = []
    categories = []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Combine title and abstract
            title = row.get("title", "").strip()
            abstract = row.get("abstract", "").strip()
            text = f"{title} {abstract}".strip()

            # Get category (use first category if multiple)
            category = row.get("categories", "").strip()
            if not category:
                continue  # Skip rows without category

            if text:  # Only add if we have text
                texts.append(text)
                categories.append(category)

    # Encode categories as integer labels
    unique_categories = sorted(set(categories))
    category_to_label = {cat: idx for idx, cat in enumerate(unique_categories)}
    labels = [category_to_label[cat] for cat in categories]

    # Split into train and test
    random.seed(seed)
    indices = list(range(len(texts)))
    random.shuffle(indices)

    split_idx = int(len(indices) * (1 - test_split))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_texts = [texts[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    # Save processed data
    with open(processed_dir / "train_texts.json", "w", encoding="utf-8") as f:
        json.dump(train_texts, f, ensure_ascii=False)
    torch.save(torch.tensor(train_labels, dtype=torch.long), processed_dir / "train_labels.pt")

    with open(processed_dir / "test_texts.json", "w", encoding="utf-8") as f:
        json.dump(test_texts, f, ensure_ascii=False)
    torch.save(torch.tensor(test_labels, dtype=torch.long), processed_dir / "test_labels.pt")

    # Save category mapping
    with open(processed_dir / "category_mapping.json", "w", encoding="utf-8") as f:
        json.dump(category_to_label, f, ensure_ascii=False)

    print(f"Processed data saved to {processed_dir}")
    print(f"Training set size: {len(train_texts)}")
    print(f"Test set size: {len(test_texts)}")
    print(f"Number of categories: {len(unique_categories)}")


class ArXivDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for ArXiv paper text classification.

    Stores pairs of text samples and their corresponding integer labels.
    """

    def __init__(self, texts: List[str], labels: torch.Tensor) -> None:
        """
        Initialize ArXiv dataset.

        Args:
            texts: List of text strings (title + abstract).
            labels: Tensor of integer labels corresponding to each text.
        """
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (text, label) for the sample at index idx.
        """
        return self.texts[idx], self.labels[idx]


def arxiv_dataset() -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Load and return train and test ArXiv datasets.

    Loads preprocessed data from data/processed directory and creates
    PyTorch Dataset objects for training and testing.

    Returns:
        Tuple of (train_dataset, test_dataset) containing ArXivDataset instances.

    Raises:
        FileNotFoundError: If processed data files are not found.
    """
    processed_dir = Path("data/processed")

    with open(processed_dir / "train_texts.json", "r", encoding="utf-8") as f:
        train_texts = json.load(f)
    train_labels = torch.load(processed_dir / "train_labels.pt").long()

    with open(processed_dir / "test_texts.json", "r", encoding="utf-8") as f:
        test_texts = json.load(f)
    test_labels = torch.load(processed_dir / "test_labels.pt").long()

    train_set = ArXivDataset(train_texts, train_labels)
    test_set = ArXivDataset(test_texts, test_labels)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)
