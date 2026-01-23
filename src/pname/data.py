import csv
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import requests
import torch
import typer
from loguru import logger

# GCS support removed - using only local and mounted paths


def _extract_typer_default(value, default):
    """Extract default value from typer OptionInfo if needed, otherwise return value."""
    if isinstance(value, typer.models.OptionInfo):
        return default
    return value


def download_data(
    raw_dir: str = "data/raw",
    dataset_url: str = "https://www.kaggle.com/api/v1/datasets/download/sumitm004/arxiv-scientific-research-papers-dataset",
    force_download: bool = False,
) -> None:
    """Download ArXiv dataset from Kaggle and extract it.

    Downloads the dataset zip file, extracts it to raw_dir, and removes the zip file.
    Skips download if CSV files already exist in raw_dir (unless force_download=True).

    Args:
        raw_dir: Path to directory where raw data will be saved (default: "data/raw").
        dataset_url: URL to download the dataset from (default: Kaggle API URL).
        force_download: If True, download even if data already exists (default: False).

    Raises:
        FileNotFoundError: If required tools (curl or unzip) are not available.
        requests.RequestException: If download fails.
        zipfile.BadZipFile: If downloaded file is not a valid zip file.
    """
    raw_dir_path = Path(raw_dir)
    raw_dir_path.mkdir(parents=True, exist_ok=True)

    # Check if data already exists (look for CSV files)
    csv_files = list(raw_dir_path.glob("*.csv"))
    if csv_files and not force_download:
        logger.info(f"Data already exists in {raw_dir_path} ({len(csv_files)} CSV files found). Skipping download.")
        logger.info("To force re-download, use --force-download flag or set force_download=True")
        return

    if csv_files and force_download:
        logger.info(f"Force download requested. Existing {len(csv_files)} CSV files will be overwritten.")

    zip_path = raw_dir_path / "arxiv-scientific-research-papers-dataset.zip"

    logger.info(f"Downloading ArXiv dataset from {dataset_url}")
    logger.info(f"Target directory: {raw_dir_path}")

    try:
        # Download using requests
        response = requests.get(dataset_url, stream=True, timeout=300)
        response.raise_for_status()

        # Save to zip file
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                            logger.info(
                                f"Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)"
                            )

        logger.info(f"Download complete. File size: {zip_path.stat().st_size / (1024*1024):.1f} MB")

        # Extract zip file
        logger.info(f"Extracting {zip_path.name} to {raw_dir_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir_path)
        logger.info("Extraction complete")

        # Remove zip file
        zip_path.unlink()
        logger.info(f"Removed zip file: {zip_path}")

        # Verify extraction by checking for CSV files
        extracted_csv_files = list(raw_dir_path.glob("*.csv"))
        if not extracted_csv_files:
            logger.warning(f"No CSV files found after extraction. Contents of {raw_dir_path}:")
            for item in raw_dir_path.iterdir():
                logger.warning(f"  - {item.name} ({'directory' if item.is_dir() else 'file'})")
        else:
            logger.info(f"Successfully extracted {len(extracted_csv_files)} CSV files")

    except requests.RequestException as e:
        logger.error(f"Failed to download dataset: {e}")
        if zip_path.exists():
            zip_path.unlink()
        raise
    except zipfile.BadZipFile as e:
        logger.error(f"Downloaded file is not a valid zip file: {e}")
        if zip_path.exists():
            zip_path.unlink()
        raise
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        if zip_path.exists():
            zip_path.unlink()
        raise


def preprocess_data(
    raw_dir: str = typer.Argument(..., help="Path to raw data directory"),
    processed_dir: str = typer.Argument(..., help="Path to processed data directory"),
    train_split: float = typer.Option(0.7, help="Fraction of data to use for training"),
    val_split: float = typer.Option(0.15, help="Fraction of data to use for validation"),
    test_split: float = typer.Option(0.15, help="Fraction of data to use for testing"),
    seed: int = typer.Option(42, help="Random seed for train/val/test split"),
    num_categories: int = typer.Option(5, help="Number of categories to keep (top N most frequent)"),
    download: bool = typer.Option(False, help="Download dataset if not present in raw_dir"),
    force_download: bool = typer.Option(False, help="Force re-download even if data exists"),
) -> None:
    """Process raw ArXiv data and save it to processed directory.

    Loads ArXiv dataset from raw_dir, processes text (title + abstract),
    filters to top N categories, encodes categories as labels, splits into train/val/test,
    and saves to processed_dir.

    Optionally downloads the dataset if it's not present (use --download flag).

    Args:
        raw_dir: Path to directory containing raw CSV files.
        processed_dir: Path to directory where processed data will be saved.
        train_split: Fraction of data to use for training (default: 0.7).
        val_split: Fraction of data to use for validation (default: 0.15).
        test_split: Fraction of data to use for testing (default: 0.15).
        seed: Random seed for reproducible train/val/test split (default: 42).
        num_categories: Number of categories to keep - top N most frequent (default: 5).
        download: If True, download dataset if not present in raw_dir (default: False).
        force_download: If True, force re-download even if data exists (default: False).

    Raises:
        FileNotFoundError: If no CSV file is found in raw_dir.
        ValueError: If train_split + val_split + test_split != 1.0.
    """
    # Extract defaults from typer OptionInfo if called directly (not via CLI)
    train_split = _extract_typer_default(train_split, 0.7)
    val_split = _extract_typer_default(val_split, 0.15)
    test_split = _extract_typer_default(test_split, 0.15)
    seed = _extract_typer_default(seed, 42)
    num_categories = _extract_typer_default(num_categories, 5)
    download = _extract_typer_default(download, False)
    force_download = _extract_typer_default(force_download, False)

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    # Check for existing CSV files
    csv_files = list(raw_dir.glob("*.csv"))

    # Download data if requested
    if download:
        logger.info("Downloading dataset...")
        download_data(raw_dir=str(raw_dir), force_download=force_download)
        # Re-check for CSV files after download
        csv_files = list(raw_dir.glob("*.csv"))

    # Validate splits sum to 1.0
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError(
            f"train_split + val_split + test_split must equal 1.0, got {train_split + val_split + test_split}"
        )

    # Create processed directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Find CSV file in raw directory (should exist after download if download was requested)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {raw_dir}. "
            "Use --download flag to download the dataset, or ensure data is present in raw_dir."
        )
    csv_file = csv_files[0]  # Use first CSV file found
    logger.info(f"Using CSV file: {csv_file}")

    # Load and process data
    texts = []
    categories = []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Combine title and abstract (support both "abstract" and "summary" column names)
            title = row.get("title", "").strip()
            abstract = row.get("abstract", row.get("summary", "")).strip()
            text = f"{title} {abstract}".strip()

            # Get category (support both "category" and "categories" column names)
            category = row.get("categories", row.get("category", "")).strip()
            if not category:
                continue  # Skip rows without category

            if text:  # Only add if we have text
                texts.append(text)
                categories.append(category)

    # Filter to top N categories (exactly N, no "OTHER")
    category_counts = Counter(categories)
    top_categories = [cat for cat, _ in category_counts.most_common(num_categories)]

    # Filter data to only include samples from top N categories
    filtered_texts = []
    filtered_categories = []
    for text, cat in zip(texts, categories):
        if cat in top_categories:
            filtered_texts.append(text)
            filtered_categories.append(cat)

    logger.info(
        f"Filtered dataset from {len(texts)} to {len(filtered_texts)} samples (keeping only top {num_categories} categories)"
    )
    logger.info(f"Top {num_categories} categories: {top_categories}")

    # Encode categories as integer labels (0 to num_categories-1)
    category_to_label = {cat: idx for idx, cat in enumerate(sorted(top_categories))}
    labels = [category_to_label[cat] for cat in filtered_categories]

    # Split into train, validation, and test
    random.seed(seed)
    indices = list(range(len(filtered_texts)))
    random.shuffle(indices)

    train_end = int(len(indices) * train_split)
    val_end = train_end + int(len(indices) * val_split)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_texts = [filtered_texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_texts = [filtered_texts[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    test_texts = [filtered_texts[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    # Save processed data
    with open(processed_dir / "train_texts.json", "w", encoding="utf-8") as f:
        json.dump(train_texts, f, ensure_ascii=False)
    torch.save(torch.tensor(train_labels, dtype=torch.long), processed_dir / "train_labels.pt")

    with open(processed_dir / "val_texts.json", "w", encoding="utf-8") as f:
        json.dump(val_texts, f, ensure_ascii=False)
    torch.save(torch.tensor(val_labels, dtype=torch.long), processed_dir / "val_labels.pt")

    with open(processed_dir / "test_texts.json", "w", encoding="utf-8") as f:
        json.dump(test_texts, f, ensure_ascii=False)
    torch.save(torch.tensor(test_labels, dtype=torch.long), processed_dir / "test_labels.pt")

    # Save category mapping
    with open(processed_dir / "category_mapping.json", "w", encoding="utf-8") as f:
        json.dump(category_to_label, f, ensure_ascii=False)

    logger.info(f"Processed data saved to {processed_dir}")
    logger.info(f"Training set size: {len(train_texts)}")
    logger.info(f"Validation set size: {len(val_texts)}")
    logger.info(f"Test set size: {len(test_texts)}")
    logger.info(f"Number of categories: {len(top_categories)}")


class ArXivDataset(torch.utils.data.Dataset):
    """Dataset for ArXiv papers.

    Attributes:
        texts: List of paper texts (title + abstract).
        labels: Tensor of integer class labels.
    """

    def __init__(self, texts: list[str], labels: torch.Tensor) -> None:
        """Initialize ArXiv dataset.

        Args:
            texts: List of paper texts (title + abstract).
            labels: Tensor of integer class labels.
        """
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (text, label).
        """
        return self.texts[idx], self.labels[idx]


def arxiv_dataset(
    max_categories: int = 5,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train, validation, and test datasets for ArXiv papers.

    Loads processed data from data/processed directory or GCS mounted filesystem.
    Checks /gcs/ first (for Vertex AI), then falls back to local data/processed.

    The data should already be filtered to max_categories during preprocessing.
    If data has more categories than max_categories, remaps labels to top N categories.

    Args:
        max_categories: Maximum number of categories to keep (default: 5).
                       If data has more, keeps top (max_categories-1) and maps rest to label (max_categories-1).

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).

    Raises:
        FileNotFoundError: If processed data files are not found locally or in GCS.
    """
    # Try GCS mounted filesystem first (for Vertex AI), then local
    # Check regional bucket first (europe-west1), then multi-region, then local
    gcs_regional_dir = Path("/gcs/mlops_project_data_bucket1-europe-west1/data/processed")
    gcs_processed_dir = Path("/gcs/mlops_project_data_bucket1/data/processed")
    local_processed_dir = Path("data/processed")

    if gcs_regional_dir.exists() and (gcs_regional_dir / "train_texts.json").exists():
        processed_dir = gcs_regional_dir
    elif gcs_processed_dir.exists() and (gcs_processed_dir / "train_texts.json").exists():
        processed_dir = gcs_processed_dir
    else:
        processed_dir = local_processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Required files
        required_files = [
            "train_texts.json",
            "train_labels.pt",
            "test_texts.json",
            "test_labels.pt",
            "category_mapping.json",
        ]

        # Check if files exist locally
        missing_files = [f for f in required_files if not (processed_dir / f).exists()]

        if missing_files:
            raise FileNotFoundError(
                f"Required data files are missing: {missing_files}. "
                "Please ensure data is available in data/processed/ or in GCS mounted paths."
            )

    # Load the data files
    with open(processed_dir / "train_texts.json", "r", encoding="utf-8") as f:
        train_texts = json.load(f)
    train_labels = torch.load(processed_dir / "train_labels.pt").long()

    # Load validation set if it exists, otherwise create empty dataset
    val_texts = []
    val_labels = torch.tensor([], dtype=torch.long)
    val_file = processed_dir / "val_texts.json"
    if val_file.exists():
        with open(val_file, "r", encoding="utf-8") as f:
            val_texts = json.load(f)
        val_labels = torch.load(processed_dir / "val_labels.pt").long()

    with open(processed_dir / "test_texts.json", "r", encoding="utf-8") as f:
        test_texts = json.load(f)
    test_labels = torch.load(processed_dir / "test_labels.pt").long()

    # Remap labels if we have more categories than max_categories
    all_labels = torch.cat([train_labels, test_labels])
    if len(val_labels) > 0:
        all_labels = torch.cat([all_labels, val_labels])
    unique_labels = torch.unique(all_labels)

    if len(unique_labels) > max_categories:
        # Count label frequencies in training set
        label_counts = Counter(train_labels.tolist())
        # Get top (max_categories-1) most frequent labels (returns list of (label, count) tuples)
        top_label_counts = label_counts.most_common(max_categories - 1)
        top_labels = [label for label, _ in top_label_counts]
        # Create mapping: top labels map to 0..(max_categories-2), others map to max_categories-1
        label_remap = {old_label: new_idx for new_idx, old_label in enumerate(top_labels)}
        other_label = max_categories - 1

        def remap_labels(labels: torch.Tensor) -> torch.Tensor:
            remapped = torch.zeros_like(labels)
            for old_label in unique_labels:
                old_val = old_label.item()
                new_label = label_remap.get(old_val, other_label)
                remapped[labels == old_label] = new_label
            return remapped.long()

        train_labels = remap_labels(train_labels)
        if len(val_labels) > 0:
            val_labels = remap_labels(val_labels)
        test_labels = remap_labels(test_labels)

    train_set = ArXivDataset(train_texts, train_labels)
    val_set = (
        ArXivDataset(val_texts, val_labels)
        if len(val_texts) > 0
        else ArXivDataset([], torch.tensor([], dtype=torch.long))
    )
    test_set = ArXivDataset(test_texts, test_labels)
    return train_set, val_set, test_set


# CLI app for both download and preprocess commands
app = typer.Typer(help="ArXiv dataset download and preprocessing utilities")


@app.command()
def download(
    raw_dir: str = typer.Option("data/raw", help="Path to raw data directory"),
    force: bool = typer.Option(False, "--force", help="Force re-download even if data exists"),
) -> None:
    """Download ArXiv dataset from Kaggle.

    Downloads the dataset and extracts it to the raw data directory.
    Skips download if CSV files already exist (unless --force is used).

    Examples:
        # Download to default location (data/raw)
        uv run src/pname/data.py download

        # Download to custom location
        uv run src/pname/data.py download --raw-dir /path/to/data/raw

        # Force re-download
        uv run src/pname/data.py download --force
    """
    download_data(raw_dir=raw_dir, force_download=force)


@app.command()
def preprocess(
    raw_dir: str = typer.Argument("data/raw", help="Path to raw data directory"),
    processed_dir: str = typer.Argument("data/processed", help="Path to processed data directory"),
    train_split: float = typer.Option(0.7, help="Fraction of data to use for training"),
    val_split: float = typer.Option(0.15, help="Fraction of data to use for validation"),
    test_split: float = typer.Option(0.15, help="Fraction of data to use for testing"),
    seed: int = typer.Option(42, help="Random seed for train/val/test split"),
    num_categories: int = typer.Option(5, help="Number of categories to keep (top N most frequent)"),
    download: bool = typer.Option(False, help="Download dataset if not present in raw_dir"),
    force_download: bool = typer.Option(False, help="Force re-download even if data exists"),
) -> None:
    """Preprocess raw ArXiv data and save it to processed directory.

    This is an alias for the preprocess_data function with a cleaner CLI interface.

    Examples:
        # Preprocess with default settings
        uv run src/pname/data.py preprocess

        # Preprocess with automatic download if data missing
        uv run src/pname/data.py preprocess --download

        # Preprocess with custom splits
        uv run src/pname/data.py preprocess --train-split 0.8 --val-split 0.1 --test-split 0.1
    """
    preprocess_data(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        num_categories=num_categories,
        download=download,
        force_download=force_download,
    )


if __name__ == "__main__":
    # Support both old-style (direct preprocess_data call) and new-style (app commands)
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ["download", "preprocess"]:
        # New-style: use app commands
        app()
    else:
        # Old-style: backward compatibility - call preprocess_data directly
        # This allows: uv run src/pname/data.py data/raw data/processed
        typer.run(preprocess_data)
