from pathlib import Path

import torch
import typer


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images to have mean 0 and std 1."""
    mean = images.mean()
    std = images.std()
    return (images - mean) / std


def preprocess_data(
    raw_dir: str = typer.Argument(..., help="Path to raw data directory"),
    processed_dir: str = typer.Argument(..., help="Path to processed data directory"),
) -> None:
    """Process raw data and save it to processed directory.

    Loads corruptmnist files from raw_dir, combines them into single tensors,
    normalizes the images (using training statistics for both train and test),
    and saves to processed_dir.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    # Create processed directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load training data (6 files: train_images_0.pt through train_images_5.pt)
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(raw_dir / f"train_images_{i}.pt"))
        train_target.append(torch.load(raw_dir / f"train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # Load test data
    test_images: torch.Tensor = torch.load(raw_dir / "test_images.pt")
    test_target: torch.Tensor = torch.load(raw_dir / "test_target.pt")

    # Convert to proper types and add channel dimension if needed
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    # Normalize using training statistics (standard practice)
    # Calculate mean and std from training data
    train_mean = train_images.mean()
    train_std = train_images.std()

    # Normalize both train and test using training statistics
    train_images = (train_images - train_mean) / train_std
    test_images = (test_images - train_mean) / train_std

    # Save processed data
    torch.save(train_images, processed_dir / "train_images.pt")
    torch.save(train_target, processed_dir / "train_target.pt")
    torch.save(test_images, processed_dir / "test_images.pt")
    torch.save(test_target, processed_dir / "test_target.pt")

    print(f"Processed data saved to {processed_dir}")
    print(f"Training set size: {len(train_images)}")
    print(f"Test set size: {len(test_images)}")
    print(f"Normalized - Mean: {train_images.mean():.6f}, Std: {train_images.std():.6f}")


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    # Ensure proper dtypes to avoid NNPack errors
    train_images = train_images.float()
    test_images = test_images.float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)
