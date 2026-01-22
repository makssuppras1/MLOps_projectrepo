"""Collect current data from API logs for drift monitoring."""

from pathlib import Path

import pandas as pd
from loguru import logger

from monitoring.drift_monitor import CURRENT_DATA_PATH

# Default log file path (matches app/main.py)
LOG_FILE = Path("monitoring/api_requests.csv")


def collect_current_data(log_file: Path = LOG_FILE, output_path: Path = CURRENT_DATA_PATH) -> pd.DataFrame:
    """Collect current data from API request logs.

    Args:
        log_file: Path to CSV log file with API requests.
        output_path: Path to save current data parquet.

    Returns:
        DataFrame with extracted features.
    """
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    logger.info(f"Loading current data from: {log_file}")
    df = pd.read_csv(log_file)

    # Drop time and output columns, keep only input feature columns for drift detection
    if "time" in df.columns:
        df = df.drop(columns=["time"])
    if "predicted_class" in df.columns:
        df = df.drop(columns=["predicted_class"])
    if "confidence" in df.columns:
        df = df.drop(columns=["confidence"])

    # Ensure we have the right columns
    required_cols = ["text_length", "word_count"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")

    df = df[required_cols]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Current data saved: {len(df)} rows to {output_path}")

    return df


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def collect(log_file: str = str(LOG_FILE)) -> None:
        """Collect current data from API logs."""
        collect_current_data(Path(log_file))

    app()
