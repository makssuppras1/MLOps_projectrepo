#!/usr/bin/env python3
"""Download model from wandb artifacts.

This script downloads the latest TF-IDF model artifact from wandb.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import wandb

# Load environment variables
load_dotenv()


def download_latest_model(
    project: str = None,
    artifact_type: str = "model",
    model_name_pattern: str = "model-tfidf",
    output_dir: str = ".",
) -> Path:
    """Download the latest model artifact from wandb.

    Args:
        project: Wandb project name. Defaults to WANDB_PROJECT env var or "pname-arxiv-classifier".
        artifact_type: Type of artifact to download. Defaults to "model".
        model_name_pattern: Pattern to match artifact names. Defaults to "model-tfidf".
        output_dir: Directory to save the downloaded model. Defaults to current directory.

    Returns:
        Path to the downloaded model file.
    """
    # Get project name
    if project is None:
        project = os.getenv("WANDB_PROJECT", "pname-arxiv-classifier")

    # Initialize wandb API
    api = wandb.Api()

    print(f"Searching for artifacts in project: {project}")
    print(f"Looking for artifacts matching pattern: {model_name_pattern}*")

    # List all artifacts in the project
    try:
        # Get all runs in the project first
        runs = list(api.runs(project))
        matching_artifacts = []

        print(f"Searching through {len(runs)} runs...")

        # Search through runs to find artifacts
        for run in runs:
            try:
                # Get artifacts for this run
                for artifact_collection in run.logged_artifacts():
                    artifact_name = artifact_collection.name
                    artifact_type = artifact_collection.type

                    # Match by type "model" or by name pattern
                    if artifact_type == "model" or model_name_pattern in artifact_name:
                        # Try to get the artifact
                        try:
                            # Format: entity/project/artifact_name:version
                            artifact = api.artifact(f"{run.entity}/{project}/{artifact_name}:latest")
                            matching_artifacts.append((artifact, run))
                        except Exception:
                            # Try without latest
                            try:
                                artifact = api.artifact(f"{run.entity}/{project}/{artifact_name}")
                                matching_artifacts.append((artifact, run))
                            except Exception as e:
                                print(f"  Could not load artifact {artifact_name}: {e}")
                                continue
            except Exception:
                continue

        # Extract just artifacts and remove duplicates
        seen = set()
        unique_artifacts = []
        for artifact, run in matching_artifacts:
            if artifact.name not in seen:
                seen.add(artifact.name)
                unique_artifacts.append(artifact)
        matching_artifacts = unique_artifacts

    except Exception as e:
        print(f"Error accessing wandb project: {e}")
        print("Make sure WANDB_API_KEY is set in your .env file")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if not matching_artifacts:
        print(f"No artifacts found matching pattern '{model_name_pattern}*'")
        print("Available artifact types:")
        try:
            all_artifacts = list(api.artifacts(type=artifact_type, project=project))
            if all_artifacts:
                print(f"  Found {len(all_artifacts)} artifacts of type '{artifact_type}'")
                print("  Sample artifact names:")
                for art in all_artifacts[:5]:
                    print(f"    - {art.name}")
            else:
                print(f"  No artifacts of type '{artifact_type}' found")
        except Exception:
            pass
        sys.exit(1)

    # Sort by creation time (newest first)
    matching_artifacts.sort(key=lambda x: x.created_at, reverse=True)
    latest_artifact = matching_artifacts[0]

    print(f"\nFound {len(matching_artifacts)} matching artifacts")
    print(f"Downloading latest: {latest_artifact.name}")
    print(f"  Created: {latest_artifact.created_at}")
    print(f"  Version: {latest_artifact.version}")

    # Download artifact to a temporary directory first
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = latest_artifact.download(root=temp_dir)
        print(f"\nArtifact downloaded to temporary directory: {artifact_dir}")

        # Find the model file in the downloaded artifact
        artifact_path = Path(artifact_dir)
        model_files = (
            list(artifact_path.glob("*.pkl")) + list(artifact_path.glob("*.pt")) + list(artifact_path.glob("*.pth"))
        )

        # Also check subdirectories
        if not model_files:
            model_files = (
                list(artifact_path.rglob("*.pkl"))
                + list(artifact_path.rglob("*.pt"))
                + list(artifact_path.rglob("*.pth"))
            )

        if not model_files:
            print(f"Warning: No model files (.pkl, .pt, or .pth) found in {artifact_dir}")
            print("Contents of artifact directory:")
            for item in artifact_path.iterdir():
                if item.is_file():
                    print(f"  - {item.name}")
                elif item.is_dir():
                    print(f"  - {item.name}/ (directory)")
                    for subitem in item.iterdir():
                        print(f"    - {subitem.name}")
            sys.exit(1)

        model_file = model_files[0]
        print(f"Model file found: {model_file}")

        # Copy to final output location
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        final_model_path = output_path / model_file.name

        import shutil

        shutil.copy2(model_file, final_model_path)
        print(f"Model copied to: {final_model_path}")

    return final_model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download model from wandb")
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Wandb project name (defaults to WANDB_PROJECT env var)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="model-tfidf",
        help="Pattern to match artifact names (default: model-tfidf)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="trained_model.pkl",
        help="Final name for the model file (default: trained_model.pkl)",
    )

    args = parser.parse_args()

    # Check for wandb API key
    if not os.getenv("WANDB_API_KEY"):
        print("Error: WANDB_API_KEY not found in environment")
        print("Please set it in your .env file or export it:")
        print("  export WANDB_API_KEY=your_api_key")
        sys.exit(1)

    try:
        model_path = download_latest_model(
            project=args.project,
            model_name_pattern=args.pattern,
            output_dir=args.output,
        )

        # Rename if requested
        if args.name and model_path.name != args.name:
            final_path = model_path.parent / args.name
            model_path.rename(final_path)
            print(f"Model renamed to: {final_path}")
            model_path = final_path

        print(f"\n✓ Success! Model downloaded to: {model_path.absolute()}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
