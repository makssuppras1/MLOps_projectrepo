"""Visualize feature importance for TF-IDF + XGBoost model."""

from pathlib import Path

import matplotlib.pyplot as plt

from pname.model_tfidf import TFIDFXGBoostModel


def visualize_feature_importance(
    model_path: str,
    top_n: int = 30,
    figure_name: str = "feature_importance.png",
) -> None:
    """Visualize top feature importances.

    Args:
        model_path: Path to saved model.
        top_n: Number of top features to visualize.
        figure_name: Output figure filename.
    """
    # Load model
    model = TFIDFXGBoostModel.load(model_path)

    # Get feature importances
    features = model.get_feature_importance(top_n=top_n)

    if not features:
        print("No feature importances available")
        return

    # Extract names and importances
    feature_names = [f[0] for f in features]
    importances = [f[1] for f in features]

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, max(8, len(features) * 0.3)))
    y_pos = range(len(feature_names))

    ax.barh(y_pos, importances, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # Top features at top
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Most Important Features")
    plt.tight_layout()

    # Save figure
    fig_path = Path(figure_name)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Feature importance plot saved to {fig_path}")

    plt.close(fig)


if __name__ == "__main__":
    import typer

    typer.run(visualize_feature_importance)
