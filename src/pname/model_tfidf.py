"""TF-IDF + XGBoost model for fast, explainable text classification."""

import pickle
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


class TFIDFXGBoostModel:
    """TF-IDF vectorization + XGBoost classifier for text classification.

    Fast, explainable, and perfect for MLOps projects with hyperparameter tuning.
    """

    def __init__(self, model_cfg: DictConfig = None) -> None:
        """Initialize TF-IDF + XGBoost model.

        Args:
            model_cfg: Model configuration with hyperparameters.
        """
        if model_cfg is None:
            model_cfg = DictConfig(
                {
                    "num_labels": 5,
                    "max_features": 5000,
                    "stop_words": "english",
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                }
            )

        self.num_labels = model_cfg.get("num_labels", 5)
        self.model_cfg = model_cfg

        # TF-IDF vectorizer
        # Handle ngram_range - can be tuple, list, or separate min/max values
        ngram_range = model_cfg.get("ngram_range", None)
        if ngram_range is None:
            # Use separate min/max if ngram_range not specified
            ngram_min = model_cfg.get("ngram_range_min", 1)
            ngram_max = model_cfg.get("ngram_range_max", 2)
            ngram_range = (int(ngram_min), int(ngram_max))
        elif isinstance(ngram_range, list):
            # Convert list to tuple (Hydra loads YAML lists as Python lists)
            ngram_range = tuple(int(x) for x in ngram_range)
        elif isinstance(ngram_range, tuple):
            # Ensure tuple contains integers
            ngram_range = tuple(int(x) for x in ngram_range)
        else:
            # Fallback if it's something else
            ngram_range = (1, 2)

        # Final safety check: ensure ngram_range is a tuple of integers (required by sklearn)
        if not isinstance(ngram_range, tuple):
            ngram_range = tuple(int(x) for x in ngram_range) if hasattr(ngram_range, "__iter__") else (1, 2)

        self.vectorizer = TfidfVectorizer(
            max_features=model_cfg.get("max_features", 5000),
            stop_words=model_cfg.get("stop_words", "english"),
            ngram_range=ngram_range,
            min_df=model_cfg.get("min_df", 2),  # Minimum document frequency
            max_df=model_cfg.get("max_df", 0.95),  # Maximum document frequency
        )

        # XGBoost classifier
        # For XGBoost 1.6+, early_stopping_rounds must be set in constructor, NOT in fit()
        early_stopping_rounds = model_cfg.get("early_stopping_rounds", None)
        self.classifier = XGBClassifier(
            max_depth=model_cfg.get("max_depth", 6),
            learning_rate=model_cfg.get("learning_rate", 0.1),
            n_estimators=model_cfg.get("n_estimators", 100),
            subsample=model_cfg.get("subsample", 0.8),
            colsample_bytree=model_cfg.get("colsample_bytree", 0.8),
            reg_alpha=model_cfg.get("reg_alpha", 0.0),  # L1 regularization
            reg_lambda=model_cfg.get("reg_lambda", 1.0),  # L2 regularization
            objective="multi:softprob",
            num_class=self.num_labels,
            random_state=model_cfg.get("random_state", 42),  # Use config seed for reproducibility
            n_jobs=-1,  # Use all CPU cores
            eval_metric="mlogloss",
            early_stopping_rounds=early_stopping_rounds,  # Set in constructor for XGBoost 1.6+
        )

        # Wrap vectorizer and classifier in sklearn Pipeline for unified serialization
        # This ensures preprocessing is always applied during inference
        self.pipeline = Pipeline(
            [
                ("vectorizer", self.vectorizer),
                ("classifier", self.classifier),
            ]
        )

    def fit(
        self, texts: list[str], labels: list[int], val_texts: list[str] = None, val_labels: list[int] = None
    ) -> None:
        """Train the model.

        Args:
            texts: List of text strings to train on.
            labels: List of integer labels.
            val_texts: Optional validation texts for early stopping.
            val_labels: Optional validation labels for early stopping.
        """
        # Vectorize texts
        print(f"Vectorizing {len(texts)} training texts...")
        X = self.vectorizer.fit_transform(texts)
        print(f"Training matrix shape: {X.shape}")

        # Train classifier with early stopping if validation set provided
        # Note: Early stopping requires eval_set which doesn't work with Pipeline.fit(),
        # so we fit components separately, then update the pipeline
        if val_texts is not None and val_labels is not None:
            print(f"Vectorizing {len(val_texts)} validation texts...")
            X_val = self.vectorizer.transform(val_texts)
            print(f"Validation matrix shape: {X_val.shape}")
            # Ensure early_stopping_rounds is set if we have validation data
            early_stopping_rounds = self.model_cfg.get("early_stopping_rounds", 20)
            if self.classifier.get_params().get("early_stopping_rounds") is None:
                self.classifier.set_params(early_stopping_rounds=early_stopping_rounds)
            # CRITICAL: Do NOT pass early_stopping_rounds to fit() - it's already set in constructor
            try:
                self.classifier.fit(
                    X,
                    labels,
                    eval_set=[(X_val, val_labels)],
                    verbose=True,  # Enable verbose to see progress
                )
            except Exception as e:
                import traceback

                print(f"ERROR in XGBoost fit: {e}")
                print(traceback.format_exc())
                raise
        else:
            # Train without early stopping - disable early stopping if it was set
            if self.classifier.get_params().get("early_stopping_rounds") is not None:
                self.classifier.set_params(early_stopping_rounds=None)
            try:
                self.classifier.fit(X, labels)
            except Exception as e:
                import traceback

                print(f"ERROR in XGBoost fit: {e}")
                print(traceback.format_exc())
                raise

        # Update pipeline with fitted components
        # This ensures pipeline.predict() works correctly after fitting
        self.pipeline.steps[1] = ("classifier", self.classifier)

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict class labels.

        Args:
            texts: List of text strings to predict.

        Returns:
            Array of predicted class labels.
        """
        # Use pipeline to ensure preprocessing is always applied
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict class probabilities.

        Args:
            texts: List of text strings to predict.

        Returns:
            Array of predicted probabilities [n_samples, n_classes].
        """
        # Use pipeline to ensure preprocessing is always applied
        return self.pipeline.predict_proba(texts)

    def get_feature_importance(self, top_n: int = 20) -> list[tuple[str, float]]:
        """Get top feature importances for explainability.

        Args:
            top_n: Number of top features to return.

        Returns:
            List of (feature_name, importance) tuples.
        """
        if not hasattr(self.classifier, "feature_importances_"):
            return []

        # Get feature names from vectorizer
        feature_names = self.vectorizer.get_feature_names_out()

        # Get importances
        importances = self.classifier.feature_importances_

        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]

        return [(feature_names[i], float(importances[i])) for i in indices]

    def save(self, path: str) -> None:
        """Save model and vectorizer.

        Args:
            path: Path to save the model.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save pipeline (which contains both vectorizer and classifier) for unified serialization
        # Also save individual components for backward compatibility
        model_dict = {
            "pipeline": self.pipeline,  # Primary: unified pipeline
            "vectorizer": self.vectorizer,  # Backward compatibility
            "classifier": self.classifier,  # Backward compatibility
            "config": self.model_cfg,
        }

        with open(path, "wb") as f:
            pickle.dump(model_dict, f)

    @classmethod
    def load(cls, path: str) -> "TFIDFXGBoostModel":
        """Load model and vectorizer.

        Args:
            path: Path to load the model from.

        Returns:
            Loaded model instance.
        """
        with open(path, "rb") as f:
            model_dict = pickle.load(f)

        # Create instance
        instance = cls(model_cfg=model_dict["config"])

        # Load pipeline if available (new format), otherwise reconstruct from components (backward compatibility)
        if "pipeline" in model_dict:
            instance.pipeline = model_dict["pipeline"]
            # Extract components from pipeline for backward compatibility
            instance.vectorizer = instance.pipeline.named_steps["vectorizer"]
            instance.classifier = instance.pipeline.named_steps["classifier"]
        else:
            # Backward compatibility: reconstruct pipeline from individual components
            instance.vectorizer = model_dict["vectorizer"]
            instance.classifier = model_dict["classifier"]
            instance.pipeline = Pipeline(
                [
                    ("vectorizer", instance.vectorizer),
                    ("classifier", instance.classifier),
                ]
            )

        return instance
