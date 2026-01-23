"""TF-IDF + XGBoost model for fast, explainable text classification."""

import os
import pickle
from pathlib import Path

import numpy as np
from loguru import logger
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
        # Use n_jobs=1 to avoid threading issues that can cause crashes under emulation
        # This is especially important when running AMD64 images on ARM64 hosts
        n_jobs = model_cfg.get("n_jobs", 1)
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
            n_jobs=n_jobs,  # Default to 1 to avoid threading issues under emulation
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
        logger.info(f"Vectorizing {len(texts)} training texts...")
        X = self.vectorizer.fit_transform(texts)
        logger.info(f"Training matrix shape: {X.shape}")

        # Train classifier with early stopping if validation set provided
        # Note: Early stopping requires eval_set which doesn't work with Pipeline.fit(),
        # so we fit components separately, then update the pipeline
        if val_texts is not None and val_labels is not None:
            logger.info(f"Vectorizing {len(val_texts)} validation texts...")
            X_val = self.vectorizer.transform(val_texts)
            logger.info(f"Validation matrix shape: {X_val.shape}")
            # Ensure early_stopping_rounds is set if we have validation data
            n_estimators = self.classifier.get_params().get("n_estimators", 100)
            early_stopping_rounds = self.model_cfg.get("early_stopping_rounds", 20)

            # CRITICAL: XGBoost requires minimum n_estimators for stable operation
            # Very small values (1-2) can cause crashes, especially with validation sets
            # Use at least 10 for reliable operation
            MIN_ESTIMATORS = 10
            if n_estimators < MIN_ESTIMATORS:
                logger.warning(
                    f"n_estimators ({n_estimators}) < {MIN_ESTIMATORS}. Increasing to {MIN_ESTIMATORS} for stability."
                )
                n_estimators = MIN_ESTIMATORS
                self.classifier.set_params(n_estimators=MIN_ESTIMATORS)

            # CRITICAL: If n_estimators is too small, disable early stopping or adjust it
            if n_estimators <= early_stopping_rounds:
                logger.warning(
                    f"n_estimators ({n_estimators}) <= early_stopping_rounds ({early_stopping_rounds}). Disabling early stopping."
                )
                self.classifier.set_params(early_stopping_rounds=None)
                early_stopping_rounds = None
            else:
                if self.classifier.get_params().get("early_stopping_rounds") is None:
                    self.classifier.set_params(early_stopping_rounds=early_stopping_rounds)

            # CRITICAL: Do NOT pass early_stopping_rounds to fit() - it's already set in constructor
            logger.info(f"Starting XGBoost training with {X.shape[0]} samples and {X.shape[1]} features...")
            logger.info(f"Early stopping rounds: {early_stopping_rounds}")
            logger.info(
                f"XGBoost parameters: n_estimators={n_estimators}, max_depth={self.classifier.get_params().get('max_depth')}"
            )
            # Force flush logs before potentially long-running operation
            import sys

            sys.stdout.flush()
            sys.stderr.flush()
            # Set environment variable to ensure XGBoost output is visible
            os.environ["XGBOOST_VERBOSE"] = "1"

            # Enable Python's faulthandler to catch segfaults and print stack traces
            # Write to both stderr and a file for debugging
            import faulthandler

            faulthandler.enable(file=sys.stderr, all_threads=True)

            # Also enable faulthandler dump on signal
            import signal

            faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

            try:
                import time

                fit_start = time.time()
                logger.info("Calling XGBoost fit()...")
                logger.info(f"XGBoost n_jobs: {self.classifier.get_params().get('n_jobs')}")
                logger.info(f"XGBoost n_estimators: {n_estimators}")
                logger.info(f"XGBoost early_stopping_rounds: {early_stopping_rounds}")
                sys.stdout.flush()
                sys.stderr.flush()

                # Determine eval_set based on early stopping
                if early_stopping_rounds is not None:
                    eval_set = [(X_val, val_labels)]
                    logger.info("Using validation set for early stopping")
                else:
                    eval_set = None
                    logger.info("Training without early stopping (no validation set or n_estimators too small)")

                # Use verbose=True for progress output (XGBoost will print to stderr)
                # This will show progress every iteration
                if eval_set is not None:
                    self.classifier.fit(
                        X,
                        labels,
                        eval_set=eval_set,
                        verbose=True,  # Show progress (prints to stderr)
                    )
                else:
                    # No early stopping, train normally
                    logger.info("About to call XGBoost fit() without eval_set...")
                    sys.stdout.flush()
                    sys.stderr.flush()
                    try:
                        self.classifier.fit(
                            X,
                            labels,
                            verbose=True,  # Show progress (prints to stderr)
                        )
                        logger.info("XGBoost fit() call completed successfully")
                    except Exception as fit_error:
                        logger.error(f"XGBoost fit() raised exception: {fit_error}")
                        import traceback

                        logger.error(traceback.format_exc())
                        sys.stdout.flush()
                        sys.stderr.flush()
                        raise

                # Force flush after fit
                sys.stdout.flush()
                sys.stderr.flush()
                fit_duration = time.time() - fit_start
                logger.info(
                    f"XGBoost training completed successfully in {fit_duration:.2f} seconds ({fit_duration/60:.2f} minutes)"
                )
            except Exception as e:
                import traceback

                logger.error(f"ERROR in XGBoost fit: {e}")
                logger.error(traceback.format_exc())
                sys.stdout.flush()
                sys.stderr.flush()
                # Try to dump faulthandler if available
                try:
                    faulthandler.dump_traceback(sys.stderr, all_threads=True)
                except Exception:
                    pass
                raise
            except BaseException as e:
                # Catch even SystemExit and KeyboardInterrupt to log them
                import traceback

                logger.critical(f"CRITICAL: XGBoost fit() raised BaseException: {e}")
                logger.critical(traceback.format_exc())
                sys.stdout.flush()
                sys.stderr.flush()
                # Try to dump faulthandler if available
                try:
                    faulthandler.dump_traceback(sys.stderr, all_threads=True)
                except Exception:
                    pass
                raise
        else:
            # Train without early stopping - disable early stopping if it was set
            if self.classifier.get_params().get("early_stopping_rounds") is not None:
                self.classifier.set_params(early_stopping_rounds=None)

            # CRITICAL: XGBoost requires minimum n_estimators for stable operation
            # Very small values (1-2) can cause crashes
            # Use at least 10 for reliable operation
            n_estimators = self.classifier.get_params().get("n_estimators", 100)
            MIN_ESTIMATORS = 10
            if n_estimators < MIN_ESTIMATORS:
                logger.warning(
                    f"n_estimators ({n_estimators}) < {MIN_ESTIMATORS}. Increasing to {MIN_ESTIMATORS} for stability."
                )
                n_estimators = MIN_ESTIMATORS
                self.classifier.set_params(n_estimators=MIN_ESTIMATORS)

            logger.info(f"Starting XGBoost training with {X.shape[0]} samples and {X.shape[1]} features...")
            logger.info(
                f"XGBoost parameters: n_estimators={n_estimators}, max_depth={self.classifier.get_params().get('max_depth')}"
            )
            # Force flush logs before potentially long-running operation
            import sys

            sys.stdout.flush()
            sys.stderr.flush()
            # Set environment variable to ensure XGBoost output is visible
            os.environ["XGBOOST_VERBOSE"] = "1"

            # Enable Python's faulthandler to catch segfaults and print stack traces
            import faulthandler

            faulthandler.enable(file=sys.stderr, all_threads=True)

            try:
                import time

                fit_start = time.time()
                logger.info("Calling XGBoost fit()...")
                logger.info(f"XGBoost n_jobs: {self.classifier.get_params().get('n_jobs')}")
                logger.info(f"XGBoost n_estimators: {n_estimators}")
                logger.info("About to call XGBoost fit() without validation set...")
                sys.stdout.flush()
                sys.stderr.flush()
                try:
                    self.classifier.fit(X, labels, verbose=True)  # Show progress
                    logger.info("XGBoost fit() call completed successfully")
                except Exception as fit_error:
                    logger.error(f"XGBoost fit() raised exception: {fit_error}")
                    import traceback

                    logger.error(traceback.format_exc())
                    sys.stdout.flush()
                    sys.stderr.flush()
                    raise
                # Force flush after fit
                sys.stdout.flush()
                sys.stderr.flush()
                fit_duration = time.time() - fit_start
                logger.info(
                    f"XGBoost training completed successfully in {fit_duration:.2f} seconds ({fit_duration/60:.2f} minutes)"
                )
            except Exception as e:
                import traceback

                logger.error(f"ERROR in XGBoost fit: {e}")
                logger.error(traceback.format_exc())
                sys.stdout.flush()
                sys.stderr.flush()
                # Try to dump faulthandler if available
                try:
                    faulthandler.dump_traceback(sys.stderr, all_threads=True)
                except Exception:
                    pass
                raise
            except BaseException as e:
                # Catch even SystemExit and KeyboardInterrupt to log them
                import traceback

                logger.critical(f"CRITICAL: XGBoost fit() raised BaseException: {e}")
                logger.critical(traceback.format_exc())
                sys.stdout.flush()
                sys.stderr.flush()
                # Try to dump faulthandler if available
                try:
                    faulthandler.dump_traceback(sys.stderr, all_threads=True)
                except Exception:
                    pass
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
