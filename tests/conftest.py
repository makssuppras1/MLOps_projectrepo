import warnings

# Silence noisy DeprecationWarning from Evidently's legacy utils
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="evidently.*",
)
