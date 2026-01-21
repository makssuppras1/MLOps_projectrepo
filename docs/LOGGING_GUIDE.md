# Logging and Weights & Biases Guide

This guide explains how to use logging and Weights & Biases (wandb) in this project.

## Application Logging with Loguru

The project uses [loguru](https://github.com/Delgan/loguru) for application logging, which provides an easy-to-use logging interface.

### Features

- **Automatic log rotation**: Logs are rotated when they reach 100 MB
- **Compression**: Old log files are compressed with zip
- **Retention**: Logs are kept for 10 days
- **File and console output**: Logs are written to both console (INFO level) and file (DEBUG level)
- **Integration with Hydra**: Logs are saved to the Hydra output directory

### Usage

Simply import and use the logger:

```python
from loguru import logger

logger.info("This is an info message")
logger.debug("This is a debug message (only in file)")
logger.warning("This is a warning")
logger.error("This is an error")
logger.critical("This is a critical error")
```

### Log Levels

- **DEBUG**: Detailed information for debugging (only saved to file)
- **INFO**: General information about program execution
- **WARNING**: Warning messages for potentially problematic situations
- **ERROR**: Error messages for serious problems
- **CRITICAL**: Critical errors that may cause the program to stop

## Weights & Biases Integration

[Weights & Biases](https://wandb.ai) is used for experiment tracking, logging metrics, and hyperparameter optimization.

### Setup

1. **Create a wandb account** at https://wandb.ai
2. **Get your API key** from https://wandb.ai/settings
3. **Create a `.env` file** in the project root:

```bash
WANDB_API_KEY=your-api-key-here
WANDB_PROJECT=corrupt-mnist
WANDB_ENTITY=your-username  # Optional, leave empty for personal account
```

4. **Install dependencies** (already in `pyproject.toml`):

```bash
uv sync
```

5. **Login to wandb** (optional, API key in .env works too):

```bash
uv run wandb login
```

### What Gets Logged

During training, the following are automatically logged to wandb:

- **Hyperparameters**: All configuration parameters from Hydra config
- **Training metrics**: Loss and accuracy at each training step
- **Epoch metrics**: Average loss and accuracy per epoch
- **Sample predictions**: Images with true and predicted labels (first batch of first epoch)
- **Training statistics plot**: Loss and accuracy curves
- **Model artifact**: The trained model is saved as a wandb artifact

### Running Training with Wandb

Simply run training as usual. If `WANDB_API_KEY` is set in your `.env` file, wandb will automatically initialize:

```bash
uv run src/pname/train.py
```

Or with Hydra:

```bash
uv run src/pname/train.py experiment=exp1
```

### Viewing Results

1. Go to https://wandb.ai
2. Navigate to your project (default: `corrupt-mnist`)
3. View runs, compare metrics, and download artifacts

## Hyperparameter Sweeps

Wandb supports hyperparameter optimization through sweeps. A sweep configuration is provided in `configs/sweep.yaml`.

### Creating a Sweep

1. **Create the sweep**:

```bash
uv run wandb sweep configs/sweep.yaml
```

This will output a sweep ID (e.g., `your-entity/corrupt-mnist/abc123`).

2. **Run the sweep agent**:

```bash
uv run wandb agent <sweep-id>
```

You can run multiple agents in parallel to speed up the search:

```bash
# Terminal 1
uv run wandb agent <sweep-id>

# Terminal 2
uv run wandb agent <sweep-id>
```

### Sweep Configuration

The sweep configuration in `configs/sweep.yaml` defines:

- **Method**: Bayesian optimization (can be changed to `grid` or `random`)
- **Metric**: `train/epoch_accuracy` (maximize)
- **Parameters**: Batch size, learning rate, dropout, and model architecture parameters
- **Early termination**: Hyperband for stopping poor runs early

### Customizing the Sweep

Edit `configs/sweep.yaml` to change:

- Search space (parameter ranges)
- Optimization method
- Metric to optimize
- Early termination strategy

## Model Registry

After training, models are logged as wandb artifacts. To use the model registry:

1. **Create a registry** in the wandb web interface:
   - Go to Model Registry tab
   - Click "Create Collection"
   - Name it (e.g., `corrupt_mnist_models`)
   - Set type to `model`

2. **Link artifacts to registry**:
   - In the web interface: Go to an artifact → Click "Link to registry"
   - Or programmatically using the wandb API

3. **Download models from registry**:

```python
import wandb
import torch
from pname.model import MyAwesomeModel

api = wandb.Api()
artifact_name = "your-entity/model-registry/corrupt_mnist_models:latest"
artifact = api.artifact(name=artifact_name)
artifact_dir = artifact.download("downloaded_model")

model = MyAwesomeModel()
model.load_state_dict(torch.load(f"{artifact_dir}/trained_model.pt"))
```

## Docker Integration

To use wandb in Docker containers, pass the API key as an environment variable:

```bash
docker run -e WANDB_API_KEY=<your-api-key> your-image:tag
```

Or use docker-compose:

```yaml
services:
  train:
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
```

## Troubleshooting

### Wandb not logging

- Check that `WANDB_API_KEY` is set in your `.env` file
- Verify the API key is correct: `uv run wandb login`
- Check that you have internet connection

### Logs not appearing in file

- Check that the Hydra output directory exists
- Verify file permissions
- Check disk space (logs are rotated at 100 MB)

### Sweep not working

- Ensure the sweep configuration file path is correct
- Check that the parameter names in `sweep.yaml` match your config structure
- Verify that `train/epoch_accuracy` is being logged (check training script)

## Best Practices

1. **Always use loguru** instead of `print()` statements
2. **Use appropriate log levels**: DEBUG for detailed info, INFO for general info, WARNING/ERROR for problems
3. **Log important events**: Model initialization, data loading, training start/end, errors
4. **Use wandb for experiments**: Log all hyperparameters and metrics
5. **Version your models**: Use wandb artifacts and model registry
6. **Compare runs**: Use wandb's compare feature to understand what works
