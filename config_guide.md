# Configuration Guide

This guide explains how to use Hydra for managing hyperparameters and experiment configurations in this project.

## Overview

This project uses [Hydra](https://hydra.cc/) for configuration management. Hydra allows you to:
- Separate hyperparameters from code
- Version control configurations
- Easily run experiments with different hyperparameters
- Automatically log all hyperparameters with each run
- Override parameters from the command line

## Configuration Structure

The configuration files are organized as follows:

```
configs/
├── config.yaml              # Main configuration file
├── model_conf.yaml          # Model architecture hyperparameters
├── training_conf.yaml       # Training hyperparameters
└── experiment/              # Experiment-specific configs
    ├── null.yaml           # Default (no experiment overrides)
    ├── exp1.yaml           # Experiment 1
    └── exp2.yaml           # Experiment 2
```

### Main Configuration (`config.yaml`)

The main config file uses Hydra's `defaults` system to compose configurations:

```yaml
defaults:
  - model_conf
  - training_conf
  - experiment: null  # Can be overridden with experiment=exp1, etc.
  - _self_
```

This means:
- `model_conf.yaml` and `training_conf.yaml` are always loaded
- `experiment: null` means no experiment overrides by default
- `_self_` allows the main config to define its own parameters

### Model Configuration (`model_conf.yaml`)

Defines the model architecture:

```yaml
model:
  conv1:
    in_channels: 1
    out_channels: 32
    kernel_size: 3
    stride: 1
  # ... more layers
```

### Training Configuration (`training_conf.yaml`)

Defines training hyperparameters and uses Hydra's `instantiate` feature:

```yaml
training:
  seed: 42
  batch_size: 32
  epochs: 10
  optimizer:
    _target_: torch.optim.Adam  # Class to instantiate
    lr: 0.001
  loss_fn:
    _target_: torch.nn.CrossEntropyLoss
```

The `_target_` field tells Hydra which Python class to instantiate. This allows you to change optimizers or loss functions just by editing the config file.

### Experiment Configurations (`experiment/*.yaml`)

Experiment configs can override any parameter from the base configs:

```yaml
# @package _global_
training:
  seed: 123
  batch_size: 64
  epochs: 15
  optimizer:
    lr: 0.0005
```

The `@package _global_` directive makes these parameters merge at the root level.

## Usage

### Basic Training

Run training with the default configuration:

```bash
uv run src/pname/train.py
```

### Using Experiment Configs

Run with a specific experiment configuration:

```bash
uv run src/pname/train.py experiment=exp1
uv run src/pname/train.py experiment=exp2
```

### Command-Line Overrides

Override any parameter from the command line:

```bash
# Override single parameter
uv run src/pname/train.py training.seed=9999

# Override multiple parameters
uv run src/pname/train.py training.seed=9999 training.batch_size=128

# Override nested parameters
uv run src/pname/train.py training.optimizer.lr=0.0001
```

### Adding New Parameters

Add new parameters on the fly (they won't be in the config file, but will be logged):

```bash
uv run src/pname/train.py +training.new_param=42
```

The `+` prefix tells Hydra to add a new parameter that doesn't exist in the config.

### Combining Experiment and Overrides

You can combine experiment configs with command-line overrides:

```bash
uv run src/pname/train.py experiment=exp1 training.epochs=20
```

## Hydra Output

By default, Hydra saves all experiment outputs to:

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/
        │   ├── config.yaml        # Full configuration used
        │   ├── hydra.yaml         # Hydra settings
        │   └── overrides.yaml     # Command-line overrides
        ├── train.log              # Training logs
        └── trained_model.pt       # Saved model (if path is relative)
```

### Important Files

- **`config.yaml`**: Complete configuration used for the run (includes all merged configs)
- **`overrides.yaml`**: Command-line overrides applied
- **`train.log`**: All log output from the training script

## Reproducibility

### Testing Reproducibility

Use the reproducibility tester to compare two runs:

```bash
uv run src/pname/reproducibility_tester.py \
    outputs/2026-01-16/12-15-20 \
    outputs/2026-01-16/12-15-35
```

This script compares:
1. **Configurations**: Verifies all hyperparameters match
2. **Model weights**: Verifies trained models are identical

For true reproducibility, ensure:
- Same random seed (`training.seed`)
- Same hyperparameters
- Same code version
- Same data version

## Creating New Experiments

To create a new experiment:

1. Create a new file in `configs/experiment/`:

```yaml
# configs/experiment/exp3.yaml
# @package _global_
training:
  seed: 456
  batch_size: 128
  epochs: 20
  optimizer:
    lr: 0.0005
```

2. Run it:

```bash
uv run src/pname/train.py experiment=exp3
```

## Using `instantiate` for Complex Objects

The `instantiate` feature allows you to create Python objects directly from config:

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
  weight_decay: 0.0001
```

In code:
```python
from hydra.utils import instantiate

optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
```

This is equivalent to:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
```

## Best Practices

1. **Version Control Configs**: Always commit config files to git. They're small and essential for reproducibility.

2. **Use Experiment Configs**: Instead of modifying base configs, create experiment configs for different hyperparameter sets.

3. **Document Changes**: Add comments in config files explaining why certain values were chosen.

4. **Check Hydra Outputs**: Always verify that `outputs/.../.hydra/config.yaml` contains the expected configuration.

5. **Use Reproducibility Tester**: After making changes, test that identical configs produce identical results.

6. **Log Everything**: Use `log.info()` instead of `print()` so all output is captured in `train.log`.

## Troubleshooting

### Config Not Found

If you get an error about a config not being found:
- Check the file exists in `configs/`
- Verify the config name matches the file name (without `.yaml`)
- For experiments, ensure the file is in `configs/experiment/`

### Override Not Working

- Use dot notation: `training.batch_size=64` not `training[batch_size]=64`
- For nested dicts: `training.optimizer.lr=0.001`
- Use `+` to add new parameters: `+training.new_param=42`

### Model Weights Don't Match

If reproducibility test fails:
- Check that random seeds match
- Verify all hyperparameters are identical
- Ensure same code version and data version
- Check that no non-deterministic operations were used

## Further Reading

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [Hydra Tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/1_simple_cli/)
