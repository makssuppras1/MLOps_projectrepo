# Profiling Guide

Simple profiling for performance optimization.

## Enable Profiling

Set `training.profile: true` in `configs/training_conf.yaml` or via command line:

```bash
uv run python src/pname/train.py training.profile=true
```

## View Results

Profiling results are saved to `outputs/profiling/`:

### cProfile Results

**File**: `cprofile_stats.prof`

**Option 1: Using snakeviz (recommended)**
```bash
uv run snakeviz outputs/profiling/cprofile_stats.prof
```
Opens interactive visualization in your browser.

**Option 2: Using pstats**
```python
import pstats
stats = pstats.Stats('outputs/profiling/cprofile_stats.prof')
stats.sort_stats('cumulative').print_stats(20)
```

### PyTorch Profiler Results

**Chrome traces**: `trace_*.json`
1. Open `chrome://tracing` in Chrome/Edge
2. Load the trace file

**TensorBoard (recommended)**
```bash
tensorboard --logdir=outputs/profiling/tensorboard
```
Then open http://localhost:6006/#pytorch_profiler

The profiler automatically prints a summary table of top operations by CPU time after training completes.
