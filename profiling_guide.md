# Profiling Guide

Simple profiling for performance optimization.

## Enable Profiling

Set `training.profile: true` in `configs/training_conf.yaml` or via command line:

```bash
uv run python src/pname/train.py training.profile=true
```

## View Results

Profiling results are saved to `outputs/profiling/`:

- **cProfile stats**: `cprofile_stats.prof` - Analyze with:
  ```python
  import pstats
  stats = pstats.Stats('outputs/profiling/cprofile_stats.prof')
  stats.sort_stats('cumulative').print_stats(20)
  ```

- **PyTorch traces**: `trace_*.json` - View in Chrome:
  1. Open `chrome://tracing`
  2. Load the trace file

That's it!
