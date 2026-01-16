"""Simple profiling utilities for training performance analysis."""

import cProfile
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

log = logging.getLogger(__name__)


@contextmanager
def profile_training(
    output_dir: str = "outputs/profiling",
    with_memory: bool = False,
    print_table: bool = True
) -> Generator[profile, None, None]:
    """
    Context manager for profiling training performance.

    Provides both cProfile (Python-level) and PyTorch profiler (GPU/CPU operations)
    for comprehensive performance analysis. Saves results to output_dir and optionally
    prints summary statistics.

    Usage:
        with profile_training(output_dir="profiling", with_memory=True):
            # Training code here
            train_model()

    Args:
        output_dir: Directory path to save profiling results. Defaults to "outputs/profiling".
        with_memory: If True, enable memory profiling (slower but more detailed). Defaults to False.
        print_table: If True, print summary table of top operations. Defaults to True.

    Yields:
        PyTorch profiler instance that can be stepped during training.

    Example:
        >>> with profile_training() as prof:
        ...     for epoch in range(epochs):
        ...         prof.step()  # Mark profiling step
        ...         train_one_epoch()
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tensorboard_logdir = output_path / "tensorboard"

    # Start cProfile for Python-level profiling
    cprofiler = cProfile.Profile()
    cprofiler.enable()

    # Start PyTorch profiler for GPU/CPU operation profiling
    activities: list[ProfilerActivity] = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    pytorch_profiler = profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
        record_shapes=True,
        with_stack=True,
        with_memory=with_memory,
        on_trace_ready=[
            lambda prof: prof.export_chrome_trace(str(output_path / f"trace_{prof.step_num}.json")),
            tensorboard_trace_handler(str(tensorboard_logdir)),
        ],
    )
    pytorch_profiler.start()

    try:
        yield pytorch_profiler
    finally:
        # Stop cProfile and save
        cprofiler.disable()
        cprofiler.dump_stats(output_path / "cprofile_stats.prof")
        log.info(f"Saved cProfile stats to {output_path / 'cprofile_stats.prof'}")
        log.info(f"View with: snakeviz {output_path / 'cprofile_stats.prof'}")

        # Stop PyTorch profiler
        if pytorch_profiler:
            pytorch_profiler.stop()

            # Print table summary
            if print_table:
                sort_key = "self_cpu_memory_usage" if with_memory else "cpu_time_total"
                log.info("\n" + "=" * 80)
                log.info("Top operations by CPU time:")
                log.info(pytorch_profiler.key_averages().table(sort_by=sort_key, row_limit=10))
                log.info("=" * 80)

            log.info(f"PyTorch traces saved to {output_path}")
            log.info(f"TensorBoard logs saved to {tensorboard_logdir}")
            log.info(f"View TensorBoard: tensorboard --logdir={tensorboard_logdir}")
