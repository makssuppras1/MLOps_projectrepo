"""Simple profiling utilities."""
import cProfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import torch
from loguru import logger
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler


@contextmanager
def profile_training(
    output_dir: str = "outputs/profiling",
    with_memory: bool = False,
    print_table: bool = True
) -> Generator[profile, None, None]:
    """Simple context manager for profiling training.

    Args:
        output_dir: Directory to save profiling results.
        with_memory: Enable memory profiling.
        print_table: Print profiling table summary.

    Yields:
        PyTorch profiler instance.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tensorboard_logdir = output_path / "tensorboard"

    # Start cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    # Start PyTorch profiler
    activities = [ProfilerActivity.CPU]
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
        profiler.disable()
        profiler.dump_stats(output_path / "cprofile_stats.prof")
        logger.info(f"Saved cProfile stats to {output_path / 'cprofile_stats.prof'}")
        logger.info(f"View with: snakeviz {output_path / 'cprofile_stats.prof'}")

        # Stop PyTorch profiler
        if pytorch_profiler:
            pytorch_profiler.stop()

            # Print table summary
            if print_table:
                sort_key = "self_cpu_memory_usage" if with_memory else "cpu_time_total"
                logger.info("\n" + "=" * 80)
                logger.info("Top operations by CPU time:")
                logger.info(pytorch_profiler.key_averages().table(sort_by=sort_key, row_limit=10))
                logger.info("=" * 80)

            logger.info(f"PyTorch traces saved to {output_path}")
            logger.info(f"TensorBoard logs saved to {tensorboard_logdir}")
            logger.info(f"View TensorBoard: tensorboard --logdir={tensorboard_logdir}")
