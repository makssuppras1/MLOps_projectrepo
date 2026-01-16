"""Simple profiling utilities."""
import cProfile
import logging
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.profiler import tensorboard_trace_handler

log = logging.getLogger(__name__)


@contextmanager
def profile_training(output_dir: str = "outputs/profiling", with_memory: bool = False, print_table: bool = True):
    """Simple context manager for profiling training.

    Args:
        output_dir: Directory to save profiling results
        with_memory: Enable memory profiling
        print_table: Print profiling table summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tensorboard_logdir = output_path / "tensorboard"

    # Start cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    # Start PyTorch profiler
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    pytorch_profiler = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
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
