"""Simple profiling utilities."""
import cProfile
import logging
from contextlib import contextmanager
from pathlib import Path

import torch

log = logging.getLogger(__name__)


@contextmanager
def profile_training(output_dir: str = "outputs/profiling"):
    """Simple context manager for profiling training.

    Args:
        output_dir: Directory to save profiling results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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
        on_trace_ready=lambda prof: prof.export_chrome_trace(
            str(output_path / f"trace_{prof.step_num}.json")
        ),
    )
    pytorch_profiler.start()

    try:
        yield pytorch_profiler
    finally:
        # Stop cProfile and save
        profiler.disable()
        profiler.dump_stats(output_path / "cprofile_stats.prof")
        log.info(f"Saved cProfile stats to {output_path / 'cprofile_stats.prof'}")

        # Stop PyTorch profiler
        if pytorch_profiler:
            pytorch_profiler.stop()
            log.info(f"PyTorch traces saved to {output_path}")
