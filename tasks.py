import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "pname"
PYTHON_VERSION = "3.12"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context, model_checkpoint: str = "trained_model.pt") -> None:
    """Evaluate model."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/evaluate.py {model_checkpoint}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def visualize(ctx: Context, model_checkpoint: str = "trained_model.pt") -> None:
    """Visualize model embeddings."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/visualize.py {model_checkpoint}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def api(ctx: Context, host: str = "0.0.0.0", port: int = 8000, model_path: str = "") -> None:
    """Run API server."""
    cmd = f"uv run src/{PROJECT_NAME}/api.py --host {host} --port {port}"
    if model_path:
        cmd += f" --model-path {model_path}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
