"""Local Ollama model provisioning for the CLI.

Presence checks against ``ollama list`` plus the interactive pull prompt, used to
offer a one-time download of a missing chat or embedding model rather than
failing with a 404 on first use.
"""

from __future__ import annotations

import subprocess

import rich_click as click

from aiida_agents._settings import ModelSettings


def _ollama_model_names(list_output: str) -> set[str]:
    """Model names from ``ollama list`` output (first column, header skipped)."""
    return {line.split()[0] for line in list_output.splitlines()[1:] if line.strip()}


def _ollama_lists_model(model: str, list_output: str) -> bool:
    """Whether ``model`` appears in ``ollama list`` output.

    ``ollama list`` shows an *untagged* model as ``<name>:latest``, so an
    untagged configured name (e.g. ``mxbai-embed-large``) is normalised to
    ``:latest`` before the exact-name comparison; otherwise it would never match
    and the pull prompt would fire on every run.
    """
    names = _ollama_model_names(list_output)
    wanted = model if ":" in model else f"{model}:latest"
    return wanted in names


def _ollama_has_model(model: str) -> bool:  # pragma: no cover
    """Whether the local Ollama server already has ``model``.

    Returns ``True`` when the check can't run (no ``ollama`` on PATH, server
    down): a presence check we couldn't perform must never block the session, so
    a genuinely missing model instead surfaces on the first query.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return _ollama_lists_model(model, result.stdout)


def _ollama_pull(model: str) -> None:  # pragma: no cover
    """Download a model into the local Ollama server."""
    click.echo(f"Running: ollama pull {model}")
    subprocess.run(["ollama", "pull", model], check=False)  # noqa: S607


def _prompt_pull_ollama_model(model: str) -> None:  # pragma: no cover
    """Offer to pull a missing local Ollama model (a one-time download prompt)."""
    if _ollama_has_model(model):
        return
    click.echo(f"Ollama model '{model}' is not pulled.")
    if click.confirm("Pull it now?", default=True):
        _ollama_pull(model)


def _ensure_ollama_model(settings: ModelSettings) -> None:  # pragma: no cover
    """For a local Ollama chat model that isn't pulled, offer to pull it.

    Turns a missing model into a one-time download prompt rather than a 404 on
    the first query. Only the ``ollama`` provider is affected; declining falls
    through to the normal path.
    """
    if settings.provider == "ollama":
        _prompt_pull_ollama_model(settings.model)
