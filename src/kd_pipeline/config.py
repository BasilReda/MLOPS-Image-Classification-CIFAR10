"""params.yaml loading — the single source of truth for pipeline hyperparameters."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[2] / "params.yaml"


def load_params(path: str | Path = DEFAULT_PARAMS_PATH) -> dict:
    """Load params.yaml into a plain dict. Raises FileNotFoundError/YAMLError on problems."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"params file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    if not isinstance(params, dict):
        raise ValueError(f"params file did not parse to a mapping: {path}")
    return params
