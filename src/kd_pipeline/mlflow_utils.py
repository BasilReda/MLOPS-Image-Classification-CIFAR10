"""MLflow tracking setup: local SQLite backend store + local artifact store.

Division of responsibility vs. DVC (see plan): DVC owns pipeline reproducibility
and the canonical versioned artifact files under models/; MLflow owns experiment
run history, metric comparison across attempts, and the model registry. Both
reference the same underlying checkpoint file -- the DVC-tracked one is canonical.
"""

from __future__ import annotations

from contextlib import contextmanager

import mlflow


def init_tracking(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextmanager
def mlflow_run(run_name: str, params: dict):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten(params))
        yield run


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested params dict into dotted keys for mlflow.log_params."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out
