"""Checkpoint persistence via state_dict (not whole-module pickling).

Replaces the original notebook's `pickle.dump(model)` to a Colab Drive path:
whole-module pickling requires the exact class definition/environment to
survive unchanged, and doesn't play well with MLflow's model-logging
patterns. We save/load state_dict + a small metadata sidecar instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from kd_pipeline.models import build_model


def save_checkpoint(model: nn.Module, path: str | Path, arch: str, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

    sidecar = path.with_suffix(path.suffix + ".json")
    payload = {"arch": arch, **(metadata or {})}
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model(arch: str, path: str | Path, device: str = "cpu", num_classes: int = 10) -> nn.Module:
    model = build_model(arch, num_classes=num_classes)
    state_dict = torch.load(Path(path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
