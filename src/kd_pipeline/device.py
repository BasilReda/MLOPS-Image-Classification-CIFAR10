"""Accelerator detection (same logic as notebook cell 0)."""

from __future__ import annotations

import torch


def get_device() -> str:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cuda" if torch.cuda.is_available() else "cpu"
