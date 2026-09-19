"""Reproducibility helpers: seed every RNG the pipeline touches."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python's random, numpy, and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(seed: int) -> torch.Generator:
    """A seeded torch.Generator for DataLoader(shuffle=True, generator=...) and random_split."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
