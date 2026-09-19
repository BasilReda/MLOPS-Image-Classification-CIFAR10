"""CIFAR-10 loading, transforms, and DataLoader construction.

Fixes vs. the original notebook:
  - Uses CIFAR-10-specific normalization stats instead of ImageNet's.
  - Carves a real held-out validation split out of the training set (seeded),
    so the test set is reserved exclusively for final evaluation.
  - batch_size/num_workers come from params only (no stray comment mismatch).
  - Optional `max_samples` truncation for fast CI smoke-tests.
"""

from __future__ import annotations

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from kd_pipeline.seeding import make_generator

# CIFAR-10 per-channel mean/std (not ImageNet's) -- see params.yaml: data.normalize_mean/std
DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
DEFAULT_STD = [0.2470, 0.2435, 0.2616]


def get_transforms(mean: list[float] = DEFAULT_MEAN, std: list[float] = DEFAULT_STD):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_datasets(root: str, download: bool, mean: list[float], std: list[float]):
    tfm = get_transforms(mean, std)
    train_dataset = datasets.CIFAR10(root=root, train=True, download=download, transform=tfm)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=download, transform=tfm)
    return train_dataset, test_dataset


def maybe_truncate(dataset: Dataset, max_samples: int | None) -> Dataset:
    """Truncate a dataset to at most max_samples (for CI smoke-tests); no-op if None."""
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def make_splits(train_dataset: Dataset, val_fraction: float, seed: int):
    """Carve a seeded train/val split out of the CIFAR-10 training set."""
    n_val = int(len(train_dataset) * val_fraction)
    n_train = len(train_dataset) - n_val
    generator = make_generator(seed)
    train_subset, val_subset = random_split(train_dataset, [n_train, n_val], generator=generator)
    return train_subset, val_subset


def make_dataloaders(
    root: str,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    seed: int,
    mean: list[float] = DEFAULT_MEAN,
    std: list[float] = DEFAULT_STD,
    download: bool = True,
    max_samples: int | None = None,
):
    """Build train/val/test DataLoaders with a real held-out validation split."""
    train_dataset, test_dataset = get_datasets(root, download, mean, std)
    train_dataset = maybe_truncate(train_dataset, max_samples)
    test_dataset = maybe_truncate(test_dataset, max_samples)

    train_subset, val_subset = make_splits(train_dataset, val_fraction, seed)
    generator = make_generator(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader, test_dataset


def denormalize(
    image: torch.Tensor, mean: list[float] = DEFAULT_MEAN, std: list[float] = DEFAULT_STD
) -> torch.Tensor:
    """Invert get_transforms()'s normalization for display purposes."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return (image * std_t + mean_t).clamp(0, 1)
