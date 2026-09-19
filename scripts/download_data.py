"""DVC stage: download_data. Downloads CIFAR-10 to data/raw/ (idempotent)."""

from __future__ import annotations

import argparse

from kd_pipeline.config import load_params
from kd_pipeline.data import get_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    data_cfg = params["data"]

    train_dataset, test_dataset = get_datasets(
        root=data_cfg["raw_dir"],
        download=data_cfg["download"],
        mean=data_cfg["normalize_mean"],
        std=data_cfg["normalize_std"],
    )
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


if __name__ == "__main__":
    main()
