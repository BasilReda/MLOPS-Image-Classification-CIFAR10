"""DVC stage: train_teacher. Trains DeepNN via plain CE, persists checkpoint + metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow

from kd_pipeline.config import load_params
from kd_pipeline.data import make_dataloaders
from kd_pipeline.device import get_device
from kd_pipeline.engine import evaluate, train
from kd_pipeline.mlflow_utils import init_tracking, mlflow_run
from kd_pipeline.model_io import save_checkpoint
from kd_pipeline.models import build_model
from kd_pipeline.seeding import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    data_cfg = params["data"]
    teacher_cfg = params["teacher"]
    seed = params["seed"]

    set_seed(seed)
    device = get_device()
    print(f"Using {device} device")

    train_loader, val_loader, test_loader, _ = make_dataloaders(
        root=data_cfg["raw_dir"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        val_fraction=data_cfg["val_fraction"],
        seed=seed,
        mean=data_cfg["normalize_mean"],
        std=data_cfg["normalize_std"],
        download=data_cfg["download"],
        max_samples=data_cfg.get("max_samples"),
    )

    model = build_model(teacher_cfg["arch"])

    init_tracking(params["mlflow"]["tracking_uri"], params["mlflow"]["experiment_name"])
    with mlflow_run("train_teacher", {"seed": seed, "data": data_cfg, "teacher": teacher_cfg}):
        history = train(
            model,
            train_loader,
            val_loader,
            epochs=teacher_cfg["epochs"],
            learning_rate=teacher_cfg["learning_rate"],
            device=device,
            patience=teacher_cfg["patience"],
        )
        test_accuracy = evaluate(model, test_loader, device)

        for epoch, (tr, va) in enumerate(zip(history["train_losses"], history["val_losses"]), start=1):
            mlflow.log_metric("train_loss", tr, step=epoch)
            mlflow.log_metric("val_loss", va, step=epoch)
        mlflow.log_metric("test_accuracy", test_accuracy)

        save_checkpoint(
            model,
            teacher_cfg["checkpoint"],
            arch=teacher_cfg["arch"],
            metadata={"test_accuracy": test_accuracy, "stopped_epoch": history["stopped_epoch"]},
        )
        mlflow.log_artifact(teacher_cfg["checkpoint"])

        metrics = {
            "test_accuracy": test_accuracy,
            "final_train_loss": history["train_losses"][-1],
            "final_val_loss": history["val_losses"][-1],
            "stopped_epoch": history["stopped_epoch"],
        }
        metrics_path = Path(teacher_cfg["metrics_path"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metrics_path))

    print(f"Teacher test accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
