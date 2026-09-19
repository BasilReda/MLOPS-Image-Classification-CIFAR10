"""DVC stage: distill_student.

Loads the teacher checkpoint, runs KD training, persists the distilled student.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.pytorch

from kd_pipeline.config import load_params
from kd_pipeline.data import make_dataloaders
from kd_pipeline.device import get_device
from kd_pipeline.distill import train_knowledge_distillation
from kd_pipeline.engine import evaluate
from kd_pipeline.mlflow_utils import init_tracking, mlflow_run
from kd_pipeline.model_io import load_model, save_checkpoint
from kd_pipeline.models import build_model
from kd_pipeline.plotting import save_loss_curve
from kd_pipeline.seeding import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    data_cfg = params["data"]
    teacher_cfg = params["teacher"]
    distill_cfg = params["distill"]
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

    teacher = load_model(teacher_cfg["arch"], teacher_cfg["checkpoint"], device=device)
    student = build_model(distill_cfg["student_arch"])

    init_tracking(params["mlflow"]["tracking_uri"], params["mlflow"]["experiment_name"])
    with mlflow_run("distill_student", {"seed": seed, "data": data_cfg, "distill": distill_cfg}):
        history = train_knowledge_distillation(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=distill_cfg["epochs"],
            learning_rate=distill_cfg["learning_rate"],
            T=distill_cfg["temperature"],
            soft_target_loss_weight=distill_cfg["soft_target_loss_weight"],
            ce_loss_weight=distill_cfg["ce_loss_weight"],
            device=device,
        )
        test_accuracy = evaluate(student, test_loader, device)

        for epoch, (tr, va) in enumerate(zip(history["train_losses"], history["val_losses"]), start=1):
            mlflow.log_metric("train_loss", tr, step=epoch)
            mlflow.log_metric("val_loss", va, step=epoch)
        mlflow.log_metric("test_accuracy", test_accuracy)

        save_checkpoint(
            student,
            distill_cfg["checkpoint"],
            arch=distill_cfg["student_arch"],
            metadata={"test_accuracy": test_accuracy, "temperature": distill_cfg["temperature"]},
        )
        mlflow.log_artifact(distill_cfg["checkpoint"])

        loss_curve_path = distill_cfg["loss_curve_path"]
        save_loss_curve(
            history["train_losses"],
            history["val_losses"],
            loss_curve_path,
            title="Knowledge Distillation: Train vs Val Loss",
        )
        mlflow.log_artifact(loss_curve_path)

        metrics = {
            "test_accuracy": test_accuracy,
            "final_train_loss": history["train_losses"][-1],
            "final_val_loss": history["val_losses"][-1],
        }
        metrics_path = Path(distill_cfg["metrics_path"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metrics_path))

        model_info = mlflow.pytorch.log_model(student, artifact_path="model")
        mlflow.register_model(model_info.model_uri, "kd-cifar10-student")

    print(f"Distilled student test accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
