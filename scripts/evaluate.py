"""DVC stage: evaluate. Compares teacher/baseline/distilled on the held-out test set,
generates the seeded prediction-grid figure (replaces notebook cell 14's demo)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow

from kd_pipeline.config import load_params
from kd_pipeline.data import make_dataloaders
from kd_pipeline.device import get_device
from kd_pipeline.engine import evaluate as evaluate_accuracy
from kd_pipeline.mlflow_utils import init_tracking, mlflow_run
from kd_pipeline.model_io import load_model
from kd_pipeline.plotting import save_prediction_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    data_cfg = params["data"]
    teacher_cfg = params["teacher"]
    baseline_cfg = params["student_baseline"]
    distill_cfg = params["distill"]
    eval_cfg = params["evaluate"]
    seed = params["seed"]

    device = get_device()
    print(f"Using {device} device")

    _, _, test_loader, test_dataset = make_dataloaders(
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
    baseline_student = load_model(baseline_cfg["arch"], baseline_cfg["checkpoint"], device=device)
    distilled_student = load_model(distill_cfg["student_arch"], distill_cfg["checkpoint"], device=device)

    teacher_acc = evaluate_accuracy(teacher, test_loader, device)
    baseline_acc = evaluate_accuracy(baseline_student, test_loader, device)
    distilled_acc = evaluate_accuracy(distilled_student, test_loader, device)

    comparison = {
        "teacher_test_accuracy": teacher_acc,
        "student_baseline_test_accuracy": baseline_acc,
        "student_distilled_test_accuracy": distilled_acc,
    }

    comparison_path = Path(eval_cfg["comparison_path"])
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    prediction_grid_path = eval_cfg["prediction_grid_path"]
    save_prediction_grid(
        distilled_student,
        test_dataset,
        prediction_grid_path,
        device=device,
        num_samples=eval_cfg["num_demo_samples"],
        seed=eval_cfg["demo_seed"],
    )

    init_tracking(params["mlflow"]["tracking_uri"], params["mlflow"]["experiment_name"])
    with mlflow_run("evaluate", {"seed": seed}):
        mlflow.log_metrics(comparison)
        mlflow.log_artifact(str(comparison_path))
        mlflow.log_artifact(prediction_grid_path)

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
