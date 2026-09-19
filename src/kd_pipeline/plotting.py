"""Figure generation, saved to disk (the original notebook only plt.show()'d)."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe for CI/Docker
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from kd_pipeline.data import denormalize
from kd_pipeline.models import CLASS_NAMES


def save_loss_curve(
    train_losses: list[float], val_losses: list[float], out_path: str | Path, title: str
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val/Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_prediction_grid(
    model,
    dataset,
    out_path: str | Path,
    device: str = "cpu",
    num_samples: int = 6,
    seed: int = 123,
) -> None:
    """Seeded sample of predictions on `dataset`, saved as a PNG (replaces cell 14's demo)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(num_samples, len(dataset)))

    model.to(device)
    model.eval()

    fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3.5))
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        image, true_label = dataset[idx]
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)

        predicted_label = pred_idx.item()
        confidence_pct = confidence.item() * 100

        img_display = denormalize(image).permute(1, 2, 0).numpy()
        ax.imshow(img_display)
        ax.axis("off")

        color = "green" if predicted_label == true_label else "red"
        title = (
            f"True: {CLASS_NAMES[true_label]}\n"
            f"Pred: {CLASS_NAMES[predicted_label]}\n"
            f"Conf: {confidence_pct:.1f}%"
        )
        ax.set_title(title, fontsize=9, color=color)

    fig.suptitle(
        "Student Model Inference on CIFAR-10 Test Images\n(green = correct | red = wrong)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
