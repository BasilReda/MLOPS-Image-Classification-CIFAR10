"""Loads the distilled student once and serves predictions.

Reuses kd_pipeline.data.get_transforms with the same normalization stats
used at training time, so train/serve preprocessing never drifts apart.
"""

from __future__ import annotations

import io
import os

import torch
import torch.nn.functional as F
from PIL import Image

from kd_pipeline.config import load_params
from kd_pipeline.data import get_transforms
from kd_pipeline.model_io import load_model
from kd_pipeline.models import CLASS_NAMES


class Predictor:
    def __init__(self, checkpoint_path: str | None = None, arch: str | None = None):
        params = load_params()
        distill_cfg = params["distill"]
        data_cfg = params["data"]

        self.checkpoint_path = checkpoint_path or os.environ.get(
            "KD_CHECKPOINT_PATH", distill_cfg["checkpoint"]
        )
        self.arch = arch or distill_cfg["student_arch"]
        self.device = "cpu"
        self.transform = get_transforms(data_cfg["normalize_mean"], data_cfg["normalize_std"])
        self.model = load_model(self.arch, self.checkpoint_path, device=self.device)

    def predict(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((32, 32))
        tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        confidence, pred_idx = probs.max(dim=0)
        top3_values, top3_indices = probs.topk(3)

        return {
            "class_name": CLASS_NAMES[pred_idx.item()],
            "confidence": confidence.item(),
            "top3": [
                {"class_name": CLASS_NAMES[i.item()], "probability": v.item()}
                for v, i in zip(top3_values, top3_indices)
            ],
        }
