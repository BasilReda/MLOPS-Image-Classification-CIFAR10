"""Knowledge-distillation training loop (same math as the original notebook cell 11)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    T: float,
    soft_target_loss_weight: float,
    ce_loss_weight: float,
    ce_loss_fn: nn.Module | None = None,
) -> torch.Tensor:
    """Combined KL(soft targets) + CE(hard labels) distillation loss for one batch."""
    ce_loss_fn = ce_loss_fn or nn.CrossEntropyLoss()

    soft_targets = F.softmax(teacher_logits / T, dim=-1)
    soft_prob = F.log_softmax(student_logits / T, dim=-1)
    soft_targets_loss = F.kl_div(soft_prob, soft_targets, reduction="batchmean") * (T**2)

    label_loss = ce_loss_fn(student_logits, labels)

    return soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss


def train_knowledge_distillation(
    teacher: nn.Module,
    student: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    learning_rate: float,
    T: float,
    soft_target_loss_weight: float,
    ce_loss_weight: float,
    device: str,
) -> dict:
    teacher.to(device)
    student.to(device)

    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)

            loss = kd_loss(
                student_logits,
                teacher_logits,
                labels,
                T=T,
                soft_target_loss_weight=soft_target_loss_weight,
                ce_loss_weight=ce_loss_weight,
                ce_loss_fn=ce_loss_fn,
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        student.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student(inputs)
                loss = ce_loss_fn(outputs, labels)
                running_val_loss += loss.item()
        val_loss = running_val_loss / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return {"train_losses": train_losses, "val_losses": val_losses}
