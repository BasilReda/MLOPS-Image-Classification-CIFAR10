"""Generic CE training loop + accuracy evaluation.

Fixes vs. the original notebook: `train()` always receives a real val_loader
(built by data.make_splits), so the TypeError-causing signature mismatch from
the notebook's cells 5/9 (calling train() without val_loader) can't happen.
Returns structured metrics instead of only printing.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.optim as optim


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    learning_rate: float,
    device: str,
    patience: int = 5,
) -> dict:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    train_losses: list[float] = []
    val_losses: list[float] = []
    stopped_epoch = epochs

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
        val_loss = running_val_loss / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{patience} epochs")
            if patience_counter >= patience:
                print("Early stopping triggered!")
                stopped_epoch = epoch + 1
                break

    model.load_state_dict(best_model_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "stopped_epoch": stopped_epoch,
    }


def evaluate(model: nn.Module, data_loader, device: str) -> float:
    """Top-1 accuracy (%) of model on data_loader."""
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy
