"""Training entry point.
Usage: python -m src.train --model M1|M2|M3
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from src.config import (
    NUM_CLASSES,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    M3_FREEZE_EPOCHS,
    M3_FINETUNE_EPOCHS,
    LR_SCHEDULER_PATIENCE,
    EARLY_STOPPING_PATIENCE,
    SEED,
    MODELS_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
    MODEL_CHOICES,
    get_model_image_size,
)
from src.data import build_dataloader
from src.preprocessing import get_class_weights, prepare_dataset
from src.utils import get_device, save_metrics, save_model, set_seed


def build_model(model_name: str, pretrained: bool = False) -> nn.Module:
    """Initialize model by name."""
    if model_name == "M1":
        from src.models.simple_nn import SimpleNN
        return SimpleNN()
    elif model_name == "M2":
        from src.models.cnn import DeepCNN
        return DeepCNN()
    elif model_name == "M3":
        from src.models.transfer import TransferModel
        return TransferModel(pretrained=pretrained)
    else:
        raise ValueError(f"Expected M1, M2 or M3, got: {model_name}")


def train(model_name: str, epochs: int, batch_size: int, lr: float, device="auto"):
    """Train a model on the dataset.

    Flow: seed -> loaders -> model -> weighted CE + Adam -> epoch loop with
    validation macro-F1, ReduceLROnPlateau, early stopping -> save best model
    and metrics. M3 freezes the backbone for the first epochs, then fine-tunes.
    """
    set_seed(SEED)
    if epochs is None:
        epochs = (
            M3_FREEZE_EPOCHS + M3_FINETUNE_EPOCHS
            if model_name == "M3"
            else EPOCHS
        )
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    lr = LEARNING_RATE if lr is None else lr
    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError("epochs, batch_size and lr must be positive.")

    split_dir = Path(SPLITS_DIR)
    if not all((split_dir / f"{name}.csv").is_file() for name in ("train", "val")):
        prepare_dataset()

    image_size = get_model_image_size(model_name)
    device = get_device(device)
    pin_memory = device.type in {"cuda", "xpu"}
    train_loader = build_dataloader("train", image_size, batch_size, pin_memory)
    val_loader = build_dataloader("val", image_size, batch_size, pin_memory)
    model = build_model(model_name, pretrained=model_name == "M3").to(device)
    class_weights = get_class_weights("train").to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val_f1 = float("-inf")
    epochs_without_improvement = 0
    best_path = Path(MODELS_DIR) / f"{model_name}_best.pth"
    freeze_epochs = min(M3_FREEZE_EPOCHS, epochs) if model_name == "M3" else 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=LR_SCHEDULER_PATIENCE
    )

    for epoch in range(1, epochs + 1):
        if model_name == "M3" and epoch == freeze_epochs + 1:
            from src.models.transfer import unfreeze_backbone

            unfreeze_backbone(model)
            optimizer = optim.Adam(model.parameters(), lr=lr * 0.1)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=LR_SCHEDULER_PATIENCE
            )
        epoch_start = time.perf_counter()
        print(f"Epoch {epoch}/{epochs} - training...", flush=True)
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(train_loader):
                print(f"  train batch {batch_index}/{len(train_loader)}", flush=True)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_targets = []
        val_predictions = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)
                val_targets.extend(labels.cpu().tolist())
                val_predictions.extend(outputs.argmax(dim=1).cpu().tolist())

        val_macro_f1 = f1_score(
            val_targets, val_predictions, average="macro", zero_division=0
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "val_loss": val_loss / val_total,
            "val_accuracy": val_correct / val_total,
            "val_macro_f1": val_macro_f1,
        }
        history.append(epoch_metrics)
        elapsed = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch}/{epochs} complete - "
            f"train_loss={epoch_metrics['train_loss']:.4f}, "
            f"train_acc={epoch_metrics['train_accuracy']:.4f}, "
            f"val_loss={epoch_metrics['val_loss']:.4f}, "
            f"val_acc={epoch_metrics['val_accuracy']:.4f}, "
            f"val_macro_f1={val_macro_f1:.4f}, "
            f"time={elapsed / 60:.1f} min",
            flush=True,
        )
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            epochs_without_improvement = 0
            save_model(model, str(best_path))
        else:
            epochs_without_improvement += 1
        scheduler.step(val_macro_f1)
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping after {epoch} epochs: "
                f"validation macro-F1 did not improve.",
                flush=True,
            )
            break

    metrics_path = Path(RESULTS_DIR) / "metrics" / f"{model_name}_training.json"
    save_metrics(
        {
            "model": model_name,
            "device": str(device),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "class_weights": class_weights.cpu().tolist(),
            "history": history,
            "best_checkpoint": str(best_path),
        },
        str(metrics_path),
    )
    return history


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train fruit and vegetable image classifier")
    parser.add_argument("--model", type=str, default="M1", choices=MODEL_CHOICES)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "xpu"],
        help="Compute device; auto prefers CUDA, then XPU, then CPU.",
    )
    args = parser.parse_args()
    train(args.model, args.epochs, args.batch_size, args.lr, args.device)


if __name__ == "__main__":
    main()
