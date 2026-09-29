"""Entry point cho huấn luyện mô hình. Chọn model qua argument:
    python -m src.train --model M1|M2|M3
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import (
    NUM_CLASSES,
    IMG_SIZE_M1,
    IMG_SIZE_M2,
    IMG_SIZE_M3,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    SEED,
    MODELS_DIR,
    RESULTS_DIR,
)
from src.data import build_dataloader, split_dataset
from src.utils import save_metrics, save_model
from src.utils import set_seed


def build_model(model_name: str) -> nn.Module:
    """
    Khởi tạo model dựa trên tên.

    Args:
        model_name: str — "M1", "M2" hoặc "M3".

    Returns:
        torch.nn.Module
    """
    if model_name == "M1":
        from src.models.simple_nn import SimpleNN
        return SimpleNN()
    elif model_name == "M2":
        from src.models.cnn import DeepCNN
        return DeepCNN()
    elif model_name == "M3":
        from src.models.transfer import TransferModel
        return TransferModel()
    else:
        raise ValueError(f"Chọn M1, M2 hoặc M3. Nhận được: {model_name}")


def train(model_name: str, epochs: int, batch_size: int, lr: float):
    """
    Huấn luyện model trên dataset.

    Các bước (placeholder):
    1. set_seed(SEED)
    2. load dataset train / val từ src.data.build_dataloader()
    3. build model + di chuyển sang device (cuda nếu có)
    4. optimizer = optim.Adam(model.parameters(), lr=lr), loss = nn.CrossEntropyLoss()
    5. callbacks: EarlyStopping (custom), ReduceLROnPlateau
    6. vòng lặp: for epoch in range(epochs):
         - training loop (forward, loss, backward, step)
         - validation loop (tính loss + accuracy)
         - log metrics, lưu best model -> models/<model_name>.pth
    7. lưu metrics -> results/metrics/<model_name>_metrics.json
    """
    set_seed(SEED)
    epochs = EPOCHS if epochs is None else epochs
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    lr = LEARNING_RATE if lr is None else lr
    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError("epochs, batch_size và lr phải lớn hơn 0.")

    split_dir = Path(Path(__file__).resolve().parent.parent / "data" / "splits")
    if not all((split_dir / f"{name}.csv").is_file() for name in ("train", "val")):
        split_dataset()

    image_size = {
        "M1": IMG_SIZE_M1,
        "M2": IMG_SIZE_M2,
        "M3": IMG_SIZE_M3,
    }[model_name]
    train_loader = build_dataloader("train", image_size, batch_size)
    val_loader = build_dataloader("val", image_size, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val_loss = float("inf")
    best_path = Path(MODELS_DIR) / f"{model_name}_best.pth"
    for epoch in range(1, epochs + 1):
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
                print(
                    f"  train batch {batch_index}/{len(train_loader)}",
                    flush=True,
                )

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "val_loss": val_loss / val_total,
            "val_accuracy": val_correct / val_total,
        }
        history.append(epoch_metrics)
        elapsed = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch}/{epochs} complete - "
            f"train_loss={epoch_metrics['train_loss']:.4f}, "
            f"train_acc={epoch_metrics['train_accuracy']:.4f}, "
            f"val_loss={epoch_metrics['val_loss']:.4f}, "
            f"val_acc={epoch_metrics['val_accuracy']:.4f}, "
            f"time={elapsed / 60:.1f} min",
            flush=True,
        )
        if epoch_metrics["val_loss"] < best_val_loss:
            best_val_loss = epoch_metrics["val_loss"]
            save_model(model, str(best_path))

    metrics_path = Path(RESULTS_DIR) / "metrics" / f"{model_name}_training.json"
    save_metrics(
        {
            "model": model_name,
            "device": str(device),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "history": history,
            "best_checkpoint": str(best_path),
        },
        str(metrics_path),
    )
    return history


def main():
    parser = argparse.ArgumentParser(description="Train fruit and vegetable image classifier")
    parser.add_argument("--model", type=str, default="M1", choices=["M1", "M2", "M3"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    train(args.model, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
