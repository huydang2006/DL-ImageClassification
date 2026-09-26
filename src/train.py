"""Entry point cho huấn luyện mô hình. Chọn model qua argument:
    python -m src.train --model M1|M2|M3
"""

import argparse

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
)
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
    return NotImplemented


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân loại ảnh rau củ quả")
    parser.add_argument("--model", type=str, default="M1", choices=["M1", "M2", "M3"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    train(args.model, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
