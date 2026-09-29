"""Các hàm tiện ích dùng chung: set seed, vẽ ảnh, lưu/load model."""

import os
import json
from pathlib import Path

import numpy as np
import random

import torch
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """
    Cố định seed cho random, numpy, pytorch để kết quả tái lắp được.

    Seeds Python, NumPy and PyTorch and enables deterministic cuDNN behavior.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_image_grid(images, labels, class_names, n_rows=4, n_cols=4):
    """Vẽ lưới ảnh dưới dạng matplotlib subplot để visualize batch dữ liệu."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for axis in axes:
        axis.axis("off")
    for axis, image, label in zip(axes, images[: n_rows * n_cols], labels):
        image = image.detach().cpu()
        if image.ndim == 3:
            image = image.permute(1, 2, 0)
        axis.imshow(image.numpy())
        axis.set_title(class_names[int(label)])
        axis.axis("off")
    fig.tight_layout()
    return fig


def save_metrics(metrics: dict, filepath: str):
    """Lưu dict metrics ra file JSON."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_model(model: torch.nn.Module, filepath: str):
    """
    Lưu state_dict của model.
    Save the model state dictionary to a checkpoint file.
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def load_model(model_class, filepath: str, device="cpu"):
    """
    Load weights vào một model instance.
    Load a state dictionary into a newly constructed model instance.
    """
    model = model_class()
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)
