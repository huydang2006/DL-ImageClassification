"""Utility functions: device setup, seeding, and file I/O."""

import os
import json
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def get_device(requested="auto"):
    """Resolve compute device with CPU fallback."""
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda", "xpu"}:
        raise ValueError("device must be one of: auto, cpu, cuda, xpu.")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available.")
        return torch.device("cuda")
    if requested == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU not available.")
        return torch.device("xpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_image_grid(images, labels, class_names, n_rows=4, n_cols=4):
    """Plot a grid of images for visualization."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, img, label in zip(axes, images[: n_rows * n_cols], labels):
        img = img.detach().cpu()
        if img.ndim == 3:
            img = img.permute(1, 2, 0)
        ax.imshow(img.numpy())
        ax.set_title(class_names[int(label)])
    fig.tight_layout()
    return fig


def save_metrics(metrics: dict, filepath: str):
    """Save dictionary to JSON."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2))


def save_model(model: torch.nn.Module, filepath: str):
    """Save model state dict."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)


def load_model(model_class, filepath: str, device="cpu"):
    """Load model state dict into instance."""
    model = model_class()
    model.load_state_dict(torch.load(filepath, map_location=device))
    return model.to(device)
