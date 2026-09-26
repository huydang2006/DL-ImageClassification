"""Các hàm tiện ích dùng chung: set seed, vẽ ảnh, lưu/load model."""

import os

import numpy as np
import random

import torch


def set_seed(seed=42):
    """
    Cố định seed cho random, numpy, pytorch để kết quả tái lắp được.

    TODO:
    - random.seed(seed)
    - np.random.seed(seed)
    - torch.manual_seed(seed)
    - torch.cuda.manual_seed_all(seed)
    - torch.backends.cudnn.deterministic = True
    """
    # os.environ["PYTHONHASHSEED"] = str(seed)
    return NotImplemented


def plot_image_grid(images, labels, class_names, n_rows=4, n_cols=4):
    """Vẽ lưới ảnh dưới dạng matplotlib subplot để visualize batch dữ liệu."""
    return NotImplemented


def save_metrics(metrics: dict, filepath: str):
    """Lưu dict metrics ra file JSON."""
    return NotImplemented


def save_model(model: torch.nn.Module, filepath: str):
    """
    Lưu state_dict của model.
    TODO: torch.save(model.state_dict(), filepath)
    """
    return NotImplemented


def load_model(model_class, filepath: str, device="cpu"):
    """
    Load weights vào một model instance.
    TODO: model = model_class(); model.load_state_dict(torch.load(filepath, map_location=device))
    """
    return NotImplemented
