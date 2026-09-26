"""Pipeline dữ liệu: tải, tiền xử lý, chia tập và tạo DataLoader (PyTorch)."""

import os
from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    RAW_DATA_DIR,
    SPLITS_DIR,
    IMG_SIZE_M1,
    IMG_SIZE_M3,
    BATCH_SIZE,
    NUM_CLASSES,
    SEED,
)


# ---------------------------------------------------------------------------
# Hàm placeholder
# ---------------------------------------------------------------------------
def download_dataset():
    """
    Tải dataset từ Kaggle sử dụng Kaggle API.

    Dataset: Fruit and Vegetable Disease (Healthy vs Rotten)
    URL: https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten

    TODO:
    - Cài đặt kaggle API
    - Download và giải nén vào RAW_DATA_DIR
    """
    # !kaggle datasets download -d muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten
    # unzip ... -d RAW_DATA_DIR
    return NotImplemented


def get_labels_mapping() -> Dict[str, int]:
    """
    Trả về dict ánh xạ tên thư mục (class) -> số nguyên (label index).

    Dataset gồm 28 thư mục con, mỗi thư mục tương ứng 1 lớp.
    Ví dụ: {"Apple_healthy": 0, "Apple_rotten": 1, ...}

    TODO: quét RAW_DATA_DIR để sinh mapping tự động.
    """
    return NotImplemented


def split_dataset():
    """
    Chia dataset thành train / val / test (70/15/15) theo stratified sampling
    trên 28 lớp, lưu kết quả dưới dạng CSV vào SPLITS_DIR.
    """
    return NotImplemented


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
class FruitVegDataset(Dataset):
    """
    PyTorch Dataset class cho bài toán phân loại ảnh.

    TODO:
    - __init__: load CSV split file, khởi tạo transform.
    - __len__: trả về số ảnh.
    - __getitem__: load ảnh từ filepath, apply transform, trả về (image_tensor, label).
    """

    def __init__(self, split_file: str, img_size: int, transform=None):
        """
        Args:
            split_file: str, đường dẫn tới file CSV trong SPLITS_DIR.
            img_size: int, kích thước ảnh resize (128 hoặc 224).
            transform: torchvision.transforms.Compose (nếu có).
        """
        self.transform = transform
        return NotImplemented

    def __len__(self) -> int:
        return NotImplemented

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return NotImplemented


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
def build_dataloader(split_name: str, img_size: int, batch_size: int = BATCH_SIZE) -> DataLoader:
    """
    Tạo DataLoader cho một split (train/val/test).

    Args:
        split_name: str, tên file CSV trong SPLITS_DIR (ví dụ: "train.csv").
        img_size: int, kích thước ảnh.
        batch_size: int.

    Returns:
        torch.utils.data.DataLoader
    """
    # TODO: load transform (train: augmentation + normalize; val/test: chỉ normalize)
    # dataset = FruitVegDataset(split_file, img_size, transform)
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=..., num_workers=...)
    return NotImplemented


def create_augmentation_pipeline(img_size: int):
    """
    Tạo transform augmentation cho tập training.

    Augmentations đề xuất: RandomHorizontalFlip, RandomRotation, ColorJitter,
    RandomResizedCrop/Resize.

    TODO: dùng torchvision.transforms.Compose([...])
    """
    return NotImplemented


def get_default_transform(img_size: int):
    """
    Transform chuẩn cho validation/test (resize + to_tensor + normalize ImageNet stats).

    TODO: torchvision.transforms.Compose([Resize, ToTensor, Normalize])
    """
    return NotImplemented
