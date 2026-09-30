"""Tải dữ liệu, định nghĩa Dataset và tạo DataLoader cho PyTorch."""

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.config import (
    BASE_DIR,
    RAW_DATA_DIR,
    SPLITS_DIR,
    BATCH_SIZE,
    NUM_CLASSES,
    SEED,
)

from src.preprocessing import (
    IMAGE_EXTENSIONS,
    convert_rgb,
    find_cross_split_duplicates as inspect_cross_split_duplicates,
    find_dataset_root,
    get_class_directories,
    get_class_weights as calculate_class_weights,
    get_evaluation_transform,
    get_labels_mapping as build_labels_mapping,
    get_training_transform,
    prepare_dataset,
)

KAGGLE_DATASET = "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"


# ---------------------------------------------------------------------------
# Dataset download and split helpers
# ---------------------------------------------------------------------------
def _has_complete_dataset(raw_dir: Path) -> bool:
    """Return whether raw_dir contains all expected non-empty class folders."""
    try:
        root = find_dataset_root(raw_dir)
    except (FileNotFoundError, ValueError):
        return False
    return len(get_class_directories(root)) == NUM_CLASSES


def _safe_extract(zip_path: Path, destination: Path) -> None:
    """Extract a zip archive without allowing paths outside destination."""
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"Archive chứa đường dẫn không an toàn: {member.filename}")
        archive.extractall(destination)


def download_dataset() -> str:
    """
    Tải dataset từ Kaggle sử dụng Kaggle API.

    Dataset: Fruit and Vegetable Disease (Healthy vs Rotten)
    URL: https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten

    Returns:
        Mô tả hành động: dataset đã tồn tại hoặc đã tải xuống.
    """
    raw_dir = Path(RAW_DATA_DIR)
    if _has_complete_dataset(raw_dir):
        return f"Dataset đã tồn tại tại {raw_dir}; bỏ qua tải xuống."

    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kaggle-download-") as temp_dir:
        download_dir = Path(temp_dir)
        command = [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(download_dir),
        ]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Không tìm thấy Kaggle CLI. Hãy cài dependencies bằng "
                "'pip install -r requirements.txt'."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Kaggle download thất bại. Hãy kiểm tra Kaggle API credentials "
                "và quyền truy cập dataset."
            ) from exc

        archives = list(download_dir.glob("*.zip"))
        if len(archives) != 1:
            raise RuntimeError(
                f"Kỳ vọng đúng một file zip từ Kaggle, nhận được {len(archives)}."
            )

        extracted_dir = download_dir / "extracted"
        extracted_dir.mkdir()
        _safe_extract(archives[0], extracted_dir)
        candidate_roots = [extracted_dir] + [
            path for path in extracted_dir.rglob("*") if path.is_dir()
        ]
        source_root = next(
            (path for path in candidate_roots if _has_complete_dataset(path)),
            None,
        )
        if source_root is None:
            raise RuntimeError(
                "Dataset tải về không chứa đủ 28 thư mục class có ảnh hợp lệ."
            )

        source_root = find_dataset_root(source_root)
        for source in get_class_directories(source_root):
            destination = raw_dir / source.name
            if destination.exists():
                shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
            shutil.move(str(source), str(destination))

    if not _has_complete_dataset(raw_dir):
        raise RuntimeError("Dataset sau khi giải nén không hợp lệ.")
    return f"Đã tải và giải nén dataset vào {raw_dir}."


def get_labels_mapping() -> Dict[str, int]:
    """
    Trả về dict ánh xạ tên thư mục (class) -> số nguyên (label index).

    Dataset gồm 28 thư mục con, mỗi thư mục tương ứng 1 lớp.
    Ví dụ: {"Apple_healthy": 0, "Apple_rotten": 1, ...}
    """
    return build_labels_mapping(RAW_DATA_DIR)


def split_dataset(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = SEED,
) -> Dict[str, str]:
    """
    Chia dataset thành train / val / test (70/15/15) theo stratified sampling
    trên 28 lớp, lưu kết quả dưới dạng CSV vào SPLITS_DIR.

    Returns:
        Dict ánh xạ tên split ("train", "val", "test") tới đường dẫn CSV.
    """
    return prepare_dataset(
        RAW_DATA_DIR, SPLITS_DIR, train_ratio, val_ratio, seed
    )


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
class FruitVegDataset(Dataset):
    """
    PyTorch Dataset class cho bài toán phân loại ảnh.

    Loads a CSV split, decodes an image and returns a transformed tensor and label.
    """

    def __init__(self, split_file: str, img_size: int, transform=None):
        """
        Args:
            split_file: str, đường dẫn tới file CSV trong SPLITS_DIR.
            img_size: int, kích thước ảnh resize (128 hoặc 224).
            transform: torchvision.transforms.Compose (nếu có).
        """
        self.transform = transform or get_evaluation_transform(
            img_size, convert_color=False
        )
        self.frame = pd.read_csv(split_file)
        required_columns = {"filepath", "label", "class_name"}
        missing_columns = required_columns.difference(self.frame.columns)
        if missing_columns:
            raise ValueError(
                f"Split file thiếu các cột bắt buộc: {sorted(missing_columns)}"
            )
        if self.frame.empty:
            raise ValueError(f"Split file không có dữ liệu: {split_file}")
        self.frame["label"] = self.frame["label"].astype(int)
        if not self.frame["label"].between(0, NUM_CLASSES - 1).all():
            raise ValueError("Split file chứa label nằm ngoài khoảng hợp lệ.")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.frame.iloc[idx]
        image_path = Path(BASE_DIR) / Path(row["filepath"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
        try:
            with Image.open(image_path) as image:
                image = convert_rgb(image)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Không thể đọc ảnh: {image_path}") from exc
        return self.transform(image), int(row["label"])


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
def build_dataloader(
    split_name: str,
    img_size: int,
    batch_size: int = BATCH_SIZE,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Tạo DataLoader cho một split (train/val/test).

    Args:
        split_name: str, tên file CSV trong SPLITS_DIR (ví dụ: "train.csv").
        img_size: int, kích thước ảnh.
        batch_size: int.

    Returns:
        torch.utils.data.DataLoader
    """
    if split_name.endswith(".csv"):
        split_name = split_name[:-4]
    if split_name not in {"train", "val", "test"}:
        raise ValueError("split_name phải là một trong: train, val, test.")
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")

    split_file = Path(SPLITS_DIR) / f"{split_name}.csv"
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy split file: {split_file}. Hãy chạy split_dataset() trước."
        )
    transform = (
        get_training_transform(img_size, convert_color=False)
        if split_name == "train"
        else get_evaluation_transform(img_size, convert_color=False)
    )
    dataset = FruitVegDataset(str(split_file), img_size, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split_name == "train",
        num_workers=0,
        pin_memory=pin_memory,
    )


def get_class_weights(split_name: str = "train") -> torch.Tensor:
    """Wrapper tương thích; phần tính trọng số nằm trong preprocessing."""
    return calculate_class_weights(split_name, SPLITS_DIR)


def find_cross_split_duplicates():
    """Wrapper tương thích; phần kiểm tra leakage nằm trong preprocessing."""
    return inspect_cross_split_duplicates(SPLITS_DIR)
