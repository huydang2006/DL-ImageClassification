"""Pipeline dữ liệu: tải, tiền xử lý, chia tập và tạo DataLoader (PyTorch)."""

import os
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Tuple, Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms

from src.config import (
    BASE_DIR,
    RAW_DATA_DIR,
    SPLITS_DIR,
    IMG_SIZE_M1,
    IMG_SIZE_M3,
    BATCH_SIZE,
    NUM_CLASSES,
    SEED,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
KAGGLE_DATASET = "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Dataset download and split helpers
# ---------------------------------------------------------------------------
def _has_complete_dataset(raw_dir: Path) -> bool:
    """Return whether raw_dir contains all expected non-empty class folders."""
    class_dirs = [path for path in raw_dir.iterdir() if path.is_dir()] if raw_dir.is_dir() else []
    if len(class_dirs) != NUM_CLASSES:
        return False
    return all(
        any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in class_dir.rglob("*")
        )
        for class_dir in class_dirs
    )


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

        for source in source_root.iterdir():
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
    raw_dir = Path(RAW_DATA_DIR)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {raw_dir}")

    class_names = sorted(path.name for path in raw_dir.iterdir() if path.is_dir())
    if not class_names:
        raise ValueError(f"Không tìm thấy thư mục class trong: {raw_dir}")

    return {class_name: index for index, class_name in enumerate(class_names)}


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
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio và val_ratio phải dương, tổng phải nhỏ hơn 1.")

    labels_mapping = get_labels_mapping()
    records: List[dict] = []
    raw_dir = Path(RAW_DATA_DIR)

    for class_name, label in labels_mapping.items():
        class_dir = raw_dir / class_name
        image_paths = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"Class không có ảnh hợp lệ: {class_name}")

        records.extend(
            {
                "filepath": str(path.relative_to(Path(BASE_DIR))),
                "label": label,
                "class_name": class_name,
                "content_hash": _file_sha256(path),
            }
            for path in image_paths
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Không tìm thấy ảnh hợp lệ trong dataset.")

    groups = frame.groupby("content_hash", as_index=False).agg(
        label=("label", "first"),
    )
    conflicting_groups = frame.groupby("content_hash")["label"].nunique()
    conflicting_hashes = set(conflicting_groups[conflicting_groups > 1].index)
    if conflicting_hashes:
        excluded = frame[frame["content_hash"].isin(conflicting_hashes)].drop(
            columns=["content_hash"]
        )
        excluded.to_csv(
            Path(SPLITS_DIR) / "excluded_conflicting_duplicates.csv",
            index=False,
        )
        frame = frame[~frame["content_hash"].isin(conflicting_hashes)].copy()
        groups = groups[~groups["content_hash"].isin(conflicting_hashes)].copy()

    train_groups, remainder_groups = train_test_split(
        groups,
        test_size=1 - train_ratio,
        stratify=groups["label"],
        random_state=seed,
    )
    val_share_of_remainder = val_ratio / (1 - train_ratio)
    val_groups, test_groups = train_test_split(
        remainder_groups,
        test_size=1 - val_share_of_remainder,
        stratify=remainder_groups["label"],
        random_state=seed,
    )

    os.makedirs(SPLITS_DIR, exist_ok=True)
    group_splits = {
        "train": set(train_groups["content_hash"]),
        "val": set(val_groups["content_hash"]),
        "test": set(test_groups["content_hash"]),
    }
    split_frames = {
        split_name: frame[frame["content_hash"].isin(content_hashes)]
        for split_name, content_hashes in group_splits.items()
    }
    split_paths = {}
    for split_name, split_frame in split_frames.items():
        split_path = Path(SPLITS_DIR) / f"{split_name}.csv"
        split_frame.drop(columns=["content_hash"]).sort_values(
            ["label", "filepath"]
        ).to_csv(split_path, index=False)
        split_paths[split_name] = str(split_path)

    return split_paths


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
        self.transform = transform or get_default_transform(img_size)
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
                if image.mode == "P" and "transparency" in image.info:
                    image = image.convert("RGBA").convert("RGB")
                else:
                    image = image.convert("RGB")
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Không thể đọc ảnh: {image_path}") from exc
        return self.transform(image), int(row["label"])


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
        create_augmentation_pipeline(img_size)
        if split_name == "train"
        else get_default_transform(img_size)
    )
    dataset = FruitVegDataset(str(split_file), img_size, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split_name == "train",
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def get_class_weights(split_name: str = "train") -> torch.Tensor:
    """Return normalized inverse-frequency weights for the requested split."""
    split_file = Path(SPLITS_DIR) / f"{split_name}.csv"
    if not split_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy split file: {split_file}")
    frame = pd.read_csv(split_file)
    counts = frame["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    if (counts == 0).any():
        raise ValueError("Split file phải chứa ít nhất một ảnh cho mỗi class.")
    weights = len(frame) / (NUM_CLASSES * counts.astype(float))
    return torch.tensor(weights.to_numpy(), dtype=torch.float32)


def find_cross_split_duplicates() -> Dict[str, List[str]]:
    """Find byte-identical images that occur in more than one split."""
    split_hashes: Dict[str, Dict[str, str]] = {}
    for split_name in ("train", "val", "test"):
        split_file = Path(SPLITS_DIR) / f"{split_name}.csv"
        if not split_file.is_file():
            raise FileNotFoundError(f"Không tìm thấy split file: {split_file}")
        frame = pd.read_csv(split_file)
        hashes = {}
        for relative_path in frame["filepath"]:
            image_path = Path(BASE_DIR) / Path(relative_path)
            if not image_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
            hashes[_file_sha256(image_path)] = str(relative_path)
        split_hashes[split_name] = hashes

    duplicates: Dict[str, List[str]] = {}
    for split_name, hashes in split_hashes.items():
        for digest, relative_path in hashes.items():
            other_splits = [
                other_name
                for other_name, other_hashes in split_hashes.items()
                if other_name != split_name and digest in other_hashes
            ]
            if other_splits:
                locations = [f"{split_name}:{relative_path}"]
                locations.extend(
                    f"{other_name}:{split_hashes[other_name][digest]}"
                    for other_name in other_splits
                )
                duplicates[digest] = sorted(set(locations))
    return duplicates


def create_augmentation_pipeline(img_size: int):
    """
    Tạo transform augmentation cho tập training.

    Augmentations đề xuất: RandomHorizontalFlip, RandomRotation, ColorJitter,
    RandomResizedCrop/Resize.

    """
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def get_default_transform(img_size: int):
    """
    Transform chuẩn cho validation/test (resize + to_tensor + normalize ImageNet stats).

    """
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
