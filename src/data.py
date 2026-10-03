"""Data loading, Dataset definition, and DataLoader creation."""

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


def _has_complete_dataset(raw_dir: Path) -> bool:
    """Check if raw_dir contains all expected class folders."""
    try:
        root = find_dataset_root(raw_dir)
    except (FileNotFoundError, ValueError):
        return False
    return len(get_class_directories(root)) == NUM_CLASSES


def _safe_extract(zip_path: Path, destination: Path) -> None:
    """Extract zip archive safely."""
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        archive.extractall(destination)


def download_dataset() -> str:
    """Download dataset from Kaggle."""
    raw_dir = Path(RAW_DATA_DIR)
    if _has_complete_dataset(raw_dir):
        return f"Dataset already exists at {raw_dir}; skipping download."

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
                "Kaggle CLI not found. Install dependencies with "
                "'pip install -r requirements.txt'."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Kaggle download failed. Check Kaggle API credentials."
            ) from exc

        archives = list(download_dir.glob("*.zip"))
        if len(archives) != 1:
            raise RuntimeError(f"Expected one zip file, found {len(archives)}.")

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
            raise RuntimeError("Downloaded dataset is incomplete.")

        source_root = find_dataset_root(source_root)
        for source in get_class_directories(source_root):
            destination = raw_dir / source.name
            if destination.exists():
                shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
            shutil.move(str(source), str(destination))

    if not _has_complete_dataset(raw_dir):
        raise RuntimeError("Extracted dataset is invalid.")
    return f"Downloaded and extracted dataset to {raw_dir}."


def get_labels_mapping() -> Dict[str, int]:
    """Map class names to label indices."""
    return build_labels_mapping(RAW_DATA_DIR)


def split_dataset(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = SEED,
) -> Dict[str, str]:
    """Split dataset into train/val/test CSVs."""
    return prepare_dataset(
        RAW_DATA_DIR, SPLITS_DIR, train_ratio, val_ratio, seed
    )


class FruitVegDataset(Dataset):
    """PyTorch Dataset for fruit/veg disease classification."""

    def __init__(self, split_file: str, img_size: int, transform=None):
        self.transform = transform or get_evaluation_transform(
            img_size, convert_color=False
        )
        self.frame = pd.read_csv(split_file)
        required_columns = {"filepath", "label", "class_name"}
        missing_columns = required_columns.difference(self.frame.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in split file: {sorted(missing_columns)}")
        if self.frame.empty:
            raise ValueError(f"Split file is empty: {split_file}")
        self.frame["label"] = self.frame["label"].astype(int)
        if not self.frame["label"].between(0, NUM_CLASSES - 1).all():
            raise ValueError("Labels in split file are out of range.")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.frame.iloc[idx]
        image_path = Path(BASE_DIR) / Path(row["filepath"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            with Image.open(image_path) as image:
                image = convert_rgb(image)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc
        return self.transform(image), int(row["label"])


def build_dataloader(
    split_name: str,
    img_size: int,
    batch_size: int = BATCH_SIZE,
    pin_memory: bool = False,
) -> DataLoader:
    """Create DataLoader for a given split."""
    if split_name.endswith(".csv"):
        split_name = split_name[:-4]
    if split_name not in {"train", "val", "test"}:
        raise ValueError("split_name must be 'train', 'val', or 'test'.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    split_file = Path(SPLITS_DIR) / f"{split_name}.csv"
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}. Run split_dataset() first.")

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
    """Calculate class weights for balancing."""
    return calculate_class_weights(split_name, SPLITS_DIR)


def find_cross_split_duplicates():
    """Check for data leakage between splits."""
    return inspect_cross_split_duplicates(SPLITS_DIR)
