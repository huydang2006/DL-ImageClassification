from pathlib import Path

import pandas as pd

from src.config import IMG_SIZE_M1
from src.data import FruitVegDataset, download_dataset, get_labels_mapping


def test_labels_mapping_matches_dataset():
    mapping = get_labels_mapping()
    assert len(mapping) == 28
    assert sorted(mapping.values()) == list(range(28))


def test_download_skips_existing_dataset():
    message = download_dataset()
    assert "bỏ qua tải xuống" in message


def test_validation_dataset_returns_tensor_and_label():
    split_path = Path("data/splits/val.csv")
    dataset = FruitVegDataset(str(split_path), IMG_SIZE_M1)
    image, label = dataset[0]
    assert tuple(image.shape) == (3, IMG_SIZE_M1, IMG_SIZE_M1)
    assert 0 <= label < 28


def test_splits_are_disjoint_and_complete():
    frames = [
        pd.read_csv(Path("data/splits") / f"{name}.csv")
        for name in ("train", "val", "test")
    ]
    paths = [set(frame["filepath"]) for frame in frames]
    assert not (paths[0] & paths[1])
    assert not (paths[0] & paths[2])
    assert not (paths[1] & paths[2])
    assert sum(len(items) for items in paths) == 29291
