import json
import shutil

import pandas as pd
import pytest
import torch
from PIL import Image

from src.preprocessing import (
    PADDING_COLOR,
    ResizeWithPadding,
    convert_rgb,
    find_cross_split_duplicates,
    find_dataset_root,
    get_class_weights,
    get_evaluation_transform,
    get_training_transform,
    prepare_dataset,
)


def _make_dataset(root, class_count=2, images_per_class=30):
    for label in range(class_count):
        folder = root / f"class_{label}"
        folder.mkdir(parents=True)
        for index in range(images_per_class):
            Image.new("RGB", (12 + index, 8), (label * 100, index * 7, 30)).save(
                folder / f"{index:02}.png"
            )
    return root


def test_find_dataset_root_supports_nested_archive(tmp_path):
    expected = _make_dataset(tmp_path / "raw" / "wrapper" / "dataset")
    (tmp_path / "raw" / "__MACOSX").mkdir()
    assert find_dataset_root(tmp_path / "raw", expected_classes=2) == expected.resolve()


def test_find_dataset_root_rejects_ambiguous_data(tmp_path):
    _make_dataset(tmp_path / "first")
    _make_dataset(tmp_path / "second")
    with pytest.raises(ValueError, match="nhiều dataset"):
        find_dataset_root(tmp_path, expected_classes=2)


def test_rgb_padding_and_transforms():
    transparent = Image.new("RGBA", (40, 20), (255, 0, 0, 0))
    rgb = convert_rgb(transparent)
    assert rgb.mode == "RGB"
    assert rgb.getpixel((0, 0)) == PADDING_COLOR
    padded = ResizeWithPadding(80)(Image.new("RGB", (40, 20), "red"))
    assert padded.size == (80, 80)
    assert padded.getpixel((40, 10)) == PADDING_COLOR
    evaluation = get_evaluation_transform(128)
    assert torch.equal(evaluation(rgb), evaluation(rgb))
    assert evaluation(rgb).shape == (3, 128, 128)
    assert get_training_transform(224)(rgb).shape == (3, 224, 224)


def test_prepare_dataset_filters_errors_and_prevents_leakage(tmp_path):
    root = _make_dataset(tmp_path / "raw" / "wrapper")
    shutil.copyfile(root / "class_0/00.png", root / "class_0/copy.png")
    shutil.copyfile(root / "class_0/01.png", root / "class_1/conflict.png")
    (root / "class_1/broken.png").write_bytes(b"not an image")
    output = tmp_path / "splits"

    paths = prepare_dataset(tmp_path / "raw", output, expected_classes=2)
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    catalog = pd.read_csv(output / "preprocessing_catalog.csv")
    hashes = {
        name: set(
            frame.merge(catalog, on=["filepath", "label", "class_name"])["content_hash"]
        )
        for name, frame in frames.items()
    }
    assert not hashes["train"] & hashes["val"]
    assert not hashes["train"] & hashes["test"]
    assert not hashes["val"] & hashes["test"]
    assert len(pd.read_csv(output / "excluded_invalid_images.csv")) == 1
    assert len(pd.read_csv(output / "excluded_conflicting_duplicates.csv")) == 2
    assert find_cross_split_duplicates(output) == {}

    weights = get_class_weights(split_dir=output, expected_classes=2)
    assert weights.shape == (2,)
    assert torch.isfinite(weights).all()
    report = json.loads((output / "preprocessing_report.json").read_text(encoding="utf-8"))
    assert report["invalid_images"] == 1
    assert report["conflicting_images"] == 2


def test_prepare_dataset_is_reproducible(tmp_path):
    raw = _make_dataset(tmp_path / "raw")
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare_dataset(raw, first, expected_classes=2, seed=42)
    prepare_dataset(raw, second, expected_classes=2, seed=42)
    for name in ("train", "val", "test"):
        pd.testing.assert_frame_equal(
            pd.read_csv(first / f"{name}.csv"),
            pd.read_csv(second / f"{name}.csv"),
        )


def test_invalid_ratios_fail_before_reading_data(tmp_path):
    with pytest.raises(ValueError, match="Tỉ lệ"):
        prepare_dataset(tmp_path / "missing", train_ratio=float("nan"))
