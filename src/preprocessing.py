"""Image preprocessing: validation, splitting, transforms.

Original images are never modified; transforms run when DataLoader reads.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torchvision import transforms

from src.config import BASE_DIR, NUM_CLASSES, RAW_DATA_DIR, SEED, SPLITS_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Background close to ImageNet mean, avoiding bright/dark borders.
PADDING_COLOR = (124, 116, 104)


def get_class_directories(root):
    """List class directories, skipping metadata and links."""
    return sorted(
        path for path in Path(root).iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and not getattr(path, "is_junction", lambda: False)()
        and not path.name.startswith(".")
        and path.name != "__MACOSX"
    )


def find_dataset_root(raw_dir=RAW_DATA_DIR, expected_classes=NUM_CLASSES):
    """Find the unique directory containing all expected class folders.

    Handles datasets directly in raw_dir or nested in wrapper folders from ZIPs.
    Raises if multiple complete datasets are found.
    """
    if (isinstance(expected_classes, bool) or not isinstance(expected_classes, int)
            or expected_classes <= 0):
        raise ValueError("expected_classes must be a positive integer.")
    root = Path(raw_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    candidates = []
    pending = [root]
    while pending:
        current = pending.pop()
        children = get_class_directories(current)
        if len(children) == expected_classes and all(
            any(
                image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
                for image in class_dir.rglob("*")
            )
            for class_dir in children
        ):
            candidates.append(current)
        pending.extend(reversed(children))

    # Keep only the deepest root in case wrappers also have enough subfolders.
    candidates = [
        candidate for candidate in candidates
        if not any(candidate in other.parents for other in candidates)
    ]
    if not candidates:
        raise ValueError(
            f"No dataset with {expected_classes} image-bearing classes found in {root}."
        )
    if len(candidates) > 1:
        raise ValueError(
            "Found multiple datasets; specify the correct directory: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def get_labels_mapping(raw_dir=RAW_DATA_DIR, expected_classes=NUM_CLASSES):
    """Map class names to label indices in stable sorted order."""
    root = find_dataset_root(raw_dir, expected_classes)
    return {
        class_dir.name: label
        for label, class_dir in enumerate(get_class_directories(root))
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_rgb(image):
    """Fix EXIF orientation, flatten alpha onto PADDING_COLOR, return RGB."""
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, PADDING_COLOR + (255,))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


class ResizeWithPadding:
    """Fit the whole image into a square frame without distortion or cropping."""

    def __init__(self, image_size):
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        self.image_size = image_size

    def __call__(self, image):
        return ImageOps.pad(
            image,
            (self.image_size, self.image_size),
            method=Image.Resampling.BILINEAR,
            color=PADDING_COLOR,
        )


def get_evaluation_transform(image_size, *, convert_color=True):
    """Deterministic transform for validation/test."""
    operations = [convert_rgb] if convert_color else []
    return transforms.Compose(operations + [
        ResizeWithPadding(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_training_transform(image_size, *, convert_color=True):
    """Training transform with light augmentation from EDA recommendations."""
    operations = [convert_rgb] if convert_color else []
    return transforms.Compose(operations + [
        ResizeWithPadding(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(
            10,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=PADDING_COLOR,
        ),
        # No hue jitter: color is a freshness signal.
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# Legacy aliases kept for backward compatibility.
create_augmentation_pipeline = get_training_transform
get_default_transform = get_evaluation_transform


def scan_dataset(raw_dir=RAW_DATA_DIR, expected_classes=NUM_CLASSES):
    """Decode and hash every image; return valid catalog and invalid list."""
    root = find_dataset_root(raw_dir, expected_classes)
    valid_records = []
    invalid_records = []
    for label, class_dir in enumerate(get_class_directories(root)):
        image_paths = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for path in image_paths:
            try:
                filepath = path.relative_to(Path(BASE_DIR)).as_posix()
            except ValueError:
                filepath = str(path.resolve())
            common = {"filepath": filepath, "label": label, "class_name": class_dir.name}
            try:
                with Image.open(path) as image:
                    image.verify()
                # verify() does not decode pixels, so reopen and load fully.
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    color_mode = image.mode
                valid_records.append({
                    **common,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height,
                    "color_mode": color_mode,
                    "extension": path.suffix.lower(),
                    "file_size_bytes": path.stat().st_size,
                    "content_hash": _sha256(path),
                })
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                invalid_records.append({**common, "error": str(exc)})
        print(f"Scanned {class_dir.name}: {len(image_paths)} images", flush=True)

    valid_columns = [
        "filepath", "label", "class_name", "width", "height", "aspect_ratio",
        "color_mode", "extension", "file_size_bytes", "content_hash",
    ]
    invalid_columns = ["filepath", "label", "class_name", "error"]
    return (
        pd.DataFrame(valid_records, columns=valid_columns),
        pd.DataFrame(invalid_records, columns=invalid_columns),
    )


def prepare_dataset(raw_dir=RAW_DATA_DIR, output_dir=SPLITS_DIR,
                    train_ratio=0.7, val_ratio=0.15, seed=SEED,
                    expected_classes=NUM_CLASSES):
    """Validate data, split by SHA-256 groups, save CSVs and report."""
    if (not all(math.isfinite(value) and value > 0 for value in (train_ratio, val_ratio))
            or train_ratio + val_ratio >= 1):
        raise ValueError("Train and validation ratios must be positive and sum to less than 1.")

    catalog, invalid = scan_dataset(raw_dir, expected_classes)
    if catalog.empty:
        raise ValueError("No valid images found in dataset.")

    label_counts_per_hash = catalog.groupby("content_hash")["label"].nunique()
    conflicting_hashes = set(label_counts_per_hash[label_counts_per_hash > 1].index)
    conflicting = catalog[catalog["content_hash"].isin(conflicting_hashes)].copy()
    clean = catalog[~catalog["content_hash"].isin(conflicting_hashes)].copy()
    if clean["label"].nunique() != expected_classes:
        raise ValueError("Some classes have no valid images left after validation.")

    # One hash = one split unit so identical copies stay in the same split.
    groups = clean.drop_duplicates("content_hash")[["content_hash", "label"]]
    try:
        train_groups, remainder = train_test_split(
            groups,
            test_size=1 - train_ratio,
            stratify=groups["label"],
            random_state=seed,
        )
        validation_share = val_ratio / (1 - train_ratio)
        validation_groups, test_groups = train_test_split(
            remainder,
            test_size=1 - validation_share,
            stratify=remainder["label"],
            random_state=seed,
        )
    except ValueError as exc:
        raise ValueError(
            "Not enough independent image groups per class for the requested ratios."
        ) from exc

    group_sets = {
        "train": set(train_groups["content_hash"]),
        "val": set(validation_groups["content_hash"]),
        "test": set(test_groups["content_hash"]),
    }
    split_frames = {
        name: clean[clean["content_hash"].isin(hashes)].copy()
        for name, hashes in group_sets.items()
    }
    if any(frame["label"].nunique() != expected_classes for frame in split_frames.values()):
        raise ValueError("Each split must contain all classes.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {}
    for name, frame in split_frames.items():
        path = output_dir / f"{name}.csv"
        frame[["filepath", "label", "class_name"]].sort_values(
            ["label", "filepath"]
        ).to_csv(path, index=False)
        split_paths[name] = str(path)

    catalog.to_csv(output_dir / "preprocessing_catalog.csv", index=False)
    invalid.to_csv(output_dir / "excluded_invalid_images.csv", index=False)
    conflicting.to_csv(output_dir / "excluded_conflicting_duplicates.csv", index=False)

    train_counts = split_frames["train"]["label"].value_counts().sort_index()
    class_weights = len(split_frames["train"]) / (expected_classes * train_counts)
    report = {
        "seed": seed,
        "requested_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": 1 - train_ratio - val_ratio,
        },
        "total_files": len(catalog) + len(invalid),
        "valid_images": len(catalog),
        "invalid_images": len(invalid),
        "conflicting_images": len(conflicting),
        "retained_images": len(clean),
        "unique_hashes": len(groups),
        "duplicate_copies": len(clean) - len(groups),
        "class_to_index": get_labels_mapping(raw_dir, expected_classes),
        "split_counts": {name: len(frame) for name, frame in split_frames.items()},
        "train_class_weights": {
            str(label): float(weight) for label, weight in class_weights.items()
        },
        "color_modes": catalog["color_mode"].value_counts().to_dict(),
        "extensions": catalog["extension"].value_counts().to_dict(),
        "image_dimensions": {
            column: {
                "min": int(catalog[column].min()),
                "median": float(catalog[column].median()),
                "max": int(catalog[column].max()),
            }
            for column in ("width", "height")
        },
        "duplicate_policy": (
            "SHA-256 identical images stay in one split; same-hash different-label images excluded."
        ),
    }
    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return split_paths


def get_class_weights(split_name="train", split_dir=SPLITS_DIR,
                      expected_classes=NUM_CLASSES):
    """Inverse-frequency weights from the requested split only."""
    path = Path(split_dir) / f"{split_name.removesuffix('.csv')}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    frame = pd.read_csv(path)
    counts = frame["label"].value_counts().reindex(range(expected_classes), fill_value=0)
    if (counts == 0).any():
        raise ValueError("Each split must contain at least one image per class.")
    weights = len(frame) / (expected_classes * counts.astype(float))
    return torch.tensor(weights.to_numpy(), dtype=torch.float32)


def find_cross_split_duplicates(split_dir=SPLITS_DIR):
    """Return SHA-256 hashes that appear in more than one split."""
    locations = {}
    for split_name in ("train", "val", "test"):
        frame = pd.read_csv(Path(split_dir) / f"{split_name}.csv")
        for filepath in frame["filepath"]:
            path = Path(filepath)
            if not path.is_absolute():
                path = Path(BASE_DIR) / path
            digest = _sha256(path)
            locations.setdefault(digest, {}).setdefault(split_name, []).append(str(filepath))
    return {
        digest: [f"{split}:{path}" for split, paths in splits.items() for path in paths]
        for digest, splits in locations.items()
        if len(splits) > 1
    }


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(RAW_DATA_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(SPLITS_DIR))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = prepare_dataset(
        args.raw_dir, args.output_dir, args.train_ratio, args.val_ratio, args.seed
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
