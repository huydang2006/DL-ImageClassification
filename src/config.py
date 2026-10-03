"""Project configuration."""

import os
from pathlib import Path

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_NUM_CLASSES = 28


def _get_class_directories(root):
    """List class directories, skipping metadata and links."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        path for path in root.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and not getattr(path, "is_junction", lambda: False)()
        and not path.name.startswith(".")
        and path.name != "__MACOSX"
    )


def infer_num_classes(raw_dir=RAW_DATA_DIR, default=DEFAULT_NUM_CLASSES):
    """Infer the dataset class count from the raw directory when available.

    Falls back to the default class count when the data is not yet downloaded or
    the raw directory is not populated.
    """
    root = Path(raw_dir)
    if not root.exists():
        return default

    candidates = []
    pending = [root]
    while pending:
        current = pending.pop()
        children = _get_class_directories(current)
        if children and all(
            any(
                image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
                for image in class_dir.rglob("*")
            )
            for class_dir in children
        ):
            candidates.append(current)
        pending.extend(reversed(children))

    candidates = [
        candidate for candidate in candidates
        if not any(candidate in other.parents for other in candidates)
    ]
    if not candidates:
        return default
    return len(_get_class_directories(candidates[0]))


# Data parameters
NUM_CLASSES = infer_num_classes()
IMG_SIZE_M1 = 128
IMG_SIZE_M2 = 128
IMG_SIZE_M3 = 224
MODEL_IMAGE_SIZES = {"M1": IMG_SIZE_M1, "M2": IMG_SIZE_M2, "M3": IMG_SIZE_M3}
MODEL_CHOICES = tuple(MODEL_IMAGE_SIZES)
BATCH_SIZE = 64
SEED = 42


def get_model_image_size(model_name: str) -> int:
    """Return the expected image size for a model family."""
    if model_name not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Expected one of {MODEL_CHOICES}, got: {model_name}")
    return MODEL_IMAGE_SIZES[model_name]


# Training parameters
LEARNING_RATE = 1e-3
EPOCHS = 10
M3_FREEZE_EPOCHS = 5
M3_FINETUNE_EPOCHS = 5
LR_SCHEDULER_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 5
