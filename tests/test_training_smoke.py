from pathlib import Path

import pandas as pd
import torch
from PIL import Image

import src.data as data_module
import src.preprocessing as preprocessing_module
import src.train as train_module
from src.config import NUM_CLASSES


def _create_split_csvs(tmp_path: Path, split_name: str, labels: int = 2, images_per_label: int = 2):
    rows = []
    for label in range(labels):
        for index in range(images_per_label):
            image_path = tmp_path / f"{split_name}_{label}_{index}.png"
            Image.new("RGB", (32, 32), (label * 50 + index * 10, index * 20, 255)).save(image_path)
            rows.append({
                "filepath": str(image_path),
                "label": int(label),
                "class_name": f"class_{label}",
            })
    pd.DataFrame(rows).to_csv(tmp_path / f"{split_name}.csv", index=False)
    return tmp_path / f"{split_name}.csv"


def test_models_forward_smoke():
    for model_name, input_size in {"M1": (3, 128, 128), "M2": (3, 128, 128), "M3": (3, 224, 224)}.items():
        model = train_module.build_model(model_name, pretrained=False)
        batch = torch.randn(2, *input_size)
        logits = model(batch)
        assert logits.shape == (2, NUM_CLASSES)


def test_training_loop_smoke(tmp_path, monkeypatch):
    _create_split_csvs(tmp_path, "train", labels=2, images_per_label=2)
    _create_split_csvs(tmp_path, "val", labels=2, images_per_label=2)

    monkeypatch.setattr(train_module, "SPLITS_DIR", str(tmp_path))
    monkeypatch.setattr(data_module, "SPLITS_DIR", str(tmp_path))
    monkeypatch.setattr(preprocessing_module, "SPLITS_DIR", str(tmp_path))
    monkeypatch.setattr(train_module, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(train_module, "RESULTS_DIR", str(tmp_path / "results"))

    history = train_module.train("M1", epochs=1, batch_size=2, lr=1e-3, device="cpu")
    assert len(history) == 1
    assert history[0]["train_accuracy"] >= 0.0
    assert history[0]["val_accuracy"] >= 0.0
