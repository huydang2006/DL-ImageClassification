"""Model evaluation and report generation.
Usage: python -m src.evaluate --model M1|M2|M3
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import (
    IMG_SIZE_M1,
    IMG_SIZE_M2,
    IMG_SIZE_M3,
    MODELS_DIR,
    NUM_CLASSES,
    RESULTS_DIR,
)
from src.data import build_dataloader, get_labels_mapping
from src.train import build_model
from src.utils import get_device, save_metrics


def evaluate(model_name, checkpoint_path=None, device="auto"):
    """Load trained model and evaluate on the test set.

    Saves metrics JSON and confusion matrix plot under results/.
    """
    image_size = {
        "M1": IMG_SIZE_M1,
        "M2": IMG_SIZE_M2,
        "M3": IMG_SIZE_M3,
    }[model_name]
    checkpoint = Path(checkpoint_path or Path(MODELS_DIR) / f"{model_name}_best.pth")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = get_device(device)
    model = build_model(model_name).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    loader = build_dataloader(
        "test",
        image_size,
        pin_memory=device.type in {"cuda", "xpu"},
    )
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

    mapping = get_labels_mapping()
    class_names = [
        class_name
        for class_name, _ in sorted(mapping.items(), key=lambda item: item[1])
    ]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "classification_report": report,
    }
    metrics_path = Path(RESULTS_DIR) / "metrics" / f"{model_name}_evaluation.json"
    save_metrics(metrics, str(metrics_path))

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    plots_dir = Path(RESULTS_DIR) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, cmap="Blues", cbar=True)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{model_name} confusion matrix")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close()
    return metrics


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate fruit and vegetable classifier")
    parser.add_argument("--model", type=str, default="M1", choices=["M1", "M2", "M3"])
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "xpu"],
        help="Compute device; auto prefers CUDA, then XPU, then CPU.",
    )
    args = parser.parse_args()
    metrics = evaluate(args.model, device=args.device)
    print(f"Model: {metrics['model']}")
    print(f"Device: {metrics['device']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(
        "Metrics: "
        f"{Path(RESULTS_DIR) / 'metrics' / f'{args.model}_evaluation.json'}"
    )
    print(
        "Confusion matrix: "
        f"{Path(RESULTS_DIR) / 'plots' / f'{args.model}_confusion_matrix.png'}"
    )


if __name__ == "__main__":
    main()
