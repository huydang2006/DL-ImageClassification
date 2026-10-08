# DL-ImageClassification

## Comparative Deep Learning Models for Fruit and Vegetable Freshness Classification

This project builds and evaluates Deep Learning models to classify fruit and vegetable freshness (Healthy vs Rotten) from images.

## 1. Problem

- Classify fruit/vegetable health state: `Healthy` or `Rotten`
- 14 fruit/vegetable types × 2 states = 28 classes

## 2. Data

- **Dataset**: Fruit and Vegetable Disease (Healthy vs Rotten)
- **Source**: <https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten>
- **Size**: ~29,000 images

## 3. Models

| Model | Description |
| --- | --- |
| **M1** | Simple CNN - Two convolution/pooling blocks and a fully connected classifier |
| **M2** | Deep CNN - Multiple conv/pool/regularization blocks |
| **M3** | Transfer Learning (MobileNetV2) - 5 epochs frozen, 5 epochs fine-tuned |

## 4. Pipeline

1. Download from Kaggle
2. Preprocess: resize, normalize, augmentation
3. Stratified split (70/15/15) grouped by SHA-256 to prevent leakage
4. Train M1, M2, M3
5. Evaluate: Accuracy, Precision/Recall/F1, Confusion Matrix
6. Compare results

## 5. Project Structure

```text
DL-ImageClassification/
├── data/           # Raw/processed data (gitignored)
├── notebooks/      # EDA, demos
├── src/            # Source code
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── utils.py
│   ├── train.py
│   ├── evaluate.py
│   └── models/
├── models/         # Checkpoints (gitignored)
├── results/        # Metrics, plots
└── tests/          # Unit tests
```

## 6. Installation

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

For Intel XPU support:

```bash
pip install -r requirements-xpu.txt --extra-index-url https://download.pytorch.org/whl/xpu
```

## 7. Usage

### Download & Prepare Data

```bash
# Download from Kaggle (requires kaggle.json credentials)
python -c "from src.data import download_dataset; print(download_dataset())"

# Create splits (70/15/15)
python -m src.preprocessing
```

### Train

```bash
python -m src.train --model M1   # Simple CNN
python -m src.train --model M2   # Deep CNN
python -m src.train --model M3   # Transfer Learning

# Override params
python -m src.train --model M1 --epochs 10 --batch_size 64 --lr 0.001
```

### Evaluate

```bash
python -m src.evaluate --model M1
```

Checkpoints saved to `models/<model>_best.pth`. Metrics and confusion matrices in `results/`.

### Test

```bash
pytest -q
```

## 8. Requirements

- Python 3.9+
- PyTorch + torchvision
- See `requirements.txt` for full list
