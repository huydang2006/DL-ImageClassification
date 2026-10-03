# Source Code

## Structure

```text
src/
├── config.py        # Config: paths, constants, hyperparams
├── data.py          # Data loading, Dataset, DataLoader
├── preprocessing.py # Validation, splitting, transforms
├── utils.py         # Helpers: seed, plotting, I/O
├── train.py         # Training entry point
├── evaluate.py      # Evaluation entry point
└── models/          # Model definitions
```

## Preprocessing (based on EDA)

- Class imbalance (~14x) → class weights from train split
- Variable image sizes/aspects → resize with padding
- RGB/RGBA/P handling → EXIF transpose, alpha composite, convert to RGB
- Train augment: horizontal flip, ±10° rotation, ColorJitter (0.15)
- No augment on val/test
- SHA-256 grouping prevents leakage

Run:

```bash
python -m src.preprocessing
```

Outputs: `train.csv`, `val.csv`, `test.csv`, catalog, excluded images, report
