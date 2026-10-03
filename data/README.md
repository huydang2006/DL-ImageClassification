# Data Directory

Stores all project data.

## Subfolders

- `data/raw/` — Original images from Kaggle
- `data/processed/` — Preprocessed data (resized, normalized)
- `data/splits/` — CSV files for train/val/test splits

## Notes

- `data/` contents are gitignored
- Auto-download from Kaggle (configure `kaggle.json` first):

  ```bash
  python -c "from src.data import download_dataset; print(download_dataset())"
  ```

- Manual download: place in `data/raw/` then run:

  ```bash
  python -c "from src.data import split_dataset; split_dataset()"
  ```

- Stratified splits with seed=42
