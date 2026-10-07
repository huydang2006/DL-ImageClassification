# Copilot Instructions

## Project purpose and architecture

This repository compares three PyTorch classifiers for a 28-class fruit and
vegetable freshness dataset (`<produce>__Healthy` and `<produce>__Rotten`):

- **M1** (`src/models/simple_nn.py`) is a fully connected baseline using
  `128 x 128` RGB images.
- **M2** (`src/models/cnn.py`) is a convolutional model using `128 x 128`
  images, three convolution/pooling blocks, adaptive average pooling, and a
  dropout classifier.
- **M3** (`src/models/transfer.py`) is a transfer-learning model using a
  replaceable torchvision backbone (MobileNetV2 by default) and `224 x 224`
  images. Training freezes the backbone first and then fine-tunes it.

The normal pipeline is:

1. `src.data.download_dataset()` obtains the Kaggle archive and extracts its
   class folders into `data/raw/`.
2. `src.preprocessing.prepare_dataset()` validates and fully decodes images,
   records metadata, excludes invalid images and same-hash conflicting labels,
   and writes `data/splits/{train,val,test}.csv` plus reports. Splitting is
   stratified at the SHA-256 content-hash group level so identical files cannot
   cross splits.
3. `src.data.FruitVegDataset` reads split CSVs and converts images to RGB.
   Training uses light augmentation; validation and test use deterministic
   transforms. All transforms preserve aspect ratio with padding and apply
   ImageNet normalization.
4. `src.train` builds a model, uses inverse-frequency class-weighted
   cross-entropy and Adam, monitors validation macro-F1, reduces the learning
   rate on plateaus, early-stops, and saves the best `state_dict` under
   `models/<model>_best.pth`.
5. `src.evaluate` loads a checkpoint and writes JSON metrics under
   `results/metrics/` and a confusion-matrix image under `results/plots/`.
   `notebooks/model_comparison.ipynb` consumes these generated artifacts; it
   does not retrain models.

`src/config.py` is the shared source of truth for paths, class count, model
image sizes, seed, batch size, and training defaults. Paths are rooted from
the repository location, so run commands from the repository root.

## Environment and commands

Install the default CPU/CUDA-compatible dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

For Intel Arc/Xe XPU builds, install the matching pinned PyTorch wheels:

```powershell
python -m pip install -r requirements-xpu.txt --extra-index-url https://download.pytorch.org/whl/xpu
```

Prepare data (Kaggle credentials are required for the download command):

```powershell
python -c "from src.data import download_dataset; print(download_dataset())"
python -m src.preprocessing
```

Train and evaluate one of `M1`, `M2`, or `M3`:

```powershell
python -m src.train --model M1
python -m src.train --model M1 --epochs 10 --batch_size 64 --lr 0.001
python -m src.evaluate --model M1
```

Use `--device cpu`, `--device cuda`, or `--device xpu` to override automatic
device selection. M3 training downloads pretrained torchvision weights when
`pretrained=True`; model-only tests deliberately use `pretrained=False` so
they do not require a network connection.

Run the complete test suite:

```powershell
python -m pytest -q
```

Run one file, one test, or the smoke training test:

```powershell
python -m pytest tests/test_preprocessing.py -q
python -m pytest tests/test_models.py::test_transfer_model_output_shape_without_download -q
python -m pytest tests/test_training_smoke.py::test_training_loop_smoke -q
```

There is no repository-defined lint or formatter command/configuration.

## Codebase-specific conventions

- Keep model selection consistent with the uppercase choices `M1`, `M2`, and
  `M3`; use `src.config.MODEL_CHOICES` and `get_model_image_size()` instead of
  duplicating those mappings.
- Models return raw logits of shape `(batch, 28)`. Do not add softmax to model
  `forward()` methods because training uses `CrossEntropyLoss`.
- Use `src.utils.get_device()` for CPU/CUDA/XPU selection and `set_seed(42)`
  for reproducibility. Explicitly requested unavailable devices should raise,
  while `auto` falls back to CPU.
- Preserve the preprocessing invariants: EXIF orientation is corrected,
  RGBA/P images are composited onto `PADDING_COLOR`, aspect ratio is preserved
  by padding rather than cropping, and validation/test transforms do not
  augment.
- Do not split individual image files independently when changing dataset
  preparation. The SHA-256 grouping and exclusion of same-hash images with
  conflicting labels are required leakage protections.
- Class labels come from stable sorted class-directory names. Keep split CSV
  columns (`filepath`, `label`, `class_name`) compatible with
  `FruitVegDataset`.
- Checkpoint files contain only a PyTorch model `state_dict`; load them into
  the matching architecture with `map_location` and evaluate in `.eval()`
  mode.
- Generated data, checkpoints, metrics, plots, and reports are intentionally
  gitignored. Keep the tracked README placeholders and do not add generated
  artifacts to source changes.
- When adding tests, use `tmp_path` and monkeypatch module-level data/model
  directories for isolated temporary fixtures. Tests that instantiate M3
  should disable pretrained weights unless downloading weights is the behavior
  under test.
- Preserve the CLI module entry points (`python -m src.preprocessing`,
  `python -m src.train`, and `python -m src.evaluate`) when changing scripts.
  They configure stdout/stderr for UTF-8 and expose argparse options intended
  for Windows as well as other platforms.

## Related documentation

- `README.md` describes the dataset, experiment stages, installation, and
  standard commands.
- `docs/MODEL_GUIDE.md` documents the intended architecture and training
  rationale for M1, M2, and M3.
- `src/README.md`, `src/models/README.md`, `data/README.md`, and
  `tests/README.md` document module-level contracts and outputs.
