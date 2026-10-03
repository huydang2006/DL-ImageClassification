"""Project configuration."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Data parameters
NUM_CLASSES = 28
IMG_SIZE_M1 = 128
IMG_SIZE_M2 = 128
IMG_SIZE_M3 = 224
BATCH_SIZE = 64
SEED = 42

# Training parameters
LEARNING_RATE = 1e-3
EPOCHS = 10
M3_FREEZE_EPOCHS = 5
M3_FINETUNE_EPOCHS = 5
LR_SCHEDULER_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 5
