"""Cấu hình chung cho dự án phân loại ảnh rau củ quả."""

# ---------------------------------------------------------------------------
# Đường dẫn thư mục
# ---------------------------------------------------------------------------
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # thư mục gốc (DL-ImageClassification)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------------------
# Tham số dữ liệu
# ---------------------------------------------------------------------------
NUM_CLASSES = 28          # 14 loại rau/củ × 2 trạng thái (Healthy/Rotten)
IMG_SIZE_M1 = 128         # kích thước ảnh dùng cho M1 & M2
IMG_SIZE_M3 = 224         # kích thước ảnh dùng cho M3 (transfer learning)
BATCH_SIZE = 32
SEED = 42

# ---------------------------------------------------------------------------
# Tham số huấn luyện (placeholder)
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-3       # placeholder — sẽ cấu hình chi tiết trong lúc train
EPOCHS = 50               # placeholder
