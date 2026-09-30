# DL-ImageClassification

## Comparative Deep Learning Models for Fruit and Vegetable Freshness Classification from Images

Dự án xây dựng và đánh giá các mô hình Deep Learning để phân loại tình trạng **Healthy** và **Rotten** của rau củ quả từ ảnh.

## 1. Bài toán
- Phân loại độ tươi/sức khỏe của rau củ quả từ ảnh thành 2 trạng thái:
  - `Healthy`
  - `Rotten`
- Bộ dữ liệu gồm 14 loại rau củ quả, mỗi loại có 2 trạng thái (Healthy/Rotten) tương ứng 28 lớp ảnh.

## 2. Dữ liệu
- Dataset: **Fruit and Vegetable Disease (Healthy vs Rotten)**
- Nguồn: https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten
- Quy mô xấp xỉ: ~29.000 ảnh

## 3. Mục tiêu mô hình
Dự án so sánh 3 hướng tiếp cận chính:

### M1: Simple NN
- Mạng Neural Network cơ bản để làm baseline.
- Sử dụng đặc trưng ảnh đã được vector hóa/làm phẳng.

### M2: Complex NN (Deep CNN)
- Mạng CNN sâu hơn (nhiều block convolution + pooling + regularization).
- Mục tiêu cải thiện khả năng trích xuất đặc trưng không gian so với M1.

### M3: Transfer Learning + Fine-tuning
- Dùng MobileNetV2 pretrained.
- Huấn luyện theo 2 giai đoạn: 5 epoch đóng băng backbone, sau đó 5 epoch
  fine-tuning với learning rate nhỏ hơn.

## 4. Quy trình thực hiện
1. **Collect data** từ Kaggle dataset.
2. Tiền xử lý ảnh: resize, normalize, augmentation (nếu cần).
3. Chia dữ liệu train/validation/test theo nhóm ảnh có cùng SHA-256 để tránh leakage.
4. Huấn luyện riêng cho M1, M2, M3.
5. Đánh giá và so sánh bằng các chỉ số:
   - Accuracy
   - Precision / Recall / F1-score
   - Confusion Matrix
6. Tổng hợp kết quả và rút ra mô hình tối ưu theo hiệu năng/chi phí tính toán.

## 5. Kết quả mong đợi
- Có baseline rõ ràng từ M1.
- Chứng minh hiệu quả tăng dần của M2 và M3.
- Đề xuất mô hình phù hợp nhất cho bài toán nhận diện độ tươi của rau củ quả trong thực tế.

---

## 6. Cấu trúc dự án

```
DL-ImageClassification/
├── data/                   # Dữ liệu (gitignore) + data/README.md
├── notebooks/              # EDA, demo + notebooks/README.md
├── src/                    # Code nguồn + src/README.md
│   ├── config.py           # Hằng số, paths, tham số
│   ├── data.py             # Tải dữ liệu, Dataset và DataLoader
│   ├── preprocessing.py    # Kiểm tra, chia tập và transform ảnh
│   ├── utils.py            # Seed, plot utilities
│   ├── train.py            # Entry point huấn luyện
│   ├── evaluate.py         # Entry point đánh giá
│   └── models/             # M1/M2/M3 + models/README.md
├── models/                 # Trọng số model (gitignore) + models/README.md
├── results/                # Metrics, plots, reports + results/README.md
└── tests/                  # Unit test + tests/README.md
```

## 7. Cài đặt

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

`requirements.txt` là profile mặc định, phù hợp với CPU và không khóa người dùng
vào một loại GPU cụ thể. Nếu máy dùng Intel Arc/Xe và muốn chạy bằng XPU, cài
profile tùy chọn bằng wheel chính thức của PyTorch:

```bash
pip install -r requirements-xpu.txt --extra-index-url https://download.pytorch.org/whl/xpu
```

Kiểm tra XPU:

```bash
python -c "import torch; print(torch.xpu.is_available()); print(torch.xpu.device_count())"
```

Các lệnh train/evaluate mặc định dùng `auto`: ưu tiên CUDA, sau đó XPU, rồi CPU.
Có thể chọn rõ thiết bị bằng `--device cpu`, `--device cuda` hoặc `--device xpu`.

## 8. Cách dùng

### Chuẩn bị dữ liệu

Có thể tải dataset tự động bằng Kaggle CLI:

```bash
python -c "from src.data import download_dataset; print(download_dataset())"
```

Nếu `data/raw/` đã chứa đủ 28 class có ảnh hợp lệ, lệnh trên sẽ bỏ qua download. Nếu chưa có, lệnh sẽ tải dataset, giải nén vào `data/raw/`, xóa file `.zip` tạm và kiểm tra lại cấu trúc.

Trước lần tải đầu tiên, cần cấu hình Kaggle API credentials. Xem hướng dẫn chính thức tại:
https://www.kaggle.com/docs/api

Trên Windows, file credentials thường được đặt tại:

```text
%USERPROFILE%\.kaggle\kaggle.json
```

Sau khi tải dataset, kiểm tra ảnh và tạo các split stratified:

```bash
python -m src.preprocessing
```

Lệnh này tạo:

- `data/splits/train.csv` (70%)
- `data/splits/val.csv` (15%)
- `data/splits/test.csv` (15%)

Nếu dataset chứa các file byte-identical ở nhiều class, các file có nhãn mâu thuẫn
được giữ nguyên trong `data/raw/` nhưng loại khỏi split và ghi vào
`data/splits/excluded_conflicting_duplicates.csv`.

### Huấn luyện

```bash
python -m src.train --model M1        # Simple NN (baseline)
python -m src.train --model M2        # Deep CNN
python -m src.train --model M3        # Transfer learning
```

Có thể ghi đè tham số:

```bash
python -m src.train --model M1 --epochs 10 --batch_size 64 --lr 0.001
```

### Đánh giá

```bash
python -m src.evaluate --model M1
```

Checkpoint được lưu tại `models/<model>_best.pth`. Checkpoint được chọn theo
validation macro-F1; metrics và confusion matrix được lưu trong `results/`.

### Kiểm thử

```bash
pytest -q
```

Framework: **PyTorch** + **torchvision**.
