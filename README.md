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
- Dùng backbone pretrained (ví dụ: MobileNetV2/ResNet/EfficientNet).
- Huấn luyện theo 2 giai đoạn:
  1. Transfer learning (đóng băng backbone).
  2. Fine-tuning (mở một phần/tất cả backbone để tinh chỉnh).

## 4. Quy trình thực hiện
1. **Collect data** từ Kaggle dataset.
2. Tiền xử lý ảnh: resize, normalize, augmentation (nếu cần).
3. Chia dữ liệu train/validation/test.
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
│   ├── data.py             # Dataset pipeline
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

## 8. Cách dùng

```bash
# Huấn luyện
python -m src.train --model M1        # Simple NN (baseline)
python -m src.train --model M2        # Deep CNN
python -m src.train --model M3        # Transfer learning

# Đánh giá
python -m src.evaluate --model M1
```

Framework: **PyTorch** + **torchvision**.
