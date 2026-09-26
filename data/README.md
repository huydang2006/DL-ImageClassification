# Data Directory

Nơi lưu trữ toàn bộ dữ liệu liên quan đến dự án.

## Subfolders

- `data/raw/` — Dữ liệu gốc tải từ Kaggle (ảnh gốc, belum được xử lý).
- `data/processed/` — Dữ liệu đã được tiền xử lý (resize, chuẩn hóa). Có thể lưu dưới dạng TFRecord/Tensor để tối ưu tốc độ load.
- `data/splits/` — File `.csv` hoặc `.txt` chứa danh sách ảnh được chia thành train/val/test.

## Lưu ý

- Nội dung thực tế của `data/` được bỏ qua bởi `.gitignore`.
- Để tải dữ liệu, chạy script trong `src/data.py` hoặc dùng Kaggle API.
