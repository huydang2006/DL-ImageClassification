# Data Directory

Nơi lưu trữ toàn bộ dữ liệu liên quan đến dự án.

## Subfolders

- `data/raw/` — Dữ liệu gốc tải từ Kaggle (ảnh gốc, belum được xử lý).
- `data/processed/` — Dữ liệu đã được tiền xử lý (resize, chuẩn hóa). Có thể lưu dưới dạng TFRecord/Tensor để tối ưu tốc độ load.
- `data/splits/` — File `.csv` chứa danh sách ảnh được chia thành train/val/test.

## Lưu ý

- Nội dung thực tế của `data/` được bỏ qua bởi `.gitignore`.
- Để tải dữ liệu tự động từ Kaggle, cấu hình Kaggle API credentials rồi chạy:

  ```bash
  python -c "from src.data import download_dataset; print(download_dataset())"
  ```

  Nếu dataset đã tồn tại đầy đủ, download sẽ được bỏ qua. File zip tạm sẽ được xóa sau khi giải nén.

- Hoặc tải dataset thủ công về `data/raw/`, sau đó chạy:

  ```bash
  python -c "from src.data import split_dataset; split_dataset()"
  ```

- Các split dùng stratified sampling với seed mặc định `42`.
