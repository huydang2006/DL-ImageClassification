# Source Code Directory

Chứa toàn bộ mã nguồn Python của dự án.

## Cấu trúc

```
src/
├── config.py        # Cấu hình chung: paths, hằng số, tham số huấn luyện
├── data.py          # Tải dữ liệu, Dataset và DataLoader
├── preprocessing.py # Kiểm tra, chia tập và transform ảnh
├── utils.py         # Các hàm tiện ích chung (seed, vẽ ảnh, v.v.)
├── train.py         # Script huấn luyện mô hình chính
└── evaluate.py      # Script đánh giá & sinh báo cáo
```

## Cách dùng

Tất cả các module đều được thiết kế để có thể `import` trực tiếp.

## Tiền xử lý dữ liệu

Các lựa chọn trong `preprocessing.py` dựa trên kết quả của `notebooks/EDA.ipynb`:

- Dataset mất cân bằng 14,65 lần nên class weights được tính từ train split.
- Ảnh có nhiều kích thước và tỉ lệ nên resize kèm padding, không kéo méo ảnh.
- Ảnh RGB/RGBA/P được chỉnh hướng EXIF, ghép alpha và chuyển thống nhất sang RGB.
- Train dùng lật ngang, xoay tối đa 10 độ và ColorJitter nhẹ. Validation/test
  không dùng augmentation ngẫu nhiên.
- Ảnh giống hệt theo SHA-256 luôn nằm cùng split; cùng ảnh nhưng khác nhãn bị loại.

Chạy pipeline tiền xử lý từ thư mục gốc:

```bash
python -m src.preprocessing
```

Kết quả được lưu trong `data/splits/`: ba CSV split, catalog ảnh, danh sách ảnh
lỗi, danh sách duplicate mâu thuẫn và `preprocessing_report.json`. Ảnh gốc không
bị sửa hoặc sao chép; resize và augmentation chạy khi DataLoader đọc ảnh.
