# Source Code Directory

Chứa toàn bộ mã nguồn Python của dự án.

## Cấu trúc

```
src/
├── config.py        # Cấu hình chung: paths, hằng số, tham số huấn luyện
├── data.py          # Pipeline dữ liệu: tải, tiền xử lý, load ảnh
├── utils.py         # Các hàm tiện ích chung (seed, vẽ ảnh, v.v.)
├── train.py         # Script huấn luyện mô hình chính
└── evaluate.py      # Script đánh giá & sinh báo cáo
```

## Cách dùng

Tất cả các module đều được thiết kế để có thể `import` trực tiếp.
