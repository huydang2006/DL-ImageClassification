# Tests Directory

Chứa các unit test cho các thành phần của dự án.

## Kế hoạch test

- `test_data.py` — Kiểm tra mapping, split và pipeline load dữ liệu.
- `test_models.py` — Kiểm tra output shape và forward pass của M1/M2/M3.
- `test_utils.py` — Kiểm tra seed và save/load checkpoint.

Chạy toàn bộ test từ thư mục gốc:

```bash
pytest -q
```
