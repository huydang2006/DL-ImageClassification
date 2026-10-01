# Models Directory (Output)

Lưu trữ các file trọng số (`.pth`) hoặc checkpoint của mô hình sau khi huấn luyện (PyTorch).

## Lưu ý

- Nội dung của folder này được bỏ qua bởi `.gitignore`.
- Mỗi model được lưu với tên dạng: `<model_name>_best.pth`.
- Checkpoint được chọn theo validation macro-F1 trước khi được đánh giá trên test set.
