# Models Directory

Chứa các định nghĩa mô hình Deep Learning cho 3 hướng tiếp cận:

## Các mô hình

- `simple_nn.py` — **M1**: Mạng Neural Network cơ bản (baseline).
- `cnn.py` — **M2**: Mạng CNN sâu (nhiều block convolution + pooling).
- `transfer.py` — **M3**: Transfer Learning + Fine-tuning dùng backbone pretrained (ResNet/MobileNetV2).

## Cách import

```python
from src.models import SimpleNN, DeepCNN, TransferModel

model = TransferModel(num_classes=28)
```

Mỗi mô hình được thiết kế để trả về một `tf.keras.Model` chuẩn.
