# Model Definitions

Contains 3 model architectures:

- `simple_nn.py` — **M1**: Basic NN (baseline)
- `cnn.py` — **M2**: Deep CNN (conv/pool blocks)
- `transfer.py` — **M3**: Transfer Learning (MobileNetV2/ResNet/EfficientNet)

## Usage

```python
from src.models import SimpleNN, DeepCNN, TransferModel

model = TransferModel(num_classes=28)
```

All models return standard `torch.nn.Module`
