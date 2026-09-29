"""M1: Simple Neural Network — baseline cho bài toán phân loại ảnh rau củ quả.

Kiến trúc đề xuất (placeholder):
    Flatten -> Linear(788, 256) -> ReLU -> Dropout(0.3)
    -> Linear(256, 128) -> ReLU -> Dropout(0.3)
    -> Linear(128, NUM_CLASSES) -> Softmax

Lưu ý:
- Đây mô hình cơ bản chỉ để làm baseline.
- Input phải được flatten thành vector 1D trước khi đưa vào Linear.
"""

import torch.nn as nn

from src.config import NUM_CLASSES, IMG_SIZE_M1


class SimpleNN(nn.Module):
    """
    Mạng Neural Network cơ bản.

    The input size is derived from the configured image dimensions.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(SimpleNN, self).__init__()
        input_size = IMG_SIZE_M1 * IMG_SIZE_M1 * 3
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: tensor [batch, 3, IMG_SIZE_M1, IMG_SIZE_M1]
        Returns:
            logits: tensor [batch, num_classes]
        """
        return self.net(x)
