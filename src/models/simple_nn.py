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

    TODO:
    - Tính input_size = IMG_SIZE_M1 * IMG_SIZE_M1 * 3
    - xây dựng Sequential các Linear layer theo kiến trúc trên.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(SimpleNN, self).__init__()
        # TODO: self.net = nn.Sequential(...)
        return

    def forward(self, x):
        """
        Args:
            x: tensor [batch, 3, IMG_SIZE_M1, IMG_SIZE_M1]
        Returns:
            logits: tensor [batch, num_classes]
        """
        # x = x.view(x.size(0), -1)  # flatten
        # logits = self.net(x)
        # return logits
        return NotImplemented
