"""M2: Deep CNN — nâng cấp so với Simple NN.

Kiến trúc đề xuất (placeholder):
    Block 1: Conv2d(3, 32) -> BatchNorm2d -> ReLU -> MaxPool2d
    Block 2: Conv2d(32, 64) -> BatchNorm2d -> ReLU -> MaxPool2d
    Block 3: Conv2d(64, 128) -> BatchNorm2d -> ReLU -> MaxPool2d
    AdaptiveAvgPool2d(1) -> Flatten -> Dropout(0.5) -> Linear(128, NUM_CLASSES)

Regularizations đề xuất: BatchNorm2d, Dropout, L2 (weight_decay).
"""

import torch.nn as nn

from src.config import IMG_SIZE_M2


class DeepCNN(nn.Module):
    """
    Mạng CNN sâu.

    TODO: dùng nn.Sequential hoặc custom module để build kiến trúc trên.
    """

    def __init__(self, num_classes: int = 28):
        super(DeepCNN, self).__init__()
        # TODO: self.features = nn.Sequential([...])
        #       self.classifier = nn.Sequential([...])
        return

    def forward(self, x):
        """
        Args:
            x: tensor [batch, 3, IMG_SIZE_M2, IMG_SIZE_M2]
        Returns:
            logits: tensor [batch, num_classes]
        """
        # x = self.features(x)
        # x = self.classifier(x)
        # return x
        return NotImplemented
